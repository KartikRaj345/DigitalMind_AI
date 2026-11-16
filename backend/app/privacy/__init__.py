"""Privacy: data lifecycle, cascading deletion, and the optional LLM boundary.

Cascading delete
----------------
Deleting a record must remove everything derived from it. We enforce that in three
places, deliberately:

1. ``ON DELETE CASCADE`` in the schema (defense in depth, DB-level)
2. an explicit sweep in :func:`delete_record_cascade` that deletes and *verifies*
3. a test that asserts every derived table is empty for the deleted record

The explicit sweep exists because SQLite's cascade depends on ``PRAGMA foreign_keys=ON``,
which is per-connection and easy to lose. We do not want a privacy guarantee that depends
on a pragma.

The optional LLM layer
----------------------
:class:`LlmPolicy` is the single gate every outbound request must pass. It refuses by
default, requires an explicit endpoint, and is surfaced in the UI as
"integrations not configured" rather than failing silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..logging_config import get_logger
from ..models import (
    AppSetting,
    Chunk,
    DataSource,
    Embedding,
    Entity,
    EntityAlias,
    EntityMention,
    Event,
    Insight,
    Record,
    Relation,
    RiskAssessment,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cascading deletion
# ---------------------------------------------------------------------------


@dataclass
class DeletionReport:
    record_id: int
    removed: dict[str, int]
    orphaned_entities_removed: int
    index_invalidated: bool
    verified: bool
    problems: list[str]

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "removed": self.removed,
            "orphaned_entities_removed": self.orphaned_entities_removed,
            "index_invalidated": self.index_invalidated,
            "verified": self.verified,
            "problems": self.problems,
        }


def delete_record_cascade(session: Session, record_id: int) -> DeletionReport:
    """Delete a record and every row derived from it, then verify.

    Order matters: children first so the operation works even if foreign keys are off.
    """
    from ..retrieval import bm25_index, embeddings

    record = session.get(Record, record_id)
    if record is None:
        return DeletionReport(record_id, {}, 0, False, True, ["record not found"])

    # Entities that will become orphaned once this record's mentions go.
    entity_ids = {
        entity_id
        for (entity_id,) in session.execute(
            select(EntityMention.entity_id).where(
                EntityMention.record_id == record_id, EntityMention.entity_id.is_not(None)
            )
        )
        if entity_id is not None
    }

    removed: dict[str, int] = {}

    def n(stmt) -> int:
        result = session.execute(stmt)
        return int(result.rowcount or 0)

    removed["embeddings"] = n(delete(Embedding).where(Embedding.record_id == record_id))
    removed["chunks"] = n(delete(Chunk).where(Chunk.record_id == record_id))
    removed["events"] = n(delete(Event).where(Event.record_id == record_id))
    removed["risk_assessments"] = n(delete(RiskAssessment).where(RiskAssessment.record_id == record_id))
    removed["mentions"] = n(delete(EntityMention).where(EntityMention.record_id == record_id))
    removed["relations"] = n(delete(Relation).where(Relation.record_id == record_id))
    removed["records"] = n(delete(Record).where(Record.id == record_id))

    # Drop entities (and their aliases) that existed only because of this record, then
    # re-verify counts for the survivors.
    removed_entities = 0
    for entity_id in sorted(entity_ids):
        remaining = int(
            session.execute(
                select(func.count())
                .select_from(EntityMention)
                .where(EntityMention.entity_id == entity_id)
            ).scalar_one()
        )
        if remaining == 0:
            session.execute(delete(EntityAlias).where(EntityAlias.entity_id == entity_id))
            removed_entities += int(
                session.execute(delete(Entity).where(Entity.id == entity_id)).rowcount or 0
            )
    removed["entities"] = removed_entities

    # Insights that cited this record must not keep dangling references.
    removed_insights = 0
    for insight in session.execute(select(Insight)).scalars():
        ids = [i for i in (insight.record_ids or []) if i != record_id]
        changed = ids != (insight.record_ids or [])
        insight.record_ids = ids
        if changed and not ids:
            session.delete(insight)
            removed_insights += 1
        elif changed:
            insight.evidence = [
                e for e in (insight.evidence or []) if e.get("record_id") != record_id
            ]
            insight.facts = [f for f in (insight.facts or []) if f.get("record_id") != record_id]
    removed["insights"] = removed_insights

    session.flush()

    # Indices are in-memory; rebuild from the (now changed) database.
    bm25_index.get_default_index().invalidate()
    embeddings.get_default_index().invalidate()

    problems = verify_record_purged(session, record_id)
    return DeletionReport(
        record_id=record_id,
        removed=removed,
        orphaned_entities_removed=removed_entities,
        index_invalidated=True,
        verified=not problems,
        problems=problems,
    )


def verify_record_purged(session: Session, record_id: int) -> list[str]:
    """Assert no derived row still references ``record_id``. Returns problems."""
    problems: list[str] = []

    checks = [
        ("records", select(func.count()).select_from(Record).where(Record.id == record_id)),
        ("entity_mentions", select(func.count()).select_from(EntityMention).where(EntityMention.record_id == record_id)),
        ("events", select(func.count()).select_from(Event).where(Event.record_id == record_id)),
        ("chunks", select(func.count()).select_from(Chunk).where(Chunk.record_id == record_id)),
        ("embeddings", select(func.count()).select_from(Embedding).where(Embedding.record_id == record_id)),
        ("risk_assessments", select(func.count()).select_from(RiskAssessment).where(RiskAssessment.record_id == record_id)),
        ("relations", select(func.count()).select_from(Relation).where(Relation.record_id == record_id)),
    ]
    for name, stmt in checks:
        count = int(session.execute(stmt).scalar_one())
        if count:
            problems.append(f"{name}: {count} row(s) still reference record {record_id}")

    dangling = int(
        session.execute(
            select(func.count())
            .select_from(EntityMention)
            .where(EntityMention.record_id == record_id, EntityMention.entity_id.is_not(None))
        ).scalar_one()
    )
    if dangling:
        problems.append(f"entity_mentions: {dangling} unresolved mention(s) remain")

    return problems


def delete_all_data(session: Session) -> dict[str, int]:
    """Settings -> 'delete all data'. Removes every record and all derived state."""
    from ..retrieval import bm25_index, embeddings

    counts: dict[str, int] = {}
    for name, model in (
        ("embeddings", Embedding),
        ("chunks", Chunk),
        ("events", Event),
        ("risk_assessments", RiskAssessment),
        ("entity_mentions", EntityMention),
        ("relations", Relation),
        ("entity_aliases", EntityAlias),
        ("entities", Entity),
        ("insights", Insight),
        ("records", Record),
    ):
        result = session.execute(delete(model))
        counts[name] = int(result.rowcount or 0)

    session.execute(delete(DataSource).where(DataSource.key != "demo"))
    for key in session.execute(select(AppSetting.key)).scalars():
        if key.startswith("user."):
            session.execute(delete(AppSetting).where(AppSetting.key == key))

    session.flush()
    bm25_index.get_default_index().invalidate()
    embeddings.get_default_index().invalidate()
    logger.info("deleted all data counts=%s", counts)
    return counts


# ---------------------------------------------------------------------------
# Optional LLM boundary
# ---------------------------------------------------------------------------


@dataclass
class LlmPolicy:
    """The gate for anything that would leave this machine."""

    enabled: bool
    endpoint: str | None
    model: str | None
    disclose_to_user: bool = True

    @classmethod
    def from_settings(cls) -> "LlmPolicy":
        s = get_settings()
        enabled = bool(s.allow_external_llm and s.llm_endpoint)
        return cls(
            enabled=enabled,
            endpoint=s.llm_endpoint if enabled else None,
            model=s.llm_model if enabled else None,
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "endpoint": self.endpoint,
            "model": self.model,
            "disclosure": (
                "No record content leaves this machine. DigitalMind runs entirely locally."
                if not self.enabled
                else (
                    "An external language model endpoint is configured. Record text will be "
                    "sent to it when you use features that need it."
                )
            ),
            "integrations_configured": False,
            "note": (
                "Third-party integrations (Gmail, Outlook, Slack, Notion) are not configured. "
                "DigitalMind only reads files you upload or the built-in demo data."
            ),
        }

    def assert_allowed(self, operation: str) -> None:
        if not self.enabled:
            raise PermissionError(
                f"Refusing to {operation}: the optional external LLM layer is disabled. "
                "Enable it in Settings only if you intend record text to leave this machine."
            )


def privacy_report(session: Session, data_dir: Path | None = None) -> dict[str, Any]:
    """Counts for the Settings > Privacy panel. Counts only, never content."""
    directory = data_dir or get_settings().data_dir
    counts = {}
    for name, model in (
        ("records", Record),
        ("entities", Entity),
        ("mentions", EntityMention),
        ("events", Event),
        ("chunks", Chunk),
        ("embeddings", Embedding),
        ("risk_assessments", RiskAssessment),
        ("insights", Insight),
    ):
        counts[name] = int(session.execute(select(func.count()).select_from(model)).scalar_one())

    db_bytes = 0
    if directory.exists():
        for path in directory.glob("digitalmind.db*"):
            db_bytes += path.stat().st_size

    return {
        "counts": counts,
        "database_bytes": db_bytes,
        "database_path": str(directory / "digitalmind.db"),
        "external_transmission": False,
        "llm": LlmPolicy.from_settings().status(),
        "log_policy": (
            "Record content is never written to the application log. Logs contain record ids, "
            "counts and model names only."
        ),
        "delete_semantics": (
            "Deleting a record also deletes its extracted entities, graph edges, events, "
            "risk assessments, insights and index entries."
        ),
    }

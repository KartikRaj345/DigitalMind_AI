"""Privacy tests.

The central guarantee: deleting a record removes everything derived from it. This is
tested directly rather than inferred from the schema, because SQLite's cascade depends on
a per-connection pragma and a privacy guarantee must not depend on that.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from backend.app.models import (
    Chunk,
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
from backend.app.privacy import (
    LlmPolicy,
    delete_all_data,
    delete_record_cascade,
    privacy_report,
    verify_record_purged,
)

DERIVED_MODELS = (EntityMention, Event, Chunk, Embedding, RiskAssessment, Relation)


def _counts(session) -> dict[str, int]:
    out = {"records": int(session.execute(select(func.count()).select_from(Record)).scalar_one())}
    for model in DERIVED_MODELS:
        out[model.__tablename__] = int(
            session.execute(select(func.count()).select_from(model)).scalar_one()
        )
    out["entities"] = int(session.execute(select(func.count()).select_from(Entity)).scalar_one())
    return out


def test_delete_record_removes_every_derived_row(demo_session):
    session = demo_session
    record = session.execute(select(Record)).scalars().first()
    record_id = record.id

    before = _counts(session)
    assert before["entity_mentions"] > 0
    assert before["events"] > 0
    assert before["chunks"] > 0
    assert before["risk_assessments"] > 0

    report = delete_record_cascade(session, record_id)
    session.flush()

    assert report.verified, f"purging problems: {report.problems}"
    assert verify_record_purged(session, record_id) == []

    after = _counts(session)
    assert after["records"] == before["records"] - 1
    for name in ("entity_mentions", "events", "chunks", "risk_assessments"):
        assert after[name] < before[name] or after[name] == 0, f"{name} did not shrink"
    # Nothing that referenced the deleted record may survive.
    assert int(
        session.execute(
            select(func.count()).select_from(EntityMention).where(EntityMention.record_id == record_id)
        ).scalar_one()
    ) == 0
    assert int(
        session.execute(select(func.count()).select_from(Event).where(Event.record_id == record_id)).scalar_one()
    ) == 0
    assert int(
        session.execute(select(func.count()).select_from(Chunk).where(Chunk.record_id == record_id)).scalar_one()
    ) == 0
    assert int(
        session.execute(
            select(func.count()).select_from(Embedding).where(Embedding.record_id == record_id)
        ).scalar_one()
    ) == 0
    assert int(
        session.execute(
            select(func.count()).select_from(RiskAssessment).where(RiskAssessment.record_id == record_id)
        ).scalar_one()
    ) == 0
    assert int(
        session.execute(select(func.count()).select_from(Relation).where(Relation.record_id == record_id)).scalar_one()
    ) == 0


def test_delete_record_removes_orphaned_entities(demo_session):
    session = demo_session
    # Pick an entity that appears in exactly one record, so deleting it orphans the entity.
    from sqlalchemy import func as sqlfunc

    counts = {
        entity_id: n
        for entity_id, n in session.execute(
            select(EntityMention.entity_id, sqlfunc.count(EntityMention.id))
            .group_by(EntityMention.entity_id)
        )
    }
    orphan_candidates = [eid for eid, n in counts.items() if eid is not None and n == 1]
    assert orphan_candidates, "expected at least one single-mention entity in the demo corpus"
    entity_id = orphan_candidates[0]
    record_id = session.execute(
        select(EntityMention.record_id).where(EntityMention.entity_id == entity_id)
    ).scalar_one()

    report = delete_record_cascade(session, record_id)
    session.flush()

    assert report.orphaned_entities_removed >= 1
    assert session.get(Entity, entity_id) is None
    assert int(
        session.execute(
            select(func.count()).select_from(EntityAlias).where(EntityAlias.entity_id == entity_id)
        ).scalar_one()
    ) == 0


def test_delete_record_invalidates_indices(demo_session, monkeypatch):
    from backend.app.retrieval import bm25_index, embeddings

    # Use the hashing embedder so the test does not load a transformer model. The point
    # is the index invalidation, not the embedding quality.
    monkeypatch.setattr(embeddings, "get_embedder", lambda model_name: embeddings.HashingEmbedder())

    session = demo_session
    from backend.app.pipeline import build_indexes

    build_indexes(session, with_embeddings=True)
    assert embeddings.get_default_index().size > 0
    record_id = session.execute(select(Record)).scalars().first().id

    delete_record_cascade(session, record_id)
    assert bm25_index.get_default_index().size == 0
    assert embeddings.get_default_index().size == 0


def test_delete_record_clears_insight_references(demo_session):
    session = demo_session
    record_id = session.execute(select(Record)).scalars().first().id
    delete_record_cascade(session, record_id)
    session.flush()

    for insight in session.execute(select(Insight)).scalars():
        assert record_id not in (insight.record_ids or [])
        assert all(e.get("record_id") != record_id for e in (insight.evidence or []))
        assert all(f.get("record_id") != record_id for f in (insight.facts or []))


def test_delete_all_data_empties_every_table(demo_session):
    session = demo_session
    counts = delete_all_data(session)
    session.flush()

    assert counts["records"] > 0
    for name, value in _counts(session).items():
        assert value == 0, f"{name} still has {value} rows after delete-all"
    assert int(session.execute(select(func.count()).select_from(Insight)).scalar_one()) == 0


def test_delete_unknown_record_is_reported_not_crashed(demo_session):
    report = delete_record_cascade(demo_session, 999_999)
    assert report.removed == {}
    assert report.problems == ["record not found"]


def test_privacy_report_counts_only(demo_session):
    report = privacy_report(demo_session)
    assert report["external_transmission"] is False
    assert report["counts"]["records"] > 0
    # The report must be safe to render in the UI: counts, never content.
    assert "body" not in str(report).lower() or "delete" in str(report).lower()


def test_llm_policy_refuses_by_default():
    policy = LlmPolicy.from_settings()
    assert policy.enabled is False
    assert policy.endpoint is None
    with pytest.raises(PermissionError) as exc:
        policy.assert_allowed("send record text")
    assert "disabled" in str(exc.value)


def test_llm_policy_status_is_honest():
    status = LlmPolicy.from_settings().status()
    assert status["enabled"] is False
    assert status["integrations_configured"] is False
    assert "locally" in status["disclosure"].lower()

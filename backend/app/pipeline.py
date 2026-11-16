"""Analysis pipeline: ingest -> extract -> resolve -> graph -> timeline -> risk -> index.

This module is the only place that orchestrates the stages, so ordering guarantees live
in one readable place:

1. **Ingest**   - records stored verbatim; no interpretation yet.
2. **Extract**  - per record: spans, normalized dates, events, back-references.
3. **Resolve**  - mention clusters become canonical entities; merges are recorded.
4. **Graph**    - entities + records + evidenced edges.
5. **Risk**     - per record, per indicator, with cited spans.
6. **Index**    - chunks, BM25 and embeddings.
7. **Insights** - cross-record conclusions, always with citations and confidence.

Every stage is idempotent: re-running it for one record replaces that record's derived
rows rather than accumulating duplicates.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .entity import graph as graph_mod
from .entity import resolve as resolve_mod
from .ingest.demo import DEMO_RECORDS, DemoRecord
from .logging_config import get_logger, log_counts
from .models import (
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
from .nlp import extract as extract_mod
from .retrieval import bm25_index, chunker, embeddings, hybrid
from .risk import assess as assess_mod
from .risk import tfidf_lr, transformer_clf
from .timeline import builder as timeline_mod

logger = get_logger(__name__)

UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_utc(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def content_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


_EMAIL_IN_TEXT = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    inserted: int = 0
    skipped: int = 0
    ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"inserted": self.inserted, "skipped": self.skipped, "record_ids": self.ids}


def ingest_records(
    session: Session,
    records: Sequence[DemoRecord],
    origin: str = "demo",
    replace: bool = False,
) -> IngestStats:
    """Store records verbatim. De-duplicates on ``(origin, external_id)``."""
    stats = IngestStats()
    if replace:
        session.execute(delete(Record).where(Record.origin == origin))

    existing = {
        ext
        for (ext,) in session.execute(
            select(Record.external_id).where(Record.origin == origin)
        )
    }

    for item in records:
        if item.external_id and item.external_id in existing:
            stats.skipped += 1
            continue
        rec = Record(
            source_type=item.source_type,
            title=item.title,
            body=item.body,
            occurred_at=_as_utc(item.occurred_at),
            origin=origin,
            external_id=item.external_id,
            participants=list(item.participants or []),
            extra=dict(item.extra or {}),
            content_hash=content_hash(item.body),
            char_count=len(item.body),
        )
        session.add(rec)
        session.flush()
        stats.inserted += 1
        stats.ids.append(rec.id)
        if item.external_id:
            existing.add(item.external_id)

    _update_source(session, origin, label=f"Demo corpus ({len(records)} records)")
    log_counts(logger, "ingested", inserted=stats.inserted, skipped=stats.skipped, origin=origin)
    return stats


def _update_source(session: Session, key: str, label: str, kind: str = "local", **detail) -> None:
    # DataSource's primary key is `id`, not `key`, so session.get() would always miss and
    # the subsequent INSERT would violate the UNIQUE constraint on `key`.
    source = session.execute(
        select(DataSource).where(DataSource.key == key)
    ).scalars().first()
    count = int(
        session.execute(select(func.count()).select_from(Record).where(Record.origin == key)).scalar_one()
    )
    if source is None:
        session.add(
            DataSource(key=key, label=label, kind=kind, configured=True, record_count=count, detail=detail)
        )
    else:
        source.label = label
        source.record_count = count
        source.configured = True
        if detail:
            source.detail = {**(source.detail or {}), **detail}
    session.flush()


def load_demo(session: Session, replace: bool = True) -> IngestStats:
    return ingest_records(session, DEMO_RECORDS, origin="demo", replace=replace)


# ---------------------------------------------------------------------------
# Stage 2: extraction
# ---------------------------------------------------------------------------


def analyze_record(session: Session, record: Record, use_spacy: bool = True) -> None:
    """Run extraction, events and chunking for a single record and persist the result."""
    anchor = _as_utc(record.occurred_at) or dt.datetime.now(UTC)
    result = extract_mod.extract(
        record.body, anchor, title=record.title, use_spacy=use_spacy
    )

    session.execute(delete(EntityMention).where(EntityMention.record_id == record.id))
    session.execute(delete(Event).where(Event.record_id == record.id))
    session.execute(delete(Chunk).where(Chunk.record_id == record.id))
    session.execute(delete(Embedding).where(Embedding.record_id == record.id))

    for span in result.spans:
        session.add(
            EntityMention(
                record_id=record.id,
                entity_id=None,  # assigned by the resolve stage
                text=span.text,
                label=span.label,
                start_char=span.start,
                end_char=span.end,
                confidence=span.confidence,
                extractor=span.extractor,
            )
        )

    for ev in result.events:
        if ev.start_at is None:
            continue
        session.add(
            Event(
                record_id=record.id,
                title=(ev.title or ev.raw_expression)[:500],
                description=ev.description,
                note=ev.note,
                kind=ev.kind,
                certainty=ev.certainty,
                start_at=ev.start_at,
                end_at=ev.end_at,
                precision=ev.precision,
                raw_expression=ev.raw_expression[:250],
                start_char=ev.start_char,
                end_char=ev.end_char,
                confidence=ev.confidence,
                extractor=ev.extractor,
                participants=list(record.participants or []),
            )
        )

    for i, chunk in enumerate(chunker.renumber(chunker.chunk_text(record.body))):
        session.add(
            Chunk(
                record_id=record.id,
                ordinal=i,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                indexed_lexical=False,
                indexed_semantic=False,
            )
        )
    session.flush()


# ---------------------------------------------------------------------------
# Stage 3: entity resolution
# ---------------------------------------------------------------------------


def _harmonize_labels(mentions: list[EntityMention]) -> int:
    """Resolve type disagreements between extractors for the same surface form.

    spaCy tags "Marcus Delgado" PERSON in prose and ORG inside a signature block; the
    rules baseline says PERSON in both. Left alone, entity resolution would create two
    canonical entities with the same name, and every downstream count would be wrong.

    Rule: for each normalized surface form, take the majority label; on a tie prefer
    PERSON, because a personal name is a more specific claim than a generic ORG guess.
    Returns the number of mentions whose label was changed.
    """
    votes: dict[tuple[str, str], dict[str, int]] = {}
    for m in mentions:
        norm = resolve_mod.normalize(m.text, m.label)
        votes.setdefault((norm, m.text.lower()), {}).setdefault(m.label, 0)
        votes[(norm, m.text.lower())][m.label] += 1

    changed = 0
    for _key, counts in votes.items():
        if len(counts) < 2:
            continue
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] != "PERSON", kv[0]))[0][0]
        for m in mentions:
            if resolve_mod.normalize(m.text, m.label) != _key[0]:
                continue
            if m.text.lower() != _key[1] or m.label == best:
                continue
            m.label = best
            changed += 1
    if changed:
        logger.info("harmonized entity labels for %d mention(s)", changed)
    return changed


def resolve_entities(session: Session) -> dict[str, Any]:
    """Cluster every mention, persist canonical entities and attach mentions to them.

    Returns a summary including the low-cohesion clusters, which are surfaced to the user
    as "these might be separate entities" rather than silently merged.
    """
    mentions = list(session.execute(select(EntityMention)).scalars())
    if not mentions:
        return {"entities": 0, "mentions": 0, "merges": 0, "low_cohesion": []}

    _harmonize_labels(mentions)

    bodies = {
        rid: body
        for rid, body in session.execute(select(Record.id, Record.body))
    }
    email_domains: dict[str, str] = {}
    for m in mentions:
        body = bodies.get(m.record_id, "")
        domain_match = _EMAIL_IN_TEXT.search(m.text)
        if domain_match is None and 0 <= m.start_char <= len(body):
            domain_match = _EMAIL_IN_TEXT.search(body[m.start_char : m.end_char])
        if domain_match:
            email_domains[f"m{m.id}"] = domain_match.group(1).lower()

    inputs = [
        resolve_mod.Mention(
            key=f"m{m.id}", record_id=m.record_id, text=m.text, label=m.label
        )
        for m in mentions
    ]
    clusters = resolve_mod.cluster_mentions(inputs, email_domains)

    # Reset entity links so re-running does not duplicate entities. The bulk DELETE
    # detaches the ORM objects from their old rows, so the list is rebuilt afterwards -
    # otherwise assigning `m.entity_id` below would emit an UPDATE against a deleted row
    # and SQLAlchemy would raise StaleDataError.
    session.execute(delete(Entity))
    session.execute(delete(EntityAlias))
    session.flush()
    mentions = list(session.execute(select(EntityMention)).scalars())
    for m in mentions:
        m.entity_id = None
    session.flush()

    key_to_entity: dict[str, int] = {}
    low_cohesion: list[dict[str, Any]] = []

    for cluster in clusters:
        entity = Entity(
            name=cluster.canonical,
            normalized_name=resolve_mod.normalize(cluster.canonical, cluster.label),
            entity_type=cluster.label,
            attributes={"surface_forms": sorted(cluster.surface_forms)},
            resolution_confidence=round(cluster.cohesion, 4),
            mention_count=cluster.total_count,
        )
        session.add(entity)
        session.flush()
        for key in cluster.mentions:
            key_to_entity[key] = entity.id

        for form in sorted(cluster.surface_forms):
            if resolve_mod.normalize(form, cluster.label) == entity.normalized_name:
                continue
            session.add(
                EntityAlias(
                    entity_id=entity.id,
                    alias=form,
                    normalized_alias=resolve_mod.normalize(form, cluster.label),
                    mention_count=1,
                    source="merge",
                )
            )

        if cluster.cohesion < resolve_mod.MIN_CLUSTER_COHESION and len(cluster.surface_forms) > 1:
            low_cohesion.append(
                {
                    "entity_id": entity.id,
                    "canonical": cluster.canonical,
                    "forms": sorted(cluster.surface_forms),
                    "cohesion": round(cluster.cohesion, 3),
                }
            )

    for m in mentions:
        # cluster_mentions keys its members by "<normalized form>::<label>".
        key = f"{resolve_mod.normalize(m.text, m.label)}::{m.label}"
        m.entity_id = key_to_entity.get(key)
    session.flush()

    summary = {
        "entities": len(clusters),
        "mentions": len(mentions),
        "merges": sum(1 for c in clusters for _ in c.merges),
        "low_cohesion": low_cohesion,
    }
    log_counts(
        logger,
        "resolved entities",
        entities=summary["entities"],
        mentions=summary["mentions"],
        merges=summary["merges"],
    )
    return summary


# ---------------------------------------------------------------------------
# Stage 4: graph
# ---------------------------------------------------------------------------


def build_graph(session: Session) -> dict[str, Any]:
    mentions = list(session.execute(select(EntityMention)).scalars())
    events = list(session.execute(select(Event)).scalars())
    assessments = list(session.execute(select(RiskAssessment)).scalars())

    session.execute(delete(Relation))
    session.flush()

    for row in graph_mod.build_relation_candidates(mentions, events, assessments):
        if row["target_entity_id"] < 0:
            # mentioned_in / participant edges are derived from mentions at read time;
            # storing them as relations with a negative id would be a lie. Skip here.
            continue
        session.add(
            Relation(
                source_entity_id=row["source_entity_id"],
                target_entity_id=row["target_entity_id"],
                relation_type=row["relation_type"],
                record_id=row["record_id"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                weight=row["weight"],
                confidence=row["confidence"],
                method=row["method"],
            )
        )
    session.flush()

    # Refresh mention counts on entities.
    session.execute(
        Entity.__table__.update().values(
            mention_count=0,
        )
    )
    for entity_id, count in session.execute(
        select(EntityMention.entity_id, func.count(EntityMention.id)).group_by(
            EntityMention.entity_id
        )
    ):
        if entity_id is not None:
            session.get(Entity, entity_id).mention_count = int(count)
    session.flush()

    entities = list(session.execute(select(Entity)).scalars())
    records = list(session.execute(select(Record)).scalars())
    relations = list(session.execute(select(Relation)).scalars())
    g = graph_mod.to_networkx(entities, mentions, relations, records)
    log_counts(logger, "built graph", nodes=g.number_of_nodes(), edges=g.number_of_edges())
    return graph_mod.GraphBuildResult(
        nodes=g.number_of_nodes(),
        edges=g.number_of_edges(),
        mentions=len(mentions),
        entities=len(entities),
    ).as_dict()


# ---------------------------------------------------------------------------
# Stage 5: risk
# ---------------------------------------------------------------------------


@dataclass
class RiskBundle:
    """Loaded once per analysis run, reused for every record."""

    tfidf: tfidf_lr.TfidfRiskModel | None = None
    transformer: transformer_clf.ClassifierBundle | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "RiskBundle":
        settings = get_settings()
        notes: list[str] = []
        tfidf = tfidf_lr.TfidfRiskModel.load(settings.artifacts_dir)
        if tfidf is None:
            notes.append(
                "The TF-IDF risk baseline has not been trained yet; rule-based detection is "
                "in use. Run backend/scripts/train_risk.py to train it."
            )
        bundle = transformer_clf.load_classifier(settings.artifacts_dir / "distilbert-risk")
        if bundle is None:
            notes.append(
                "The DistilBERT risk classifier is not available (needs requirements-ml.txt "
                "and a training run). Detection uses the rules and TF-IDF models only."
            )
        return cls(tfidf=tfidf, transformer=bundle, notes=notes)

    def available_methods(self) -> list[str]:
        out = ["rules"]
        if self.tfidf is not None:
            out.append("tfidf-lr")
        if self.transformer is not None:
            out.append("distilbert")
        return out


def assess_records(session: Session, bundle: RiskBundle | None = None) -> dict[str, Any]:
    """Assess every record and persist one row per (record, label, method)."""
    bundle = bundle or RiskBundle.load()
    records = list(session.execute(select(Record).order_by(Record.id)).scalars())
    if not records:
        return {"records": 0, "methods": bundle.available_methods(), "notes": bundle.notes}

    texts = [rec.body for rec in records]

    model_scores: dict[str, Any] = {}
    model_thresholds: dict[str, float] = {}
    model_names: dict[str, str] = {}

    if bundle.tfidf is not None:
        probs, _ = bundle.tfidf.predict(texts)
        model_scores["tfidf-lr"] = probs
        model_thresholds.update(bundle.tfidf.thresholds)
        model_names["tfidf-lr"] = "TF-IDF + logistic regression"

    calibrators = None
    if bundle.transformer is not None:
        cal_probs, _ = bundle.transformer.predict(texts)
        model_scores["distilbert"] = cal_probs
        model_thresholds.update(bundle.transformer.thresholds)
        model_names["distilbert"] = "DistilBERT (fine-tuned, calibrated)"
        calibrators = bundle.transformer.calibrators

    session.execute(delete(RiskAssessment))
    session.flush()

    flagged = 0
    for idx, rec in enumerate(records):
        scores = {name: arr[idx] for name, arr in model_scores.items()}
        thresholds = {
            name: {lbl: model_thresholds.get(lbl, 0.5) for lbl in assess_mod.RISK_LABELS}
            for name in scores
        }
        report = assess_mod.combine(
            rec.id,
            rec.body,
            model_scores=scores,
            model_thresholds=thresholds,
            model_names=model_names,
            calibrators=calibrators,
        )
        if report.flagged:
            flagged += 1
        for indicator in report.indicators:
            primary = max(indicator.methods.items(), key=lambda kv: kv[1])[0]
            session.add(
                RiskAssessment(
                    record_id=rec.id,
                    label=indicator.label,
                    present=indicator.present,
                    score=indicator.score,
                    threshold=indicator.threshold,
                    method=primary,
                    explanation=indicator.explanation,
                    evidence=indicator.evidence,
                    calibrated=indicator.calibrated,
                )
            )
    session.flush()
    log_counts(logger, "assessed risk", records=len(records), flagged=flagged)
    return {
        "records": len(records),
        "flagged": flagged,
        "methods": bundle.available_methods(),
        "notes": bundle.notes,
    }


# ---------------------------------------------------------------------------
# Stage 6: index
# ---------------------------------------------------------------------------


def build_indexes(session: Session, with_embeddings: bool = True) -> dict[str, Any]:
    """Chunk (already done), then BM25 and the embedding index."""
    settings = get_settings()

    chunks = list(session.execute(select(Chunk).order_by(Chunk.id)).scalars())
    lexical = bm25_index.get_default_index()
    lexical.build(
        [(c.id, c.record_id, c.ordinal, c.text, c.start_char, c.end_char) for c in chunks],
        signature=(len(chunks),),
    )
    session.execute(
        Chunk.__table__.update().values(indexed_lexical=True, indexed_semantic=False)
    )

    semantic_status: dict[str, Any] = {"built": 0, "model": "", "backend": "numpy", "fallback": True}
    if with_embeddings and chunks:
        embedder = embeddings.get_embedder(settings.embeddings_model)
        semantic = embeddings.get_default_index()
        texts = [c.text for c in chunks]
        vectors = embedder.encode(texts)

        session.execute(delete(Embedding))
        rows = []
        for chunk, vec in zip(chunks, vectors):
            packed = embeddings.pack_vector(vec)
            emb = Embedding(
                chunk_id=chunk.id,
                record_id=chunk.record_id,
                model=getattr(embedder, "name", "unknown"),
                dim=int(vec.shape[0]),
                vector=packed,
            )
            session.add(emb)
            rows.append((chunk.id, chunk.record_id, packed))
            chunk.indexed_semantic = True
            chunk.embedding_row = None
        session.flush()

        semantic.build(rows, model_name=getattr(embedder, "name", "unknown"))
        semantic_status = {
            "built": len(rows),
            "model": getattr(embedder, "name", "unknown"),
            "backend": "faiss" if embeddings.faiss_available() else "numpy",
            "fallback": bool(getattr(embedder, "is_fallback", False)),
            "dim": int(vectors.shape[1]) if len(vectors) else 0,
        }
        session.execute(Chunk.__table__.update().values(indexed_semantic=True))

    session.flush()
    log_counts(logger, "built indexes", chunks=len(chunks), semantic=semantic_status.get("built", 0))
    return {"lexical_chunks": len(chunks), "semantic": semantic_status}


def rebuild_indexes_after_change(session: Session) -> dict[str, Any]:
    """Called after any insert/delete so indices can never hold deleted rows."""
    return build_indexes(session)


# ---------------------------------------------------------------------------
# Stage 7: insights
# ---------------------------------------------------------------------------


def generate_insights(session: Session, replace: bool = True) -> dict[str, Any]:
    """Cross-record conclusions. Every insight is inspectable and cites its facts."""
    if replace:
        session.execute(delete(Insight))
        session.flush()

    records = {r.id: r for r in session.execute(select(Record)).scalars()}
    if not records:
        return {"created": 0, "kinds": {}}

    assessments = list(session.execute(select(RiskAssessment)).scalars())
    risk_by_record: dict[int, list[RiskAssessment]] = {}
    for a in assessments:
        risk_by_record.setdefault(a.record_id, []).append(a)

    created: list[Insight] = []
    kinds: dict[str, int] = {}

    def add(**kwargs) -> None:
        nonlocal created
        insight = Insight(**kwargs)
        session.add(insight)
        created.append(insight)
        kinds[kwargs["kind"]] = kinds.get(kwargs["kind"], 0) + 1

    # --- 1. Risk observations -----------------------------------------
    for record_id, items in sorted(risk_by_record.items()):
        rec = records.get(record_id)
        if rec is None:
            continue
        flagged = [a for a in items if a.present]
        if not flagged:
            continue
        evidence = [
            {
                "record_id": record_id,
                "start_char": e.get("start_char"),
                "end_char": e.get("end_char"),
                "quote": e.get("text"),
                "note": e.get("note"),
                "label": a.label,
            }
            for a in flagged
            for e in (a.evidence or [])
            if e.get("start_char") is not None
        ]
        facts = [f for f in evidence if f.get("quote")]
        score = max((a.score for a in flagged), default=0.0)
        label_names = ", ".join(sorted(a.label.replace("_", " ") for a in flagged))
        add(
            kind="risk_observation",
            severity="warning" if len(flagged) >= 2 else "notice",
            title=f"Potential manipulation risk detected in “{rec.display_title}”",
            statement=(
                f"This record contains {label_names} indicators. That combination often appears "
                "in messages that apply time or emotional pressure, which is why it is shown "
                "for review. It is a pattern match on wording, not a judgement about the sender."
            ),
            facts=facts,
            evidence=evidence,
            confidence=round(score, 4),
            confidence_basis=(
                f"Highest single-indicator score among {len(flagged)} flagged indicator(s), "
                "capped by how many independent methods agreed."
            ),
            record_ids=[record_id],
            entity_ids=[],
        )

    # --- 2. Unresolved back-references (Scenario B) --------------------
    _backreference_insights(session, records, add)

    # --- 3. Topic clusters ---------------------------------------------
    _cluster_insights(session, records, add)

    session.flush()
    log_counts(logger, "generated insights", count=len(created))
    return {"created": len(created), "kinds": kinds}


def _backreference_insights(
    session: Session,
    records: dict[int, Record],
    add,
) -> None:
    """Records that refer back to a discussion we may not have.

    The wording is the whole point: we report the *search*, never the absence of an event.
    """
    from .nlp.dates import find_backreferences
    from .retrieval import qa as qa_mod

    for record_id, rec in sorted(records.items()):
        body = rec.body
        hits = find_backreferences(body)
        if not hits:
            continue

        earlier_ids = [
            other.id
            for other in records.values()
            if other.occurred_at
            and rec.occurred_at
            and other.occurred_at < rec.occurred_at
        ]

        for start, end, phrase in hits:
            window_end = min(len(body), end + 160)
            topic = body[end : window_end].strip(" .,:;")
            topic_query = topic[:120] or phrase
            matched = _search_records(session, topic_query, earlier_ids)

            if not earlier_ids:
                # Nothing exists to search. Saying "no matching discussion was found" here
                # would be misleading: the search never had a candidate to miss.
                add(
                    kind="unverifiable_discussion_reference",
                    severity="info",
                    title=f"This reference cannot be checked — “{rec.display_title}”",
                    statement=(
                        f"This record refers back to an earlier discussion (“{phrase}”), but it "
                        "is the earliest record in the data you have analyzed, so there is "
                        "nothing earlier to search. That is not evidence that no discussion "
                        "happened - it means the conversation it refers to has not been imported."
                    ),
                    facts=[
                        {
                            "record_id": record_id,
                            "start_char": start,
                            "end_char": end,
                            "quote": phrase,
                            "note": "Phrase asserting a prior discussion.",
                        }
                    ],
                    evidence=[
                        {
                            "record_id": record_id,
                            "start_char": start,
                            "end_char": end,
                            "quote": phrase,
                            "note": "Phrase asserting a prior discussion.",
                        }
                    ],
                    confidence=1.0,
                    confidence_basis=(
                        "Certain that nothing earlier is available to search. No confidence "
                        "either way about whether the discussion happened."
                    ),
                    record_ids=[record_id],
                    entity_ids=[],
                )
                continue

            if matched:
                # Point at the first entity mention in each candidate record as a
                # concrete place to look, rather than inventing a span.
                anchor_mentions = session.execute(
                    select(EntityMention.record_id, EntityMention.start_char, EntityMention.end_char)
                    .where(EntityMention.record_id.in_(matched[:3]))
                    .order_by(EntityMention.record_id, EntityMention.start_char)
                ).all()
                first_anchor: dict[int, tuple[int, int]] = {}
                for mid, s, e in anchor_mentions:
                    first_anchor.setdefault(mid, (int(s), int(e)))

                facts = [
                    {
                        "record_id": record_id,
                        "start_char": start,
                        "end_char": end,
                        "quote": phrase,
                        "note": "The record refers back to a prior discussion.",
                    }
                ] + [
                    {
                        "record_id": mid,
                        "start_char": first_anchor[mid][0],
                        "end_char": first_anchor[mid][1],
                        "quote": None,
                        "note": f"Possible matching discussion in record {mid}.",
                    }
                    for mid in matched[:3]
                    if mid in first_anchor
                ]
                add(
                    kind="discussion_reference",
                    severity="info",
                    title=f"“{rec.display_title}” refers to an earlier discussion",
                    statement=(
                        f"This record says “{phrase}”. {len(matched)} earlier record(s) in the "
                        "analyzed data may contain that discussion. This is a suggested link "
                        "based on wording, not a confirmed match."
                    ),
                    facts=facts[:4],
                    evidence=facts[:4],
                    confidence=0.55,
                    confidence_basis=(
                        "Heuristic: the referenced phrase appears in a search over earlier "
                        "records. Not verified by reading either record."
                    ),
                    record_ids=[record_id] + matched[:3],
                    entity_ids=[],
                )
                continue

            add(
                kind="unverifiable_discussion_reference",
                severity="info",
                title=f"This reference cannot be checked — “{rec.display_title}”",
                statement=(
                    f"This record refers back to an earlier discussion (“{phrase}”), but it is "
                    "the earliest record in the data you have analyzed, so there is nothing "
                    "earlier to search. That is not evidence that no discussion happened - it "
                    "means the conversation it refers to has not been imported."
                ),
                facts=[
                    {
                        "record_id": record_id,
                        "start_char": start,
                        "end_char": end,
                        "quote": phrase,
                        "note": "Phrase asserting a prior discussion.",
                    }
                ],
                evidence=[
                    {
                        "record_id": record_id,
                        "start_char": start,
                        "end_char": end,
                        "quote": phrase,
                        "note": "Phrase asserting a prior discussion.",
                    }
                ],
                confidence=1.0,
                confidence_basis=(
                    "Certain that nothing earlier is available to search. No confidence either "
                    "way about whether the discussion happened."
                ),
                record_ids=[record_id],
                entity_ids=[],
            )
            continue

        add(
            kind="unresolved_discussion_reference",
            severity="notice",
            title=f"No matching discussion was found for “{rec.display_title}”",
            statement=(
                f"This record refers back to an earlier discussion (“{phrase}”), but "
                f"{qa_mod.NO_MATCH} I searched {len(earlier_ids)} earlier record(s) using "
                "keyword and semantic search. That is not evidence that no such discussion "
                "took place - the original channel may simply not be connected or imported yet."
            ),
            facts=[
                {
                    "record_id": record_id,
                    "start_char": start,
                    "end_char": end,
                    "quote": phrase,
                    "note": "Phrase asserting a prior discussion.",
                }
            ],
            evidence=[
                {
                    "record_id": record_id,
                    "start_char": start,
                    "end_char": end,
                    "quote": phrase,
                    "note": "Phrase asserting a prior discussion.",
                }
            ],
            confidence=1.0,
            confidence_basis=(
                "Certain that the search came up empty; no confidence either way about "
                "whether the discussion actually happened."
            ),
            record_ids=[record_id],
            entity_ids=[],
        )


def _search_records(session: Session, query: str, record_ids: Iterable[int]) -> list[int]:
    """BM25-only search over a subset of records. Used inside insight generation."""
    if not query.strip():
        return []
    ids = set(record_ids)
    if not ids:
        return []
    rows = session.execute(
        select(Chunk.id, Chunk.record_id, Chunk.ordinal, Chunk.text, Chunk.start_char, Chunk.end_char)
        .where(Chunk.record_id.in_(ids))
    ).all()
    if not rows:
        return []
    from rank_bm25 import BM25Okapi

    corpus = [re.findall(r"[a-z0-9][a-z0-9._'-]*", r[3].lower()) for r in rows]
    bm25 = BM25Okapi(corpus)
    tokens = re.findall(r"[a-z0-9][a-z0-9._'-]*", query.lower())
    scores = bm25.get_scores(tokens)
    ranked = sorted(
        ((int(r[1]), float(s)) for r, s in zip(rows, scores) if s > 0.0),
        key=lambda t: -t[1],
    )
    return [rid for rid, _s in ranked[:3]]


def _cluster_insights(session: Session, records: dict[int, Record], add) -> None:
    """Groups of records that share an entity, surfaced as a thread."""
    from collections import defaultdict

    by_entity: dict[int, list[int]] = defaultdict(list)
    for entity_id, record_id in session.execute(
        select(EntityMention.entity_id, EntityMention.record_id)
    ):
        if entity_id is not None:
            by_entity[entity_id].append(record_id)

    for entity_id, record_ids in sorted(by_entity.items()):
        unique = sorted(set(record_ids))
        if len(unique) < 3:
            continue
        entity = session.get(Entity, entity_id)
        if entity is None or entity.entity_type != "ORG":
            continue
        ordered = sorted(
            (records[rid] for rid in unique if rid in records),
            key=lambda r: (r.occurred_at or r.imported_at),
        )
        if len(ordered) < 3:
            continue
        add(
            kind="record_thread",
            severity="info",
            title=f"{len(ordered)} records involve {entity.name}",
            statement=(
                f"{len(ordered)} records in the analyzed data mention {entity.name}. "
                "They are grouped here because they share that entity, not because DigitalMind "
                "judged them to be related in meaning."
            ),
            facts=[
                {
                    "record_id": r.id,
                    "start_char": None,
                    "end_char": None,
                    "quote": None,
                    "note": f"{r.source_type}: {r.display_title}",
                }
                for r in ordered[:6]
            ],
            evidence=[],
            confidence=round(min(1.0, 0.5 + 0.1 * len(ordered)), 3),
            confidence_basis=f"Grouped by {len(unique)} shared mentions of a single entity.",
            record_ids=unique,
            entity_ids=[entity_id],
        )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(session: Session, use_spacy: bool = True, with_embeddings: bool = True) -> dict[str, Any]:
    """Run every stage over the current database contents."""
    settings = get_settings()
    started = dt.datetime.now(UTC)

    records = list(session.execute(select(Record).order_by(Record.id)).scalars())
    for rec in records:
        analyze_record(session, rec, use_spacy=use_spacy)

    resolution = resolve_entities(session)
    graph_stats = build_graph(session)
    risk_stats = assess_records(session)
    index_stats = build_indexes(session, with_embeddings=with_embeddings)
    insight_stats = generate_insights(session)

    elapsed = (dt.datetime.now(UTC) - started).total_seconds()
    summary = {
        "records": len(records),
        "resolution": resolution,
        "graph": graph_stats,
        "risk": risk_stats,
        "index": index_stats,
        "insights": insight_stats,
        "seconds": round(elapsed, 2),
        "spacy_available": _spacy_status(settings.spacy_model),
    }
    log_counts(
        logger,
        "pipeline complete",
        records=len(records),
        seconds=round(elapsed, 2),
    )
    return summary


def _spacy_status(model_name: str) -> dict[str, Any]:
    from .nlp import ner

    available = ner.spacy_available(model_name)
    return {
        "available": available,
        "model": model_name,
        "note": (
            f"spaCy {model_name} is loaded; entity extraction combines it with the rules "
            "baseline."
            if available
            else (
                f"spaCy {model_name} is not installed, so entity extraction uses the rules "
                "baseline only. Run: python -m spacy download en_core_web_sm"
            )
        ),
    }

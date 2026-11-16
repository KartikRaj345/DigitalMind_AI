"""API routes.

Every response that contains an inference includes the model that produced it, the
evidence it rests on, and a confidence with a stated basis. Every response that reports
an absence says so in terms of what was searched, never as proof that nothing happened.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_session
from .entity import graph as graph_mod
from .ingest import parsers
from .ingest.demo import DEMO_RECORDS
from .logging_config import get_logger
from .models import (
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
from .nlp import ner
from .nlp.dates import find_backreferences
from .pipeline import (
    RiskBundle,
    assess_records,
    build_indexes,
    generate_insights,
    ingest_records,
    load_demo,
    resolve_entities,
    run_pipeline,
)
from .privacy import delete_all_data, delete_record_cascade, privacy_report
from .retrieval import answering, bm25_index, embeddings
from .risk import assess as assess_mod
from .risk import lexicon, tfidf_lr, transformer_clf
from .schemas import (
    AskOut,
    AskRequest,
    GraphEdge,
    GraphNode,
    GraphOut,
    HealthOut,
    ImportResult,
    InsightOut,
    InsightUpdate,
    RecordDetail,
    RecordListOut,
    RecordSummary,
    SearchHit,
    SearchOut,
    SettingsOut,
    SettingsUpdate,
    SpanOut,
    TimelineItemOut,
    TimelineOut,
)
from .timeline import builder as timeline_mod

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_summary(rec: Record, severity: str, labels: list[str]) -> RecordSummary:
    snippet = " ".join(rec.body.split())[:200]
    return RecordSummary(
        id=rec.id,
        source_type=rec.source_type,
        title=rec.display_title,
        occurred_at=rec.occurred_at,
        imported_at=rec.imported_at,
        origin=rec.origin,
        participants=list(rec.participants or []),
        char_count=rec.char_count,
        risk_labels=labels,
        severity=severity,
        snippet=snippet,
    )


def _risk_index(session: Session) -> tuple[dict[int, list[str]], dict[int, str]]:
    labels: dict[int, list[str]] = {}
    severities: dict[int, str] = {}
    for a in session.execute(select(RiskAssessment).where(RiskAssessment.present.is_(True))).scalars():
        labels.setdefault(a.record_id, []).append(a.label)
        severity = assess_mod._severity(
            assess_mod.RiskReport(
                record_id=a.record_id,
                indicators=[
                    assess_mod.Indicator(label=l, present=True, score=0.0, threshold=0.0,
                                         methods={}, explanation="")
                    for l in labels[a.record_id]
                ],
                headline="",
                severity="none",
            )
        )
        severities[a.record_id] = severity
    return labels, severities


def _setting(session: Session, key: str, default: Any = None) -> Any:
    """Read a persisted preference from the *request's* session.

    A separate session would not see rows written earlier in the same request: SQLite in
    WAL mode gives each connection its own view of uncommitted work, so a second session
    would silently return the old value.
    """
    row = session.get(AppSetting, key)
    return row.value if row is not None else default


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthOut, tags=["meta"])
def health(session: Session = Depends(get_session)) -> HealthOut:
    from . import __version__
    from .privacy import LlmPolicy

    count = int(session.execute(select(func.count()).select_from(Record)).scalar_one())
    settings = get_settings()
    return HealthOut(
        status="ok",
        version=__version__,
        records=count,
        spacy_available=ner.spacy_available(settings.spacy_model),
        transformers_available=transformer_clf.transformers_available(),
        sentence_transformers_available=embeddings.sentence_transformers_available(),
        faiss_available=embeddings.faiss_available(),
        llm_enabled=LlmPolicy.from_settings().enabled,
        external_transmission=LlmPolicy.from_settings().enabled,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@router.get("/dashboard", tags=["dashboard"])
def dashboard(session: Session = Depends(get_session)) -> dict[str, Any]:
    labels, severities = _risk_index(session)
    total = int(session.execute(select(func.count()).select_from(Record)).scalar_one())
    by_type = {
        stype: int(
            session.execute(
                select(func.count()).select_from(Record).where(Record.source_type == stype)
            ).scalar_one()
        )
        for (stype,) in session.execute(select(Record.source_type).distinct())
    }

    indicator_counts: dict[str, int] = {label: 0 for label in lexicon.RISK_LABELS}
    for row in session.execute(
        select(RiskAssessment.label, func.count())
        .where(RiskAssessment.present.is_(True))
        .group_by(RiskAssessment.label)
    ):
        indicator_counts[row[0]] = int(row[1])

    insight_counts = {
        status: int(
            session.execute(
                select(func.count()).select_from(Insight).where(Insight.status == status)
            ).scalar_one()
        )
        for status in ("active", "dismissed", "corrected")
    }

    records_with_risk = len(labels)
    return {
        "records": {
            "total": total,
            "by_source_type": by_type,
        },
        "risk": {
            "records_with_indicators": records_with_risk,
            "warning_records": sum(1 for s in severities.values() if s == "warning"),
            "notice_records": sum(1 for s in severities.values() if s == "notice"),
            "indicator_counts": indicator_counts,
            "disclaimer": (
                "Indicators are reported separately and are based on wording. They are not "
                "verdicts about a sender. A single combined score is deliberately not shown here."
            ),
        },
        "graph": {
            "entities": int(session.execute(select(func.count()).select_from(Entity)).scalar_one()),
            "relations": int(session.execute(select(func.count()).select_from(Relation)).scalar_one()),
        },
        "timeline": {
            "events": int(session.execute(select(func.count()).select_from(Event)).scalar_one()),
        },
        "insights": insight_counts,
        "models": _models_status(),
        "privacy": {
            "external_transmission": False,
            "summary": "All analysis runs locally. No record content leaves this machine.",
        },
    }


def _models_status() -> dict[str, Any]:
    settings = get_settings()
    bundle = RiskBundle.load()
    tfidf = tfidf_lr.TfidfRiskModel.load(settings.artifacts_dir)
    return {
        "risk": {
            "rules": {"available": True, "description": "Regex/lexicon baseline"},
            "tfidf_lr": {
                "available": tfidf is not None,
                "description": "TF-IDF + one-vs-rest logistic regression",
                "thresholds": tfidf.thresholds if tfidf else None,
            },
            "distilbert": {
                "available": bundle.transformer is not None,
                "description": "Fine-tuned DistilBERT, multi-label, calibrated",
                "thresholds": bundle.transformer.thresholds if bundle.transformer else None,
                "calibration": bundle.transformer.calibration_method if bundle.transformer else None,
            },
            "notes": bundle.notes,
        },
        "ner": _spacy_status(settings.spacy_model),
        "retrieval": {
            "bm25": {"available": True, "indexed_chunks": bm25_index.get_default_index().size},
            "embeddings": {
                "available": embeddings.get_default_index().size > 0,
                "model": embeddings.get_default_index().model_name,
                "indexed": embeddings.get_default_index().size,
                "faiss": embeddings.faiss_available(),
                "fallback": "hashing" if not embeddings.sentence_transformers_available() else None,
            },
        },
    }


def _spacy_status(model_name: str) -> dict[str, Any]:
    available = ner.spacy_available(model_name)
    return {
        "available": available,
        "model": model_name,
        "note": (
            "spaCy NER is loaded; extraction combines it with the rules baseline."
            if available
            else "spaCy model not installed; rules baseline only."
        ),
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@router.get("/records", response_model=RecordListOut, tags=["records"])
def list_records(
    session: Session = Depends(get_session),
    q: str | None = None,
    source_type: str | None = None,
    risk: bool | None = Query(default=None, description="Only records with indicators"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> RecordListOut:
    labels, severities = _risk_index(session)
    stmt = select(Record).order_by(Record.occurred_at.desc().nullslast(), Record.id.desc())
    if source_type:
        stmt = stmt.where(Record.source_type == source_type)
    if risk is True:
        stmt = stmt.where(
            Record.id.in_(select(RiskAssessment.record_id).where(RiskAssessment.present.is_(True)))
        )
    elif risk is False:
        stmt = stmt.where(
            Record.id.notin_(select(RiskAssessment.record_id).where(RiskAssessment.present.is_(True)))
        )

    if q:
        results = _search_ids(session, q, limit=limit * 3)
        allowed = [rid for rid, _s in results]
        if not allowed:
            return RecordListOut(total=0, items=[])
        stmt = stmt.where(Record.id.in_(allowed))
        ordering = {rid: i for i, (rid, _s) in enumerate(results)}
        rows = list(session.execute(stmt).scalars())
        rows.sort(key=lambda r: ordering.get(r.id, 10_000))
        total = len(rows)
        rows = rows[offset : offset + limit]
    else:
        total = int(session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one())
        rows = list(session.execute(stmt.offset(offset).limit(limit)).scalars())

    items = [
        _record_summary(r, severities.get(r.id, "none"), labels.get(r.id, [])) for r in rows
    ]
    return RecordListOut(total=total, items=items)


def _search_ids(session: Session, query: str, limit: int = 50) -> list[tuple[int, float]]:
    result = answering.retrieve(session, query, top_k=limit)
    scores: dict[int, float] = {}
    for hit in result.hits:
        scores[hit.record_id] = max(scores.get(hit.record_id, 0.0), hit.score)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]


@router.get("/records/{record_id}", response_model=RecordDetail, tags=["records"])
def get_record(record_id: int, session: Session = Depends(get_session)) -> RecordDetail:
    rec = session.get(Record, record_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Record not found")

    mentions = list(
        session.execute(select(EntityMention).where(EntityMention.record_id == record_id)).scalars()
    )
    entities = {e.id: e for e in session.execute(select(Entity)).scalars()}
    spans = [
        SpanOut(
            text=m.text,
            label=m.label,
            start_char=m.start_char,
            end_char=m.end_char,
            confidence=m.confidence,
            extractor=m.extractor,
            entity_id=m.entity_id,
            entity_name=entities[m.entity_id].name if m.entity_id in entities else None,
        )
        for m in mentions
    ]

    events = [
        {
            "id": e.id,
            "title": e.title,
            "description": e.description,
            "kind": e.kind,
            "certainty": e.certainty,
            "start_at": e.start_at,
            "end_at": e.end_at,
            "precision": e.precision,
            "raw_expression": e.raw_expression,
            "start_char": e.start_char,
            "end_char": e.end_char,
            "confidence": e.confidence,
            "extractor": e.extractor,
            "note": e.note,
        }
        for e in session.execute(
            select(Event).where(Event.record_id == record_id)
        ).scalars()
    ]

    assessments = list(
        session.execute(
            select(RiskAssessment).where(RiskAssessment.record_id == record_id)
        ).scalars()
    )
    indicators = [
        {
            "label": a.label,
            "display": assess_mod.LABEL_DISPLAY.get(a.label, a.label),
            "present": a.present,
            "score": a.score,
            "threshold": a.threshold,
            "methods": {a.method: a.score},
            "explanation": a.explanation,
            "evidence": a.evidence or [],
            "calibrated": a.calibrated,
        }
        for a in sorted(assessments, key=lambda x: x.label)
    ]

    report = assess_mod.RiskReport(
        record_id=record_id,
        indicators=[
            assess_mod.Indicator(
                label=a.label, present=a.present, score=a.score, threshold=a.threshold,
                methods={a.method: a.score}, explanation=a.explanation,
                evidence=a.evidence or [], calibrated=a.calibrated,
            )
            for a in assessments
        ],
        headline="",
        severity="none",
    )
    if report.indicators:
        report.headline = assess_mod._headline(report)
        report.severity = assess_mod._severity(report)

    chunks = [
        {"ordinal": c.ordinal, "start_char": c.start_char, "end_char": c.end_char}
        for c in session.execute(
            select(Chunk).where(Chunk.record_id == record_id).order_by(Chunk.ordinal)
        ).scalars()
    ]

    return RecordDetail(
        id=rec.id,
        source_type=rec.source_type,
        title=rec.display_title,
        body=rec.body,
        occurred_at=rec.occurred_at,
        imported_at=rec.imported_at,
        origin=rec.origin,
        external_id=rec.external_id,
        participants=list(rec.participants or []),
        extra=dict(rec.extra or {}),
        spans=spans,
        events=events,
        indicators=indicators,
        overall_estimate=assess_mod.overall_estimate(report),
        entities=[
            {
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "mention_count": e.mention_count,
                "resolution_confidence": round(e.resolution_confidence, 3),
                "surface_forms": (e.attributes or {}).get("surface_forms", []),
            }
            for e in entities.values()
            if any(m.entity_id == e.id for m in mentions)
        ],
        backreferences=[
            {"start_char": s, "end_char": e, "text": t} for s, e, t in find_backreferences(rec.body)
        ],
        chunks=chunks,
    )


@router.delete("/records/{record_id}", tags=["records"])
def delete_record(record_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    report = delete_record_cascade(session, record_id)
    if report.problems and not report.removed.get("records"):
        raise HTTPException(status_code=404, detail=report.problems[0])
    build_indexes(session)
    generate_insights(session)
    return {
        "deleted": report.as_dict(),
        "message": (
            "The record and everything derived from it were deleted: extracted entities, graph "
            "edges, events, risk assessments, insights and search index entries."
        ),
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@router.get("/search", response_model=SearchOut, tags=["retrieval"])
def search(
    session: Session = Depends(get_session),
    q: str = Query(min_length=1, max_length=300),
    mode: Literal["hybrid", "lexical", "semantic"] | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> SearchOut:
    settings = get_settings()
    chosen = mode or _setting(session, "user.retrieval_mode", settings.default_retrieval_mode)
    result = answering.retrieve(session, q, top_k=limit, mode=chosen)
    records = {r.id: r for r in session.execute(select(Record)).scalars()}
    total = int(session.execute(select(func.count()).select_from(Record)).scalar_one())

    hits = [
        SearchHit(
            record_id=h.record_id,
            chunk_id=h.chunk_id,
            text=h.text,
            start_char=h.start_char,
            end_char=h.end_char,
            score=h.score,
            rank=h.rank,
            lexical_rank=h.lexical_rank,
            semantic_rank=h.semantic_rank,
            agreement=h.agreement,
            title=records[h.record_id].display_title if h.record_id in records else None,
            source_type=records[h.record_id].source_type if h.record_id in records else None,
            occurred_at=records[h.record_id].occurred_at if h.record_id in records else None,
        )
        for h in result.hits
    ]
    return SearchOut(
        query=q,
        mode=result.mode,
        hits=hits,
        channels=result.channels,
        degraded=result.degraded,
        explanation=result.explanation,
        total_records=total,
    )


@router.post("/ask", response_model=AskOut, tags=["retrieval"])
def ask(payload: AskRequest, session: Session = Depends(get_session)) -> AskOut:
    settings = get_settings()
    mode = _setting(session, "user.retrieval_mode", payload.mode) or payload.mode
    answer = answering.answer_question(
        session, payload.question, top_k=payload.top_k, mode=mode
    )

    # Never emit a claim we would not stand behind.
    if not assess_mod.assert_cautious_wording(answer.answer):
        answer.answer = assess_mod._sanitize(answer.answer)

    bodies = {r.id: r.body for r in session.execute(select(Record)).scalars()}
    problems = answering.qa_core.enforce_citation_integrity(answer, bodies)
    if problems:
        logger.warning("citation integrity problems=%d", len(problems))

    return AskOut(**answer.as_dict())


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@router.get("/graph", response_model=GraphOut, tags=["graph"])
def get_graph(
    session: Session = Depends(get_session),
    record_ids: str | None = Query(default=None, description="comma-separated record ids"),
    max_nodes: int = Query(default=120, ge=10, le=600),
) -> GraphOut:
    entities = list(session.execute(select(Entity).order_by(Entity.mention_count.desc())).scalars())
    mentions = list(session.execute(select(EntityMention)).scalars())
    relations = list(session.execute(select(Relation)).scalars())
    records = list(session.execute(select(Record)).scalars())

    g = graph_mod.to_networkx(entities, mentions, relations, records)
    if record_ids:
        ids = {int(x) for x in record_ids.split(",") if x.strip().isdigit()}
        g = graph_mod.subgraph_for_records(g, ids)

    entity_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "entity"]
    entity_nodes.sort(key=lambda n: -g.nodes[n].get("mention_count", 0))
    keep = set(entity_nodes[:max_nodes])
    record_nodes = {n for n, d in g.nodes(data=True) if d.get("kind") == "record"}
    # Keep record nodes only for the entities we kept, to bound the payload.
    record_nodes = {
        n for n in record_nodes
        if any(g.has_edge(n, e) or g.has_edge(e, n) for e in keep)
    }
    keep |= record_nodes
    sub = g.subgraph(keep)

    nodes = [
        GraphNode(
            id=n,
            kind=d.get("kind", "entity"),
            label=d.get("label", ""),
            entity_type=d.get("entity_type"),
            mention_count=d.get("mention_count"),
            source_type=d.get("source_type"),
            occurred_at=d.get("occurred_at"),
            record_id=d.get("record_id"),
            resolution_confidence=d.get("resolution_confidence"),
        )
        for n, d in sub.nodes(data=True)
    ]
    edges = [
        GraphEdge(
            source=u,
            target=v,
            relation_type=d.get("relation_type", "co_mentioned"),
            confidence=round(float(d.get("confidence", 0.0)), 3),
            record_id=d.get("record_id"),
            start_char=d.get("start_char"),
            end_char=d.get("end_char"),
            method=d.get("method", "co_mention"),
            weight=round(float(d.get("weight", 1.0)), 3),
        )
        for u, v, d in sub.edges(data=True)
    ]

    return GraphOut(
        nodes=nodes,
        edges=edges,
        centrality=graph_mod.centrality(sub),
        explanation=(
            "Edges are shown only when they are evidenced by a record. A dashed edge in the UI "
            "means the link is a co-mention heuristic rather than an explicit statement in the text."
        ),
    )


@router.get("/entities/{entity_id}", tags=["graph"])
def get_entity(entity_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    entity = session.get(Entity, entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    aliases = [
        a.alias
        for a in session.execute(
            select(EntityAlias).where(EntityAlias.entity_id == entity_id)
        ).scalars()
    ]
    mentions = [
        {
            "record_id": m.record_id,
            "text": m.text,
            "start_char": m.start_char,
            "end_char": m.end_char,
            "confidence": m.confidence,
            "extractor": m.extractor,
        }
        for m in session.execute(
            select(EntityMention).where(EntityMention.entity_id == entity_id)
        ).scalars()
    ]
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.entity_type,
        "mention_count": entity.mention_count,
        "resolution_confidence": round(entity.resolution_confidence, 4),
        "aliases": aliases,
        "merges": [
            {
                "target": entity.name,
                "alias": a,
                "kind": "merge",
                "score": round(entity.resolution_confidence, 4),
                "detail": "Merged during entity resolution.",
            }
            for a in aliases
        ],
        "mentions": mentions,
        "caveat": (
            "Entity resolution merges similar names and can be wrong. Correct it by dismissing "
            "the related insight or editing the record."
        ),
    }


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


@router.get("/timeline", response_model=TimelineOut, tags=["timeline"])
def get_timeline(
    session: Session = Depends(get_session),
    record_ids: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> TimelineOut:
    stmt = select(Record).order_by(Record.occurred_at.asc().nullslast(), Record.id.asc())
    if record_ids:
        ids = {int(x) for x in record_ids.split(",") if x.strip().isdigit()}
        stmt = stmt.where(Record.id.in_(ids))
    records = list(session.execute(stmt.limit(limit)).scalars())

    events_by_record: dict[int, list[Event]] = {}
    for e in session.execute(select(Event)).scalars():
        events_by_record.setdefault(e.record_id, []).append(e)
    risk_by_record: dict[int, list[str]] = {}
    for a in session.execute(select(RiskAssessment).where(RiskAssessment.present.is_(True))).scalars():
        risk_by_record.setdefault(a.record_id, []).append(a.label)

    items = timeline_mod.build_timeline(records, events_by_record, risk_by_record)
    return TimelineOut(
        items=[TimelineItemOut(**i.as_dict()) for i in items],
        summary=timeline_mod.lane_summary(items),
    )


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


@router.get("/risk", tags=["risk"])
def risk_overview(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = list(
        session.execute(
            select(RiskAssessment).where(RiskAssessment.present.is_(True))
        ).scalars()
    )
    by_record: dict[int, list[RiskAssessment]] = {}
    for a in rows:
        by_record.setdefault(a.record_id, []).append(a)

    records = {r.id: r for r in session.execute(select(Record)).scalars()}
    out: list[dict[str, Any]] = []
    for rid, items in sorted(by_record.items()):
        rec = records.get(rid)
        if rec is None:
            continue
        report = assess_mod.RiskReport(
            record_id=rid,
            indicators=[
                assess_mod.Indicator(
                    label=a.label, present=a.present, score=a.score, threshold=a.threshold,
                    methods={a.method: a.score}, explanation=a.explanation,
                    evidence=a.evidence or [], calibrated=a.calibrated,
                )
                for a in sorted(items, key=lambda x: x.label)
            ],
            headline="",
            severity="none",
        )
        report.headline = assess_mod._headline(report)
        report.severity = assess_mod._severity(report)
        out.append(
            {
                "record_id": rid,
                "title": rec.display_title,
                "source_type": rec.source_type,
                "occurred_at": rec.occurred_at.isoformat() if rec.occurred_at else None,
                "severity": report.severity,
                "headline": report.headline,
                "indicators": [i.as_dict() for i in report.indicators],
                "overall_estimate": assess_mod.overall_estimate(report),
            }
        )

    out.sort(key=lambda d: (d["severity"] != "warning", -(d["overall_estimate"]["value"])))
    return {
        "records": out,
        "models": _models_status()["risk"],
        "method_note": (
            "Each indicator is scored independently and reported separately. The overall figure "
            "is an explicitly derived estimate shown for sorting only - it is not a probability "
            "of fraud."
        ),
        "benign_note": (
            "Legitimate messages can contain urgent wording. Positive indicators mean 'review "
            "this', not 'this is an attack'."
        ),
    }


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


@router.get("/insights", response_model=list[InsightOut], tags=["insights"])
def list_insights(
    session: Session = Depends(get_session),
    status: Literal["active", "dismissed", "corrected", "all"] = "active",
    kind: str | None = None,
) -> list[InsightOut]:
    stmt = select(Insight).order_by(Insight.created_at.desc())
    if status != "all":
        stmt = stmt.where(Insight.status == status)
    if kind:
        stmt = stmt.where(Insight.kind == kind)
    return [
        InsightOut(
            id=i.id,
            kind=i.kind,
            severity=i.severity,
            title=i.title,
            statement=i.statement,
            facts=i.facts or [],
            evidence=i.evidence or [],
            confidence=i.confidence,
            confidence_basis=i.confidence_basis,
            status=i.status,
            user_note=i.user_note,
            record_ids=i.record_ids or [],
            entity_ids=i.entity_ids or [],
            created_at=i.created_at,
        )
        for i in session.execute(stmt).scalars()
    ]


@router.patch("/insights/{insight_id}", response_model=InsightOut, tags=["insights"])
def update_insight(
    insight_id: int, payload: InsightUpdate, session: Session = Depends(get_session)
) -> InsightOut:
    insight = session.get(Insight, insight_id)
    if insight is None:
        raise HTTPException(status_code=404, detail="Insight not found")
    if payload.status is not None:
        insight.status = payload.status
    if payload.user_note is not None:
        insight.user_note = payload.user_note
        if insight.status == "active":
            insight.status = "corrected"
    session.flush()
    return InsightOut(
        id=insight.id,
        kind=insight.kind,
        severity=insight.severity,
        title=insight.title,
        statement=insight.statement,
        facts=insight.facts or [],
        evidence=insight.evidence or [],
        confidence=insight.confidence,
        confidence_basis=insight.confidence_basis,
        status=insight.status,
        user_note=insight.user_note,
        record_ids=insight.record_ids or [],
        entity_ids=insight.entity_ids or [],
        created_at=insight.created_at,
    )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@router.post("/import/demo", response_model=ImportResult, tags=["import"])
def import_demo(
    session: Session = Depends(get_session),
    reanalyze: bool = True,
) -> ImportResult:
    stats = load_demo(session, replace=True)
    if reanalyze:
        run_pipeline(session)
    return ImportResult(inserted=stats.inserted, skipped=stats.skipped, record_ids=stats.ids)


@router.post("/import/files", response_model=ImportResult, tags=["import"])
async def import_files(
    files: list[UploadFile] = File(..., description=".txt, .md, .eml, .csv, .json or .ics"),
    reanalyze: bool = True,
    session: Session = Depends(get_session),
) -> ImportResult:
    from .ingest.demo import DemoRecord

    supported = parsers.SUPPORTED_EXTENSIONS
    records: list[DemoRecord] = []
    errors: list[str] = []
    file_reports: list[dict[str, Any]] = []

    for upload in files:
        name = upload.filename or "upload"
        ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
        if ext not in supported:
            errors.append(
                f"{name}: unsupported file type. Supported: {', '.join(sorted(supported))}"
            )
            continue
        raw = await upload.read()
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            errors.append(f"{name}: could not decode as text")
            continue
        try:
            parsed = parsers.parse(name, text)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        records.extend(parsed)
        file_reports.append({"filename": name, "records": len(parsed)})

    stats = ingest_records(
        session, records, origin=f"upload:{name}", replace=False
    ) if records else None
    if stats is None:
        return ImportResult(inserted=0, skipped=0, record_ids=[], files=file_reports, errors=errors)

    if reanalyze:
        run_pipeline(session)
    return ImportResult(
        inserted=stats.inserted,
        skipped=stats.skipped,
        record_ids=stats.ids,
        files=file_reports,
        errors=errors,
    )


@router.get("/import/status", tags=["import"])
def import_status() -> dict[str, Any]:
    return {
        "integrations": [
            {
                "key": "gmail",
                "label": "Gmail",
                "configured": False,
                "status": "not configured",
                "note": "DigitalMind never connects to third-party services. Import .eml or .mbox exports instead.",
            },
            {
                "key": "outlook",
                "label": "Outlook / Microsoft 365",
                "configured": False,
                "status": "not configured",
                "note": "No OAuth connection is configured. Export messages and upload them here.",
            },
            {
                "key": "slack",
                "label": "Slack",
                "configured": False,
                "status": "not configured",
                "note": "Export the channel or thread as text or CSV and upload it.",
            },
            {
                "key": "notion",
                "label": "Notion",
                "configured": False,
                "status": "not configured",
                "note": "Export pages as Markdown or CSV and upload them.",
            },
            {
                "key": "calendar",
                "label": "Calendar",
                "configured": False,
                "status": "not configured",
                "note": "Upload .ics files. DigitalMind reads them locally.",
            },
        ],
        "supported_files": sorted(parsers.SUPPORTED_EXTENSIONS),
        "privacy_note": (
            "Uploads are parsed locally. Files are not sent to any external service, and the "
            "application log never records record contents."
        ),
    }


# ---------------------------------------------------------------------------
# Settings / privacy
# ---------------------------------------------------------------------------


@router.get("/settings", response_model=SettingsOut, tags=["settings"])
def get_settings_out(session: Session = Depends(get_session)) -> SettingsOut:
    settings = get_settings()
    sources = [
        {
            "key": s.key,
            "label": s.label,
            "kind": s.kind,
            "configured": s.configured,
            "record_count": s.record_count,
        }
        for s in session.execute(select(DataSource).order_by(DataSource.key)).scalars()
    ]
    return SettingsOut(
        retrieval_mode=_setting(session, "user.retrieval_mode", settings.default_retrieval_mode),
        min_evidence_confidence=float(
            _setting(session, "user.min_evidence_confidence", settings.min_evidence_confidence)
        ),
        risk_models=_models_status()["risk"],
        spacy=_models_status()["ner"],
        embeddings=_models_status()["retrieval"]["embeddings"],
        privacy=privacy_report(session),
        data_sources=sources,
    )


@router.patch("/settings", response_model=SettingsOut, tags=["settings"])
def update_settings(payload: SettingsUpdate, session: Session = Depends(get_session)) -> SettingsOut:
    def upsert(key: str, value: Any) -> None:
        row = session.get(AppSetting, key)
        if row is None:
            session.add(AppSetting(key=key, value=value))
        else:
            row.value = value
        session.flush()

    if payload.retrieval_mode is not None:
        upsert("user.retrieval_mode", payload.retrieval_mode)
    if payload.min_evidence_confidence is not None:
        upsert("user.min_evidence_confidence", payload.min_evidence_confidence)
    if payload.allow_external_llm is not None:
        settings = get_settings()
        if payload.allow_external_llm and not settings.llm_endpoint:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No LLM endpoint is configured. DigitalMind works fully without one, and "
                    "leaving this off is the default."
                ),
            )
        upsert("user.allow_external_llm", payload.allow_external_llm)

    return get_settings_out(session)


@router.get("/privacy", tags=["settings"])
def get_privacy(session: Session = Depends(get_session)) -> dict[str, Any]:
    return privacy_report(session)


@router.post("/settings/delete-all", tags=["settings"])
def delete_all(session: Session = Depends(get_session)) -> dict[str, Any]:
    counts = delete_all_data(session)
    return {
        "deleted": counts,
        "message": (
            "All records, entities, graph edges, events, risk assessments, insights and index "
            "entries were deleted from the local database."
        ),
    }


# ---------------------------------------------------------------------------
# Analysis control
# ---------------------------------------------------------------------------


@router.post("/analysis/reindex", tags=["analysis"])
def reindex(session: Session = Depends(get_session)) -> dict[str, Any]:
    return build_indexes(session)


@router.post("/analysis/rerun", tags=["analysis"])
def rerun(session: Session = Depends(get_session)) -> dict[str, Any]:
    return run_pipeline(session)


@router.post("/analysis/resolve", tags=["analysis"])
def rerun_resolution(session: Session = Depends(get_session)) -> dict[str, Any]:
    summary = resolve_entities(session)
    return summary

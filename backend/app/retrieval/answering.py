"""QA service: turns a question into a cited answer using the retrieval stack.

Answers are assembled from retrieved spans, never generated. Every sentence that makes a
claim is followed by its citation, and the optional LLM layer (off by default) may only
rephrase - :func:`~app.retrieval.qa.enforce_citation_integrity` verifies the result.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..logging_config import get_logger
from ..models import Chunk, Entity, EntityMention, Record, RiskAssessment
from . import qa as qa_core
from . import hybrid

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 400


def _records_by_id(session: Session) -> dict[int, Record]:
    return {r.id: r for r in session.execute(select(Record)).scalars()}


def _chunk_lookup(session: Session) -> dict[int, tuple[int, int, str, int, int]]:
    return {
        c.id: (c.record_id, c.ordinal, c.text, c.start_char, c.end_char)
        for c in session.execute(select(Chunk)).scalars()
    }


def _method_block(result: hybrid.HybridResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "channels": result.channels,
        "degraded": result.degraded,
        "explanation": result.explanation,
        "description": (
            "BM25 keyword search fused with sentence-embedding similarity"
            if result.mode == "hybrid"
            else f"{result.mode} search"
        ),
    }


def retrieve(
    session: Session,
    query: str,
    top_k: int = 8,
    mode: str = "hybrid",
    allowed_records: set[int] | None = None,
) -> hybrid.HybridResult:
    from . import bm25_index, embeddings

    settings = _settings()
    return hybrid.search(
        query,
        chunk_lookup=_chunk_lookup(session),
        lexical_index=bm25_index.get_default_index(),
        semantic_index=embeddings.get_default_index(),
        embeddings_model=settings.embeddings_model,
        mode=mode,
        top_k=top_k,
        allowed_records=allowed_records,
    )


def _settings():
    from ..config import get_settings

    return get_settings()


def answer_question(session: Session, question: str, top_k: int = 8, mode: str = "hybrid") -> qa_core.Answer:
    """Answer a question with citations, or return the cautious empty result."""
    intent = qa_core.classify_intent(question)
    topic = qa_core.extract_topic(question)
    records = _records_by_id(session)

    coverage = {
        "records_in_database": len(records),
        "date_range": _date_range(records),
        "source_types": sorted({r.source_type for r in records.values()}),
        "integrations": "none configured - only imported and demo data is searched",
        "query_used": topic,
    }

    if intent == qa_core.INTENT_WHAT_BEFORE:
        return _answer_before(session, question, topic, records, coverage, top_k, mode)

    result = retrieve(session, topic or question, top_k=top_k, mode=mode)
    method = _method_block(result)

    if intent == qa_core.INTENT_WHO:
        return answer_who(session, question, mode=mode)

    if not result.hits:
        return qa_core.no_match_answer(
            question,
            intent,
            topic,
            {"records_considered": len(records)},
            coverage,
            method,
        )

    if intent == qa_core.INTENT_RISK:
        return _answer_risk(session, question, result, records, coverage, method)

    return _answer_generic(question, intent, result, records, coverage, method)


def _answer_before(
    session: Session,
    question: str,
    topic: str,
    records: dict[int, Record],
    coverage: dict[str, Any],
    top_k: int,
    mode: str,
) -> qa_core.Answer:
    """'What happened before this payment request?'

    The target record is the most recent record that actually carries a payment-request
    flag, unless the question names one. We then walk the timeline backwards, which keeps
    the ordering explicit and cited.
    """
    from ..timeline import builder as timeline_mod

    assessments = list(session.execute(select(RiskAssessment)).scalars())
    flagged_labels: dict[int, set[str]] = {}
    for a in assessments:
        if a.present:
            flagged_labels.setdefault(a.record_id, set()).add(a.label)

    payment_records = sorted(
        (rid for rid, labels in flagged_labels.items() if "payment_request" in labels),
        key=lambda rid: (
            # A payment demand paired with a threat or a deadline is almost certainly the
            # record the user means; a routine invoice mention is not.
            -len(flagged_labels[rid] & {"threat_language", "urgency"}),
            records[rid].occurred_at or records[rid].imported_at,
        ),
    ) if records else []

    target_id = None
    named = re.findall(r"\b(\d{2,6})\b", question)
    if named:
        target_id = next((int(n) for n in named if int(n) in records), None)
    if target_id is None and payment_records:
        target_id = payment_records[0]

    if target_id is None or target_id not in records:
        result = retrieve(session, topic or question, top_k=top_k, mode=mode)
        return qa_core.no_match_answer(
            question,
            qa_core.INTENT_WHAT_BEFORE,
            topic,
            {"records_considered": len(records)},
            coverage,
            _method_block(result),
        )

    events_by_record: dict[int, list] = {}
    from ..models import Event

    for ev in session.execute(select(Event)).scalars():
        events_by_record.setdefault(ev.record_id, []).append(ev)

    risk_by_record: dict[int, list[str]] = {
        rid: sorted(labels) for rid, labels in flagged_labels.items()
    }

    timeline = timeline_mod.build_timeline(records.values(), events_by_record, risk_by_record)
    prior = timeline_mod.records_before(timeline, target_id, window_days=45, limit=10)

    target = records[target_id]
    citations = [
        qa_core.Citation(
            record_id=target.id,
            source_type=target.source_type,
            title=target.display_title,
            occurred_at=target.occurred_at.isoformat() if target.occurred_at else None,
            start_char=0,
            end_char=min(len(target.body), MAX_CONTEXT_CHARS),
            quote=_summarize(target.body),
            retriever="timeline",
        )
    ]

    if not prior:
        answer = (
            f"No matching discussion was found in the records analyzed. I looked at the "
            f"45 days before “{target.display_title}” and found no earlier record to cite. "
            "That is not evidence that nothing happened beforehand: earlier records may simply "
            "not be imported, or may not carry timestamps."
        )
        return qa_core.Answer(
            question=question,
            intent=qa_core.INTENT_WHAT_BEFORE,
            answer=answer,
            citations=citations,
            coverage=coverage,
            method={"mode": "timeline", "description": "recorded-timestamp ordering"},
            caveat=qa_core.CAVEAT_THIN,
            no_match=True,
            confidence=0.0,
            confidence_basis="No earlier record was available to cite.",
        )

    lines = [
        f"Before “{target.display_title}” ({_fmt(target.occurred_at)}), the records analyzed contain "
        f"{len(prior)} related items, in recorded order:"
    ]
    facts: list[dict[str, Any]] = []
    for item in prior:
        when = _fmt(item.start_at)
        lane_note = {
            timeline_mod.RECORDED: "recorded timestamp",
            timeline_mod.EXPLICIT: f"date written in the text ({item.raw_expression or 'absolute'})",
            timeline_mod.INFERRED: f"relative wording in the text ({item.raw_expression}) - shown as a range",
        }[item.lane]
        lines.append(f"- {when} · {item.title} [{item.source_type}] — {lane_note}")
        rec = records.get(item.record_id)
        citations.append(
            qa_core.Citation(
                record_id=item.record_id,
                source_type=item.source_type,
                title=item.title,
                occurred_at=item.start_at.isoformat() if item.start_at else None,
                start_char=item.start_char or 0,
                end_char=item.end_char if item.end_char is not None else min(
                    len(rec.body) if rec else 0, MAX_CONTEXT_CHARS
                ),
                quote=_summarize(rec.body) if rec else "",
                retriever=f"timeline:{item.lane}",
                score=item.confidence,
            )
        )
        facts.append(
            {
                "record_id": item.record_id,
                "start_char": item.start_char,
                "end_char": item.end_char,
                "quote": item.raw_expression or item.title,
                "note": item.explanation,
                "lane": item.lane,
            }
        )

    lines.append(
        "This ordering comes from timestamps the source systems recorded. Items marked as "
        "relative wording were placed in a range, not at a specific moment."
    )

    confidence, basis = qa_core.confidence_for(len(citations), len(prior))
    return qa_core.Answer(
        question=question,
        intent=qa_core.INTENT_WHAT_BEFORE,
        answer="\n".join(lines),
        citations=citations,
        facts=facts,
        interpretation=[
            "The ordering above is a sequence of records, not a causal account. Nothing here "
            "shows that earlier records caused the later one."
        ],
        coverage=coverage,
        method={"mode": "timeline", "description": "recorded-timestamp ordering with range-estimated items"},
        caveat=(
            "Only records with a source timestamp appear in the recorded lane. A record with no "
            "timestamp is not evidence that it did not happen."
        ),
        confidence=confidence,
        confidence_basis=basis,
    )


def _answer_risk(
    session: Session,
    question: str,
    result: hybrid.HybridResult,
    records: dict[int, Record],
    coverage: dict[str, Any],
    method: dict[str, Any],
) -> qa_core.Answer:
    """Answer a risk question with indicator-level evidence, never a single verdict."""
    record_ids = list(dict.fromkeys(h.record_id for h in result.hits))[:6]
    assessments = list(session.execute(select(RiskAssessment)).scalars())
    by_record: dict[int, list[RiskAssessment]] = {}
    for a in assessments:
        by_record.setdefault(a.record_id, []).append(a)

    flagged_records = [rid for rid in record_ids if any(a.present for a in by_record.get(rid, []))]
    lines: list[str] = []
    citations: list[qa_core.Citation] = []
    facts: list[dict[str, Any]] = []

    if not flagged_records:
        return qa_core.no_match_answer(
            question,
            qa_core.INTENT_RISK,
            qa_core.extract_topic(question),
            {"records_considered": len(records)},
            coverage,
            method,
        )

    lines.append(
        f"{len(flagged_records)} of the records I retrieved contain at least one risk indicator. "
        "Each is listed separately with the wording that triggered it:"
    )
    for rid in flagged_records:
        rec = records.get(rid)
        if rec is None:
            continue
        present = [a for a in by_record[rid] if a.present]
        names = ", ".join(sorted(a.label.replace("_", " ") for a in present))
        lines.append(f"- “{rec.display_title}” — {names}")
        for a in present:
            for e in (a.evidence or []):
                if e.get("start_char") is None:
                    continue
                citations.append(
                    qa_core.Citation(
                        record_id=rid,
                        source_type=rec.source_type,
                        title=rec.display_title,
                        occurred_at=rec.occurred_at.isoformat() if rec.occurred_at else None,
                        start_char=int(e["start_char"]),
                        end_char=int(e["end_char"]),
                        quote=str(e.get("text") or ""),
                        retriever="risk:rules",
                    )
                )
                facts.append(
                    {
                        "record_id": rid,
                        "start_char": e["start_char"],
                        "end_char": e["end_char"],
                        "quote": e.get("text"),
                        "note": e.get("note"),
                        "indicator": a.label,
                    }
                )

    lines.append(
        "These are indicators based on wording. They are not a judgement about any sender, "
        "and legitimate messages can contain this phrasing."
    )

    confidence, basis = qa_core.confidence_for(len(citations), len(flagged_records))
    return qa_core.Answer(
        question=question,
        intent=qa_core.INTENT_RISK,
        answer="\n".join(lines),
        citations=citations,
        facts=facts,
        interpretation=[],
        coverage=coverage,
        method=method,
        caveat=(
            "Indicators are reported separately and never combined into a verdict. "
            "Review the cited wording yourself before acting on it."
        ),
        confidence=confidence,
        confidence_basis=basis,
    )


def _answer_generic(
    question: str,
    intent: str,
    result: hybrid.HybridResult,
    records: dict[int, Record],
    coverage: dict[str, Any],
    method: dict[str, Any],
) -> qa_core.Answer:
    """Default answer: a ranked list of records with the matching sentences quoted."""
    grouped: dict[int, list[hybrid.FusedHit]] = {}
    for hit in result.hits:
        grouped.setdefault(hit.record_id, []).append(hit)

    ordered = sorted(
        grouped.items(),
        key=lambda kv: (-max(h.score for h in kv[1]), kv[0]),
    )[:8]

    if not ordered:
        return qa_core.no_match_answer(
            question, intent, coverage.get("query_used", ""),
            {"records_considered": len(records)}, coverage, method,
        )

    citations: list[qa_core.Citation] = []
    facts: list[dict[str, Any]] = []
    lines: list[str] = [
        f"{len(ordered)} record(s) in the analyzed data match. Each quote below is copied "
        "from the record at the cited position:"
    ]

    for rid, hits in ordered:
        rec = records.get(rid)
        if rec is None:
            continue
        best = max(hits, key=lambda h: h.score)
        span = _best_span(rec.body, best.text)
        when = _fmt(rec.occurred_at)
        lines.append(f"- “{rec.display_title}” ({rec.source_type}, {when})")
        citations.append(
            qa_core.Citation(
                record_id=rid,
                source_type=rec.source_type,
                title=rec.display_title,
                occurred_at=rec.occurred_at.isoformat() if rec.occurred_at else None,
                start_char=span[0],
                end_char=span[1],
                quote=rec.body[span[0] : span[1]],
                retriever=best.agreement,
                score=best.score,
            )
        )
        facts.append(
            {
                "record_id": rid,
                "start_char": span[0],
                "end_char": span[1],
                "quote": rec.body[span[0] : span[1]],
                "note": f"Matched via {best.agreement.replace('_', ' ')}.",
            }
        )

    agreement_both = sum(1 for h in result.hits if h.agreement == "both")
    lines.append(
        f"Retrieval note: {result.explanation}"
        + (f" {agreement_both} result(s) were ranked by both channels." if agreement_both else "")
    )

    confidence, basis = qa_core.confidence_for(len(citations), len(ordered))
    return qa_core.Answer(
        question=question,
        intent=intent,
        answer="\n".join(lines),
        citations=citations,
        facts=facts,
        interpretation=[],
        coverage=coverage,
        method=method,
        caveat=(
            "This is a retrieval result, not a summary of meaning. DigitalMind does not "
            "generate an answer beyond the quoted text."
        ),
        confidence=confidence,
        confidence_basis=basis,
    )


def _best_span(body: str, chunk_text: str) -> tuple[int, int]:
    """Narrow a chunk to the sentence(s) that overlap the query terms."""
    if not chunk_text:
        return 0, min(len(body), MAX_CONTEXT_CHARS)
    idx = body.find(chunk_text[:80])
    if idx < 0:
        return 0, min(len(body), MAX_CONTEXT_CHARS)
    end = idx + len(chunk_text)
    # Trim to sentences within the chunk for a tighter citation.
    snippet = body[idx:end]
    sentences = re.split(r"(?<=[.!?])\s+", snippet)
    acc = ""
    offset = idx
    for sentence in sentences:
        if len(acc) + len(sentence) > MAX_CONTEXT_CHARS:
            break
        acc = (acc + " " + sentence).strip() if acc else sentence
        offset_end = offset + len(acc)
        if len(acc) >= MAX_CONTEXT_CHARS * 0.6:
            break
    return idx, min(end, idx + max(len(acc), 1))


def _summarize(body: str, limit: int = 240) -> str:
    text = " ".join(body.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _fmt(value) -> str:
    if value is None:
        return "no timestamp"
    return value.strftime("%Y-%m-%d %H:%M")


def _date_range(records: dict[int, Record]) -> dict[str, str | None] | None:
    times = [r.occurred_at for r in records.values() if r.occurred_at]
    if not times:
        return None
    return {"from": min(times).isoformat(), "to": max(times).isoformat()}


def answer_who(session: Session, question: str, mode: str = "hybrid") -> qa_core.Answer:
    """Answer 'who was involved' from the entity graph rather than from text search."""
    topic = qa_core.extract_topic(question)
    result = retrieve(session, topic or question, top_k=8, mode=mode)
    record_ids = list(dict.fromkeys(h.record_id for h in result.hits))[:6]

    entities = {e.id: e for e in session.execute(select(Entity)).scalars()}

    involved: dict[int, int] = {}
    if record_ids:
        for entity_id, in session.execute(
            select(EntityMention.entity_id).where(EntityMention.record_id.in_(record_ids))
        ):
            if entity_id is not None:
                involved[entity_id] = involved.get(entity_id, 0) + 1

    # "Who" questions want people and organizations. Dates, amounts and event phrases are
    # extracted entities too, but listing them here would bury the answer.
    involved = {
        eid: count
        for eid, count in involved.items()
        if eid in entities and entities[eid].entity_type in {"PERSON", "ORG", "PRODUCT"}
    }

    ranked = sorted(involved.items(), key=lambda kv: (-kv[1], entities[kv[0]].name))
    if not ranked:
        return qa_core.no_match_answer(
            question, qa_core.INTENT_WHO, topic,
            {"records_considered": len(record_ids)}, {"query_used": topic},
            {"mode": "entity_graph", "description": "entity graph over retrieved records"},
        )
    if not ranked:
        return qa_core.no_match_answer(
            question, qa_core.INTENT_WHO, topic,
            {"records_considered": len(record_ids)}, {"query_used": topic},
            {"mode": mode, "description": "entity graph over retrieved records"},
        )

    records = _records_by_id(session)
    names = []
    citations: list[qa_core.Citation] = []
    for entity_id, count in ranked[:6]:
        entity = entities.get(entity_id)
        if entity is None or entity.entity_type not in {"PERSON", "ORG", "PRODUCT"}:
            continue
        names.append(f"{entity.name} ({entity.entity_type.lower()}, {count} mention(s))")
        mention = session.execute(
            select(EntityMention.record_id, EntityMention.start_char, EntityMention.end_char)
            .where(EntityMention.entity_id == entity_id)
            .limit(1)
        ).first()
        if mention:
            rid, s, e = mention
            rec = records.get(rid)
            if rec is not None:
                citations.append(
                    qa_core.Citation(
                        record_id=rid,
                        source_type=rec.source_type,
                        title=rec.display_title,
                        occurred_at=rec.occurred_at.isoformat() if rec.occurred_at else None,
                        start_char=int(s),
                        end_char=int(e),
                        quote=rec.body[int(s) : int(e)],
                        retriever="entity",
                    )
                )

    lines = [
        "Entities extracted from the records that match your question:",
        *[f"- {n}" for n in names],
        "These come from entity extraction and linking. A mention is not proof of the "
        "person's or organization's involvement.",
    ]
    confidence, basis = qa_core.confidence_for(len(citations), len(ranked))
    return qa_core.Answer(
        question=question,
        intent=qa_core.INTENT_WHO,
        answer="\n".join(lines),
        citations=citations,
        coverage={"records_in_database": len(records), "query_used": topic},
        method={"mode": "entity_graph", "description": "entity resolution over retrieved records"},
        caveat="Entity resolution can merge or split names incorrectly. Check the record detail.",
        confidence=confidence,
        confidence_basis=basis,
    )

"""Product-rule tests: the promises the UI makes must hold in the data."""

from __future__ import annotations

import re

import pytest
from sqlalchemy import select

from backend.app.models import Insight, Record, RiskAssessment
from backend.app.risk import assess as assess_mod
from backend.app.risk.lexicon import RISK_LABELS
from backend.app.retrieval.qa import NO_MATCH, classify_intent, extract_topic, enforce_citation_integrity
from backend.app.timeline import builder as timeline_mod


# ---------------------------------------------------------------------------
# Cautious wording
# ---------------------------------------------------------------------------


def test_headline_uses_cautious_wording():
    text = (
        "Please transfer $4,250.00 within 10 minutes or your account will be closed "
        "permanently. We are terrified you will lose everything."
    )
    report = assess_mod.combine(1, text)
    assert report.flagged
    assert report.headline.startswith("Potential manipulation risk detected")
    for claim in assess_mod.FORBIDDEN_CLAIMS:
        assert claim not in report.headline.lower()
    assert assess_mod.assert_cautious_wording(report.headline)


def test_headline_when_nothing_found():
    report = assess_mod.combine(1, "The quarterly report is attached for your review.")
    assert not report.flagged
    assert "No risk indicators were detected" in report.headline


def test_benign_record_is_not_flagged():
    benign = (
        "Your scheduled payment of $240.00 will be collected on the 3rd as usual. "
        "Our response target is 30 minutes, per our SLA. "
        "Sorry for the inconvenience this may have caused."
    )
    report = assess_mod.combine(2, benign)
    assert not report.flagged, [i.label for i in report.flagged]


def test_attack_record_flags_all_four():
    attack = (
        "We are extremely worried about what this may mean for you and your family. "
        "Your account will be closed permanently unless you act immediately. "
        "Please transfer $4,250.00 within 10 minutes."
    )
    report = assess_mod.combine(3, attack)
    assert {i.label for i in report.flagged} == set(RISK_LABELS)


def test_indicators_are_reported_separately():
    report = assess_mod.combine(4, "Please respond immediately.")
    labels = {i.label: i.present for i in report.indicators}
    assert labels["urgency"] is True
    assert labels["payment_request"] is False
    assert labels["threat_language"] is False
    assert len(report.indicators) == 4


# ---------------------------------------------------------------------------
# The aggregate figure
# ---------------------------------------------------------------------------


def test_overall_estimate_is_labelled_and_derived():
    report = assess_mod.combine(5, "Transfer $1000 immediately or your account will be closed.")
    estimate = assess_mod.overall_estimate(report)
    assert estimate["is_estimate"] is True
    assert "weighted mean" in estimate["formula"]
    assert "not a probability of fraud" in estimate["caveat"]
    assert len(estimate["contributions"]) == 4
    weights = {c["label"]: c["weight"] for c in estimate["contributions"]}
    assert weights == {
        "payment_request": 0.35,
        "threat_language": 0.30,
        "urgency": 0.20,
        "emotional_pressure": 0.15,
    }
    assert 0.0 <= estimate["value"] <= 1.0


def test_every_indicator_has_an_explanation():
    report = assess_mod.combine(6, "Respond now.")
    for indicator in report.indicators:
        assert indicator.explanation.strip()
        assert "threshold" in indicator.explanation or "signal" in indicator.explanation


def test_flagged_indicator_without_span_is_declared_model_only():
    # A model score with no regex hit must still say so rather than invent a quote.
    report = assess_mod.combine(
        7,
        "Nothing suspicious in this wording at all.",
        model_scores={"fake-model": __import__("numpy").array([[0.1, 0.9, 0.1, 0.1]])},
        model_thresholds={"fake-model": {"urgency": 0.5, "threat_language": 0.5,
                                         "emotional_pressure": 0.5, "payment_request": 0.5}},
        model_names={"fake-model": "Fake model"},
    )
    threat = next(i for i in report.indicators if i.label == "threat_language")
    assert threat.present
    assert threat.evidence
    assert threat.evidence[0]["start_char"] is None
    assert "without a specific span" in threat.evidence[0]["note"]
    assert any("no cited span" in n for n in report.model_notes)


# ---------------------------------------------------------------------------
# Timeline lanes
# ---------------------------------------------------------------------------


def test_timeline_separates_recorded_and_inferred(demo_session):
    session = demo_session
    from backend.app.models import Event

    records = list(session.execute(select(Record)).scalars())
    events: dict[int, list[Event]] = {}
    for e in session.execute(select(Event)).scalars():
        events.setdefault(e.record_id, []).append(e)
    risk: dict[int, list[str]] = {}
    for a in session.execute(select(RiskAssessment)).scalars():
        if a.present:
            risk.setdefault(a.record_id, []).append(a.label)

    timeline = timeline_mod.build_timeline(records, events, risk)
    assert timeline
    lanes = {item.lane for item in timeline}
    assert "recorded" in lanes
    for item in timeline:
        if item.lane == "recorded":
            assert item.explanation
            assert "source system" in item.explanation
        if item.lane == "inferred":
            assert item.explanation
            # An inferred item must be a range, not a point.
            assert item.start_at is not None
            assert item.raw_expression


def test_record_without_timestamp_is_not_presented_as_a_fact(session):
    from backend.app.ingest.demo import DemoRecord

    rec = Record(
        source_type="note",
        title="No date here",
        body="Some text",
        occurred_at=None,
        origin="test",
        external_id="no-date",
    )
    session.add(rec)
    session.flush()
    timeline = timeline_mod.build_timeline([rec], {}, {})
    assert timeline[0].lane != timeline_mod.RECORDED
    assert "no source timestamp" in timeline[0].explanation


# ---------------------------------------------------------------------------
# Absence wording
# ---------------------------------------------------------------------------


def test_required_no_match_wording():
    assert NO_MATCH == "No matching discussion was found in the records analyzed."


@pytest.mark.parametrize(
    "question,intent",
    [
        ("What records mention Project X?", "which_records_mention"),
        ("What happened before this payment request?", "what_happened_before"),
        ("Which records are risky?", "risk_analysis"),
        ("Who was involved in the account review?", "who_involved"),
    ],
)
def test_intent_classification(question, intent):
    assert classify_intent(question) == intent


def test_topic_extraction_prefers_quoted_text():
    assert extract_topic('What records mention "Project Atlas"?') == "Project Atlas"
    assert extract_topic("What records mention Project X?") == "Project X"


def test_no_match_answer_never_claims_absence(demo_session):
    from backend.app.retrieval import answering

    answer = answering.answer_question(
        demo_session, "What records mention the Zephyrine acquisition?"
    )
    assert answer.no_match
    assert NO_MATCH in answer.answer
    lowered = answer.answer.lower()
    for forbidden in (
        "did not happen",
        "never happened",
        "there was no discussion",
        "does not exist",
        "no such discussion existed",
    ):
        assert forbidden not in lowered
    assert "not evidence" in lowered
    assert answer.caveat
    assert "not evidence of absence" in answer.caveat.lower()


def test_citation_integrity_catches_a_bad_quote(demo_session):
    from backend.app.retrieval.qa import Answer, Citation

    record = demo_session.execute(select(Record)).scalars().first()
    bad = Answer(
        question="q",
        intent="generic_search",
        answer="a",
        citations=[
            Citation(
                record_id=record.id,
                source_type=record.source_type,
                title=record.title,
                occurred_at=None,
                start_char=0,
                end_char=20,
                quote="something that was never written",
            )
        ],
    )
    problems = enforce_citation_integrity(bad, {record.id: record.body})
    assert problems
    assert "does not match" in problems[0]


def test_live_answers_have_valid_citations(demo_session):
    from backend.app.retrieval import answering

    bodies = {r.id: r.body for r in demo_session.execute(select(Record)).scalars()}
    for question in (
        "What records mention Northwind Bank?",
        "What happened before this payment request?",
        "Who was involved in the account review?",
    ):
        answer = answering.answer_question(demo_session, question)
        assert enforce_citation_integrity(answer, bodies) == [], question


# ---------------------------------------------------------------------------
# Insights are inspectable
# ---------------------------------------------------------------------------


def test_every_insight_cites_its_facts(demo_session):
    insights = list(demo_session.execute(select(Insight)).scalars())
    assert insights
    for insight in insights:
        assert insight.title.strip()
        assert insight.statement.strip()
        assert insight.confidence_basis.strip()
        assert insight.record_ids
        for fact in insight.facts or []:
            assert fact.get("record_id") in insight.record_ids
            if fact.get("start_char") is not None:
                assert isinstance(fact.get("end_char"), int)


def test_unresolved_insight_uses_required_wording(demo_session):
    unresolved = [
        i
        for i in demo_session.execute(select(Insight)).scalars()
        if i.kind == "unresolved_discussion_reference"
    ]
    assert unresolved, "the demo corpus must contain an unmatched back-reference"
    for insight in unresolved:
        assert NO_MATCH in insight.statement
        assert re.search(r"not evidence that no such discussion", insight.statement)


def test_risk_insight_is_cautious(demo_session):
    for insight in demo_session.execute(select(Insight)).scalars():
        assert insight.kind != "risk_observation" or insight.title.startswith(
            "Potential manipulation risk detected"
        )
        assert assess_mod.assert_cautious_wording(insight.statement)
        assert assess_mod.assert_cautious_wording(insight.title)

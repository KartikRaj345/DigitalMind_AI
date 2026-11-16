"""Question answering over the records, with citations.

Design constraint from the brief: **a missing record is never proof that something did not
happen.** So this module has no "not found" answer that implies absence. It reports what
was searched, over what window, and what was found - and the empty result is phrased as
"No matching discussion was found in the records analyzed."

Every answer carries:

* ``citations``    - record id, source type, exact character span, verbatim quote
* ``method``       - which retrievers ran and how they fused
* ``coverage``     - what was searched, so the user can judge whether the absence means
  anything at all
* ``caveat``       - required whenever the answer is thin

The optional LLM layer is off by default. When it is enabled it may only *rephrase* an
answer that already has citations; it can never introduce a fact that is not in the
evidence. :func:`enforce_citation_integrity` checks that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ..logging_config import get_logger

logger = get_logger(__name__)

NO_MATCH = "No matching discussion was found in the records analyzed."

# Intents, in priority order. Deliberately narrow: a general open-domain QA system would
# need to invent answers, which is exactly what this project refuses to do.
INTENT_WHICH_RECORDS = "which_records_mention"
INTENT_WHAT_BEFORE = "what_happened_before"
INTENT_WHAT_AFTER = "what_happened_after"
INTENT_WHO = "who_involved"
INTENT_RISK = "risk_analysis"
INTENT_LIST = "list_mentions"
INTENT_GENERIC = "generic_search"

INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        INTENT_WHAT_BEFORE,
        re.compile(
            r"\b(what|which)\s+(happened|occurred|was said|was agreed|else\s+\w+)\b"
            r".*\bbefore\b|"
            r"\bprior to\b|\bpreceding\b|\bwhat came before\b|\bwhat led (up )?to\b",
            re.IGNORECASE,
        ),
    ),
    (
        INTENT_WHAT_AFTER,
        re.compile(r"\bwhat\b.*\b(after|since|following|subsequently)\b", re.IGNORECASE),
    ),
    (
        INTENT_WHAT_BEFORE,
        re.compile(r"\b(before|prior to)\s+(this|the)\s+(payment|request|message|email)\b",
                   re.IGNORECASE),
    ),
    # Risk is checked before "which records" because "which records are risky" is a risk
    # question that happens to mention records; answering it with a plain list would hide
    # the indicator breakdown the user actually wants.
    (
        INTENT_RISK,
        re.compile(
            r"\b(risk|risky|suspicious|manipulat\w+|scam\w*|fraud\w*|phishing|threat\w*|"
            r"pressure|pressure-heavy)\b",
            re.IGNORECASE,
        ),
    ),
    (
        INTENT_WHO,
        re.compile(r"\bwho\b.*\b(involved|mentioned|is|sent|owns?|contact)\b", re.IGNORECASE),
    ),
    (
        INTENT_WHICH_RECORDS,
        re.compile(
            r"\b(what|which)\s+(records?|emails?|messages?|documents?|notes?|files?)\b"
            r"|\bwhat\s+(is|are)\s+in\s+my\b|\bshow me\s+(all\s+)?(the\s+)?"
            r"(records?|emails?|messages?|notes?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        INTENT_LIST,
        re.compile(r"\b(mentions?|references?|talks? about|about)\b", re.IGNORECASE),
    ),
]

_QUOTED = re.compile(r"[“\"']([^”\"']{3,120})[”\"']")
_CAPITALIZED_PHRASE = re.compile(r"\b(?:Project|Account|Invoice|Order|Ticket)\s+[A-Z][\w-]*")


@dataclass
class Citation:
    record_id: int
    source_type: str
    title: str
    occurred_at: str | None
    start_char: int
    end_char: int
    quote: str
    retriever: str = ""
    score: float | None = None

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "source_type": self.source_type,
            "title": self.title,
            "occurred_at": self.occurred_at,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote": self.quote,
            "retriever": self.retriever,
            "score": None if self.score is None else round(self.score, 6),
        }


@dataclass
class Answer:
    question: str
    intent: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    method: dict[str, Any] = field(default_factory=dict)
    caveat: str = ""
    no_match: bool = False
    confidence: float = 0.0
    confidence_basis: str = ""

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "intent": self.intent,
            "answer": self.answer,
            "citations": [c.as_dict() for c in self.citations],
            "facts": self.facts,
            "interpretation": self.interpretation,
            "coverage": self.coverage,
            "method": self.method,
            "caveat": self.caveat,
            "no_match": self.no_match,
            "confidence": round(self.confidence, 3),
            "confidence_basis": self.confidence_basis,
        }


def classify_intent(question: str) -> str:
    q = question.strip()
    for intent, pattern in INTENT_PATTERNS:
        if pattern.search(q):
            return intent
    return INTENT_GENERIC


def extract_topic(question: str) -> str:
    """Best-effort retrieval query from a question.

    Strips interrogative scaffolding so "What records mention Project X?" becomes
    "Project X". Quoted phrases win: they are what the user actually cares about.
    """
    quoted = _QUOTED.findall(question)
    if quoted:
        return quoted[0].strip()
    phrase = _CAPITALIZED_PHRASE.search(question)
    if phrase:
        return phrase.group(0)

    cleaned = re.sub(
        r"^(what|which|who|when|where|why|how)\s+"
        r"(records?|emails?|messages?|documents?|notes?|files?|people|persons?|"
        r"discussions?|things?)?\s*"
        r"(mention|mentions|mentioned|involve|involves|involved|about|regarding|"
        r"say|says|said|talk about|talks about|contain|contains|happened)?\s*",
        "",
        question.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\?$", "", cleaned.strip()
    ).strip(" .,:;")
    stop = {
        "me", "my", "it", "this", "that", "there", "any", "some", "all",
        "the", "a", "an", "and", "or", "please", "show", "find", "list",
    }
    words = [w for w in cleaned.split() if w.lower() not in stop]
    result = " ".join(words).strip()
    return result or question.strip()


def no_match_answer(
    question: str,
    intent: str,
    topic: str,
    searched: dict[str, Any],
    coverage: dict[str, Any],
    method: dict[str, Any],
) -> Answer:
    """The cautious empty result.

    Explicitly *not* "no such discussion exists". It says what was searched and over what
    data, so the user can see that the answer is bounded by what DigitalMind has.
    """
    phrase = f"“{topic}”" if topic else "that topic"
    answer = (
        f"{NO_MATCH} I searched {searched.get('records_considered', 0)} records using "
        f"{method.get('description', 'keyword and semantic search')} for {phrase}. "
        "This means only that the records I have do not contain a matching discussion - it is "
        "not evidence that no such discussion took place. It may exist in a record that was "
        "never imported, or in a channel that is not connected yet."
    )
    return Answer(
        question=question,
        intent=intent,
        answer=answer,
        citations=[],
        coverage=coverage,
        method=method,
        caveat=(
            "Absence of a matching record is not evidence of absence. Import the relevant "
            "channel, or check the original conversation, before drawing a conclusion."
        ),
        no_match=True,
        confidence=0.0,
        confidence_basis="No supporting evidence was retrieved.",
    )


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def _quote(record_body: str, start: int, end: int, limit: int = 320) -> str:
    if start is None or end is None:
        return ""
    start = max(0, start)
    end = min(len(record_body), end)
    text = record_body[start:end].strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def enforce_citation_integrity(answer: Answer, bodies: dict[int, str]) -> list[str]:
    """Verify every citation points at the words it claims to.

    Returns a list of problems (empty when clean). Called after any LLM rephrasing and
    asserted in tests. This is the guard that stops an optional model from inventing a
    quote.
    """
    problems: list[str] = []
    for cite in answer.citations:
        body = bodies.get(cite.record_id)
        if body is None:
            problems.append(f"citation references unknown record {cite.record_id}")
            continue
        if cite.start_char < 0 or cite.end_char > len(body):
            problems.append(f"citation for record {cite.record_id} has out-of-range span")
            continue
        actual = body[cite.start_char : cite.end_char]
        norm = re.sub(r"\s+", " ", actual).strip().rstrip("…")
        quoted = re.sub(r"\s+", " ", cite.quote).strip().rstrip("…")
        if norm and quoted and norm not in quoted and quoted not in norm:
            problems.append(
                f"citation quote for record {cite.record_id} does not match the source span"
            )
    return problems


CAVEAT_THIN = (
    "Only a small amount of supporting text was found. Treat this as a starting point and "
    "open the cited records to check the context yourself."
)

CONFIDENCE_BANDS = [
    (0.75, "strong", "Several independent records support this answer."),
    (0.45, "moderate", "One or two records support this answer."),
    (0.0, "weak", "Little supporting text was found."),
]


def confidence_for(n_citations: int, n_records: int) -> tuple[float, str]:
    """Confidence reflects evidence volume, not model certainty.

    This is a retrieval-side measure: how much of the corpus agrees. It is labelled as
    such in the UI so it is not confused with a probability of truth.
    """
    if n_citations == 0:
        return 0.0, "No supporting evidence was retrieved."
    base = min(0.9, 0.35 + 0.18 * n_records + 0.04 * n_citations)
    for threshold, band, basis in CONFIDENCE_BANDS:
        if base >= threshold:
            return round(base, 3), f"{band.capitalize()} evidence: {basis}"
    return round(base, 3), "Weak evidence."

"""Extraction pipeline.

Combines the rules baseline with spaCy NER and date/amount normalization into a single
:class:`ExtractionResult`. The result deliberately separates:

* ``spans``      - located text with a label (a *fact about the text*)
* ``events``     - normalized temporal statements with their certainty
* ``backrefs``   - phrases asserting a prior discussion exists

It contains no interpretation. Interpretation happens downstream (risk, insights) and is
always attributed to a named model.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..logging_config import get_logger
from . import dates as dates_mod
from . import ner, rules

logger = get_logger(__name__)


@dataclass
class ExtractedEvent:
    title: str
    kind: str
    certainty: str  # recorded | explicit | inferred
    start_at: dt.datetime | None
    end_at: dt.datetime | None
    precision: str
    raw_expression: str
    start_char: int | None
    end_char: int | None
    confidence: float
    extractor: str
    note: str = ""
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "kind": self.kind,
            "certainty": self.certainty,
            "start_at": self.start_at.isoformat() if self.start_at else None,
            "end_at": self.end_at.isoformat() if self.end_at else None,
            "precision": self.precision,
            "raw_expression": self.raw_expression,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": round(self.confidence, 4),
            "extractor": self.extractor,
            "note": self.note,
            "description": self.description,
        }


@dataclass
class ExtractionResult:
    spans: list[rules.Span] = field(default_factory=list)
    events: list[ExtractedEvent] = field(default_factory=list)
    backrefs: list[tuple[int, int, str]] = field(default_factory=list)
    extractors_used: list[str] = field(default_factory=list)

    @property
    def persons(self) -> list[rules.Span]:
        return [s for s in self.spans if s.label == rules.PERSON]

    @property
    def orgs(self) -> list[rules.Span]:
        return [s for s in self.spans if s.label == rules.ORG]


def _merge_spans(rule_spans: list[rules.Span], ner_spans: list[ner.NerSpan]) -> list[rules.Span]:
    """Union the two extractors, keeping provenance per span.

    When both extractors agree on the same span we bump confidence and record that they
    agreed; that agreement is later used as a confidence signal.
    """
    merged: list[rules.Span] = list(rule_spans)
    for ns in ner_spans:
        overlap = next(
            (
                s
                for s in merged
                if not (ns.end <= s.start or ns.start >= s.end) and s.label == ns.label
            ),
            None,
        )
        if overlap is not None:
            object.__setattr__(overlap, "confidence", min(0.99, overlap.confidence + 0.1))
            continue
        merged.append(
            rules.Span(
                text=ns.text,
                label=ns.label,
                start=ns.start,
                end=ns.end,
                confidence=ns.confidence,
                extractor="spacy",
            )
        )
    return rules.resolve_overlaps(merged)


def _money_event(title_text: str, span: rules.Span) -> ExtractedEvent | None:
    """A money mention is a fact, not an event; only surface amounts in `spans`."""
    return None


def _events_from_dates(
    nd: list[dates_mod.NormalizedDate],
    body: str,
    record_time: dt.datetime | None,
    title: str,
) -> list[ExtractedEvent]:
    events: list[ExtractedEvent] = []
    for d in nd:
        kind = "reference"
        if d.kind == "absolute":
            certainty = "explicit"
        elif d.kind == "relative":
            # Deadlines phrased relative to now are inferred, not facts.
            certainty = "inferred"
        else:
            certainty = "inferred"

        if d.raw.lower().startswith(("in ", "within ")) or "ago" in d.raw.lower():
            kind = "deadline" if d.raw.lower().startswith("in ") else "reference"
        elif d.kind == "backreference":
            kind = "prior_discussion"
        else:
            kind = "temporal_reference"

        description = _sentence_around(body, d.start_char)
        events.append(
            ExtractedEvent(
                title=title or d.raw,
                kind=kind,
                certainty=certainty,
                start_at=d.start,
                end_at=d.end,
                precision=d.precision,
                raw_expression=d.raw,
                start_char=d.start_char,
                end_char=d.end_char,
                confidence=d.confidence,
                extractor="rules.dates",
                note=d.note,
                description=description,
            )
        )
    return events


def _sentence_around(text: str, pos: int, window: int = 110) -> str:
    """Return a short quotation around a span, for display alongside an event."""
    left = max(0, pos - window // 2)
    right = min(len(text), pos + window)
    snippet = text[left:right].strip()
    if left > 0:
        snippet = "…" + snippet
    if right < len(text):
        snippet = snippet + "…"
    return snippet


def extract(
    text: str,
    record_time: dt.datetime | None = None,
    title: str = "",
    use_spacy: bool = True,
) -> ExtractionResult:
    """Run the full extraction pipeline over one record."""
    anchor = record_time or dt.datetime.now(dt.timezone.utc)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=dt.timezone.utc)

    rule_spans = rules.extract_all(text)
    extractors = {"rules"}

    ner_spans: list[ner.NerSpan] = []
    if use_spacy:
        ner_spans = ner.extract(text)
        if ner_spans:
            extractors.add("spacy")

    spans = _merge_spans(rule_spans, ner_spans)

    nd = dates_mod.extract_dates(text, anchor)
    if nd:
        extractors.add("rules.dates")
    events = _events_from_dates(nd, text, record_time, title)

    backrefs = dates_mod.find_backreferences(text)

    logger.debug(
        "extracted spans=%d events=%d backrefs=%d extractors=%s",
        len(spans), len(events), len(backrefs), sorted(extractors),
    )
    return ExtractionResult(
        spans=spans, events=events, backrefs=backrefs, extractors_used=sorted(extractors)
    )

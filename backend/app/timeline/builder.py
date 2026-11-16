"""Timeline construction.

The timeline is where a naive implementation quietly lies: it sorts "last week" and
"2026-03-04" into one list and shows both as facts. We keep them structurally separate.

Three buckets, always labelled in the UI:

``recorded``
    The source system gave a timestamp (email ``Date:``, ICS ``DTSTART``, task due date).
    High trust: this is what the system recorded.

``explicit``
    The text names an absolute date. Trust depends on the writer.

``inferred``
    Only relative language ("last week", "yesterday"). Rendered as a **range**, placed in a
    separate lane, and never presented as a point in time.

Ordering guarantees:
* Records are ordered by their recorded timestamp.
* Inferred events are placed by the *end* of their range (a "last week" event is placed at
  the end of last week), so a range never hides a fact that came later.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..models import Event, Record

RECORDED = "recorded"
EXPLICIT = "explicit"
INFERRED = "inferred"


@dataclass
class TimelineItem:
    record_id: int
    title: str
    source_type: str
    lane: str  # recorded | explicit | inferred
    start_at: dt.datetime | None
    end_at: dt.datetime | None
    sort_key: dt.datetime
    certainty: str
    precision: str
    raw_expression: str
    confidence: float
    explanation: str
    start_char: int | None = None
    end_char: int | None = None
    has_risk: bool = False
    risk_labels: list[str] = field(default_factory=list)
    kind: str = "record"

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "title": self.title,
            "source_type": self.source_type,
            "lane": self.lane,
            "kind": self.kind,
            "start_at": self.start_at.isoformat() if self.start_at else None,
            "end_at": self.end_at.isoformat() if self.end_at else None,
            "sort_key": self.sort_key.isoformat(),
            "certainty": self.certainty,
            "precision": self.precision,
            "raw_expression": self.raw_expression,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "has_risk": self.has_risk,
            "risk_labels": self.risk_labels,
        }


LANE_EXPLANATIONS = {
    RECORDED: (
        "Timestamped by the source system (email Date, calendar event, task due date). "
        "This is a recorded fact about the record, though not proof the content is true."
    ),
    EXPLICIT: (
        "The record's text names an absolute date. Recorded by the writer, so it reflects "
        "what they said, not independently verified time."
    ),
    INFERRED: (
        "Only relative language appears (for example 'last week'). DigitalMind resolved this "
        "against the record's own timestamp and shows a range, not a specific moment."
    ),
}


def _sort_key(start: dt.datetime | None, end: dt.datetime | None, fallback: dt.datetime) -> dt.datetime:
    if start and end:
        return end
    return start or fallback


def build_timeline(
    records: Iterable[Record],
    events_by_record: dict[int, list[Event]],
    risk_by_record: dict[int, list[str]],
    limit: int | None = None,
) -> list[TimelineItem]:
    """Merge records and events into ordered, lane-separated timeline items."""
    items: list[TimelineItem] = []
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

    for rec in records:
        when = rec.occurred_at or rec.imported_at or epoch
        if rec.occurred_at is not None:
            lane = RECORDED
            explanation = LANE_EXPLANATIONS[RECORDED]
            precision = "minute"
            raw = when.strftime("%Y-%m-%d %H:%M")
        else:
            # No source timestamp at all. We do NOT fall back to import time and call it
            # a fact; the item goes to the explicit lane with a caveat.
            lane = EXPLICIT
            explanation = (
                "This record carries no source timestamp. Only its date content is shown."
            )
            precision = "unknown"
            raw = ""
        labels = risk_by_record.get(rec.id, [])
        items.append(
            TimelineItem(
                record_id=rec.id,
                title=rec.display_title,
                source_type=rec.source_type,
                lane=lane,
                start_at=when,
                end_at=None,
                sort_key=when,
                certainty=lane,
                precision=precision,
                raw_expression=raw,
                confidence=0.95 if rec.occurred_at is not None else 0.4,
                explanation=explanation,
                has_risk=bool(labels),
                risk_labels=labels,
                kind="record",
            )
        )

        for ev in events_by_record.get(rec.id, []):
            lane = EXPLICIT if ev.certainty in {RECORDED, EXPLICIT} else INFERRED
            if ev.certainty == RECORDED:
                explanation = LANE_EXPLANATIONS[RECORDED]
            elif lane == EXPLICIT:
                explanation = LANE_EXPLANATIONS[EXPLICIT]
            else:
                explanation = ev.note or LANE_EXPLANATIONS[INFERRED]
            items.append(
                TimelineItem(
                    record_id=rec.id,
                    title=ev.title or ev.raw_expression or rec.display_title,
                    source_type=rec.source_type,
                    lane=lane,
                    start_at=ev.start_at,
                    end_at=ev.end_at,
                    sort_key=_sort_key(ev.start_at, ev.end_at, when),
                    certainty=ev.certainty,
                    precision=ev.precision,
                    raw_expression=ev.raw_expression,
                    confidence=ev.confidence,
                    explanation=explanation,
                    start_char=ev.start_char,
                    end_char=ev.end_char,
                    has_risk=bool(labels),
                    risk_labels=labels,
                    kind=ev.kind,
                )
            )

    items.sort(key=lambda i: (i.sort_key, i.record_id, i.lane))
    if limit:
        items = items[:limit]
    return items


def records_before(
    timeline: list[TimelineItem], record_id: int, window_days: int = 30, limit: int = 12
) -> list[TimelineItem]:
    """Items strictly before a record, in the recorded lane first.

    This is the evidence list behind "what happened before this payment request?".
    """
    anchor_index = next(
        (i for i, item in enumerate(timeline) if item.record_id == record_id and item.kind == "record"),
        None,
    )
    if anchor_index is None:
        return []
    anchor = timeline[anchor_index].sort_key
    cutoff = anchor - dt.timedelta(days=window_days)

    prior = [
        item
        for item in timeline[:anchor_index]
        if item.sort_key <= anchor and item.sort_key >= cutoff
    ]
    # Records first (they are the facts), then explicit, then inferred ranges.
    order = {RECORDED: 0, EXPLICIT: 1, INFERRED: 2}
    prior.sort(key=lambda i: (order[i.lane], -i.sort_key.timestamp()))
    return prior[:limit]


def lane_summary(timeline: list[TimelineItem]) -> dict[str, Any]:
    counts: dict[str, int] = {RECORDED: 0, EXPLICIT: 0, INFERRED: 0}
    for item in timeline:
        counts[item.lane] = counts.get(item.lane, 0) + 1
    return {
        "counts": counts,
        "explanations": LANE_EXPLANATIONS,
        "note": (
            "Inferred items are shown as ranges because the text does not name a specific "
            "time. They are estimates, not facts."
        ),
    }

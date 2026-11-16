"""Date and time expression normalization.

The key distinction this module enforces:

* **Absolute** expressions ("on 4 March 2026", "2026-03-04 09:30") are facts about the
  text. We can pin them to a day or minute.
* **Relative** expressions ("last week", "yesterday", "in 10 minutes") are only resolvable
  relative to an anchor. We always emit a *range* plus ``precision`` and mark the result
  as inferred. We never silently pretend "last week" is a specific day.

Every returned :class:`NormalizedDate` carries the literal matched text and its character
span so the caller can cite it.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from dateutil import parser as dateparser

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
DAY_NAMES = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"

# Absolute dates: "4 March 2026", "March 4, 2026", "2026-03-04", "04/03/2026"
PAT_ISO = re.compile(r"\b(?P<y>20\d{2})-(?P<m>0?[1-9]|1[0-2])-(?P<d>0?[1-9]|[12]\d|3[01])\b")
PAT_DMY = re.compile(
    rf"\b(?P<d>0?[1-9]|[12]\d|3[01])\s+(?:of\s+)?(?P<mon>{MONTHS})\.?\s+(?P<y>20\d{{2}})\b",
    re.IGNORECASE,
)
PAT_MDY = re.compile(
    rf"\b(?P<mon>{MONTHS})\.?\s+(?P<d>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(?P<y>20\d{{2}})\b",
    re.IGNORECASE,
)
PAT_MONTH_YEAR = re.compile(
    rf"\b(?P<mon>{MONTHS})\.?\s+(?P<y>20\d{{2}})\b", re.IGNORECASE
)
PAT_SLASH = re.compile(r"\b(?P<a>0?[1-9]|[12]\d|3[01])/(?P<b>0?[1-9]|1[0-2])/(?P<y>20\d{2})\b")

# Times: "at 14:30", "09:30", "2:45 pm", "at 9 am"
PAT_TIME = re.compile(
    r"\b(?:at\s+)?(?P<h>[01]?\d|2[0-3]):(?P<min>[0-5]\d)\s*(?P<ampm>am|pm)?\b",
    re.IGNORECASE,
)
PAT_TIME_WORD = re.compile(
    r"\b(?:at\s+)?(?P<h>[01]?\d|2[0-3])\s*(?P<ampm>am|pm)\b", re.IGNORECASE
)

# Relative expressions. Order matters for the alternation.
RELATIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("today", re.compile(r"\b(?:today|this morning|this afternoon|tonight)\b", re.I)),
    ("yesterday", re.compile(r"\byesterday\b", re.I)),
    ("tomorrow", re.compile(r"\btomorrow\b", re.I)),
    (
        "last_week",
        re.compile(r"\b(?:last|past)\s+week\b|\b(?:a\s+)?week\s+ago\b", re.I),
    ),
    ("next_week", re.compile(r"\bnext\s+week\b", re.I)),
    ("last_month", re.compile(r"\b(?:last|past)\s+month\b|\b(?:a\s+)?month\s+ago\b", re.I)),
    ("this_month", re.compile(r"\b(?:this|current)\s+month\b", re.I)),
    ("last_year", re.compile(r"\b(?:last|past)\s+year\b|\b(?:a\s+)?year\s+ago\b", re.I)),
    ("in_minutes", re.compile(r"\bin\s+(?P<n>\d{1,3})\s+minutes?\b", re.I)),
    ("in_hours", re.compile(r"\bin\s+(?P<n>\d{1,3})\s+hours?\b", re.I)),
    ("in_days", re.compile(r"\bin\s+(?P<n>\d{1,3})\s+days?\b", re.I)),
    ("minutes_ago", re.compile(r"\b(?P<n>\d{1,3})\s+minutes?\s+ago\b", re.I)),
    ("hours_ago", re.compile(r"\b(?P<n>\d{1,3})\s+hours?\s+ago\b", re.I)),
    ("days_ago", re.compile(r"\b(?P<n>\d{1,3})\s+days?\s+ago\b", re.I)),
    (
        "weekday",
        re.compile(rf"\b(?P<day>{DAY_NAMES})\b", re.IGNORECASE),
    ),
    (
        "relative_meeting",
        re.compile(
            r"\b(?:as\s+we\s+discussed|as\s+we\s+agreed|as\s+mentioned|as\s+we\s+spoke)"
            r"\s+(?P<when>last\s+week|earlier\s+last\s+week|previously|before|"
            r"on\s+\w+\s+\w+|in\s+our\s+last\s+\w+)",
            re.IGNORECASE,
        ),
    ),
]

# A reference back to a prior conversation. These are the ones that make Scenario B
# interesting: the record asserts something happened elsewhere.
BACKREFERENCE_PATTERNS = [
    re.compile(r"\bas\s+we\s+discussed\b", re.I),
    re.compile(r"\bas\s+we\s+agreed\b", re.I),
    re.compile(r"\bas\s+previously\s+mentioned\b", re.I),
    re.compile(r"\bas\s+I\s+(?:mentioned|said|noted)\b", re.I),
    re.compile(r"\bper\s+(?:our|the)\s+(?:last\s+)?(?:call|chat|meeting|conversation)\b", re.I),
    re.compile(r"\bfollowing\s+up\s+on\s+(?:our|the)\b", re.I),
    re.compile(r"\bin\s+line\s+with\s+(?:our|the)\b", re.I),
    re.compile(r"\bwe\s+had\s+agreed\b", re.I),
    re.compile(r"\bcontinuing\s+(?:our|the)\s+(?:discussion|conversation)\b", re.I),
    re.compile(r"\breferring\s+to\s+our\s+(?:earlier|previous|last)\b", re.I),
]

_SPAN_RE = re.compile(r"\S")


@dataclass
class NormalizedDate:
    """A date expression found in text."""

    raw: str
    start_char: int
    end_char: int
    start: dt.datetime | None
    end: dt.datetime | None
    kind: str  # 'absolute' | 'relative' | 'backreference'
    precision: str  # minute | day | week | month | year | unknown
    confidence: float
    note: str = ""
    field: dict[str, str] = field(default_factory=dict)

    @property
    def is_inferred(self) -> bool:
        return self.kind in {"relative", "backreference"}


def _span(text: str, match: re.Match[str]) -> tuple[int, int]:
    return match.start(), match.end()


def _parse_iso(text: str) -> dt.datetime | None:
    try:
        return dateparser.parse(text, fuzzy=False, default=None)
    except (ValueError, OverflowError, TypeError):
        return None


def _combine_date_time(date_part: dt.datetime | None, time_match: re.Match[str] | None,
                       text_end: int) -> dt.datetime | None:
    if date_part is None or time_match is None:
        return date_part
    hour = int(time_match.group("h"))
    ampm = (time_match.group("ampm") or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    try:
        return date_part.replace(
            hour=hour, minute=int(time_match.group("min")), second=0, microsecond=0
        )
    except ValueError:
        return date_part


def _find_time_near(text: str, position: int, window: int = 24) -> re.Match[str] | None:
    """Look for a clock time shortly after a date mention ("on 4 March 2026 at 14:30")."""
    tail = text[position : position + window]
    for pattern in (PAT_TIME, PAT_TIME_WORD):
        m = pattern.search(tail)
        if m:
            return m
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_dates(text: str, anchor: dt.datetime) -> list[NormalizedDate]:
    """Find absolute and relative date expressions.

    ``anchor`` is the record's own timestamp; relative expressions are resolved against
    it and always reported as inferred with a range.
    """
    results: list[NormalizedDate] = []
    claimed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in claimed)

    def add(nd: NormalizedDate) -> None:
        if not overlaps(nd.start_char, nd.end_char):
            claimed.append((nd.start_char, nd.end_char))
            results.append(nd)

    # --- absolute ---------------------------------------------------------
    for match in PAT_ISO.finditer(text):
        parsed = _parse_iso(match.group(0))
        if parsed is None:
            continue
        tm = _find_time_near(text, match.end())
        resolved = _combine_date_time(parsed, tm, match.end()) or parsed
        precision = "minute" if tm else "day"
        add(
            NormalizedDate(
                raw=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                start=resolved,
                end=resolved,
                kind="absolute",
                precision=precision,
                confidence=0.95,
                note="Explicit date written in the record.",
            )
        )

    for match in PAT_DMY.finditer(text):
        parsed = _parse_iso(match.group(0))
        if parsed is None:
            continue
        tm = _find_time_near(text, match.end())
        resolved = _combine_date_time(parsed, tm, match.end()) or parsed
        add(
            NormalizedDate(
                raw=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                start=resolved,
                end=resolved,
                kind="absolute",
                precision="minute" if tm else "day",
                confidence=0.93,
                note="Explicit date written in the record.",
            )
        )

    for match in PAT_MDY.finditer(text):
        parsed = _parse_iso(match.group(0))
        if parsed is None:
            continue
        tm = _find_time_near(text, match.end())
        resolved = _combine_date_time(parsed, tm, match.end()) or parsed
        add(
            NormalizedDate(
                raw=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                start=resolved,
                end=resolved,
                kind="absolute",
                precision="minute" if tm else "day",
                confidence=0.93,
                note="Explicit date written in the record.",
            )
        )

    for match in PAT_SLASH.finditer(text):
        a, b, y = int(match.group("a")), int(match.group("b")), int(match.group("y"))
        # Assume day/month order; disambiguate the impossible cases to month/day.
        day, month = (a, b) if a <= 31 and b <= 12 else (b, a)
        try:
            parsed = dt.datetime(y, month, day)
        except ValueError:
            continue
        add(
            NormalizedDate(
                raw=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                start=parsed,
                end=parsed,
                kind="absolute",
                precision="day",
                confidence=0.85,
                note="Numeric date; day/month order assumed.",
            )
        )

    for match in PAT_MONTH_YEAR.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        try:
            parsed = _parse_iso(match.group(0))
        except Exception:
            parsed = None
        if parsed is None:
            continue
        start = parsed.replace(day=1)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
        add(
            NormalizedDate(
                raw=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                start=start,
                end=end,
                kind="absolute",
                precision="month",
                confidence=0.8,
                note="Month and year only; the exact day is not stated.",
            )
        )

    # --- relative ---------------------------------------------------------
    for kind, pattern in RELATIVE_PATTERNS:
        for match in pattern.finditer(text):
            if overlaps(match.start(), match.end()):
                continue
            raw = match.group(0)
            resolved = _resolve_relative(kind, match, anchor)
            if resolved is None:
                continue
            start, end, precision, note, conf = resolved
            add(
                NormalizedDate(
                    raw=raw,
                    start_char=match.start(),
                    end_char=match.end(),
                    start=start,
                    end=end,
                    kind="backreference" if kind == "relative_meeting" else "relative",
                    precision=precision,
                    confidence=conf,
                    note=note,
                )
            )

    results.sort(key=lambda d: d.start_char)
    return results


def _resolve_relative(
    kind: str, match: re.Match[str], anchor: dt.datetime
) -> tuple[dt.datetime | None, dt.datetime | None, str, str, float] | None:
    """Map a relative expression onto an interval, given the record's own timestamp."""
    midnight = anchor.replace(hour=0, minute=0, second=0, microsecond=0)

    if kind == "today":
        return midnight, midnight + dt.timedelta(days=1), "day", "Resolved to the record's own date.", 0.9
    if kind == "yesterday":
        d = midnight - dt.timedelta(days=1)
        return d, d + dt.timedelta(days=1), "day", "Resolved to the day before the record.", 0.9
    if kind == "tomorrow":
        d = midnight + dt.timedelta(days=1)
        return d, d + dt.timedelta(days=1), "day", "Resolved to the day after the record.", 0.9
    if kind == "last_week":
        # The seven days before the record's week. We give the range, never a day.
        week_start = midnight - dt.timedelta(days=midnight.weekday() + 7)
        return (
            week_start,
            week_start + dt.timedelta(days=7),
            "week",
            "Interpreted as the seven days before the record's week.",
            0.7,
        )
    if kind == "next_week":
        week_start = midnight + dt.timedelta(days=7 - midnight.weekday())
        return week_start, week_start + dt.timedelta(days=7), "week", "Interpreted as the following week.", 0.7
    if kind == "last_month":
        start = (midnight.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
        end = midnight.replace(day=1)
        return start, end, "month", "Interpreted as the previous calendar month.", 0.7
    if kind == "this_month":
        start = midnight.replace(day=1)
        end = (start + dt.timedelta(days=32)).replace(day=1)
        return start, end, "month", "Interpreted as the record's calendar month.", 0.75
    if kind == "last_year":
        start = midnight.replace(year=midnight.year - 1, month=1, day=1)
        end = midnight.replace(month=1, day=1)
        return start, end, "year", "Interpreted as the previous calendar year.", 0.7
    if kind in {"in_minutes", "in_hours", "in_days"}:
        n = int(match.group("n"))
        delta = {
            "in_minutes": dt.timedelta(minutes=n),
            "in_hours": dt.timedelta(hours=n),
            "in_days": dt.timedelta(days=n),
        }[kind]
        target = anchor + delta
        return (
            target,
            target,
            "minute" if kind != "in_days" else "day",
            "A deadline stated relative to the record's timestamp.",
            0.85,
        )
    if kind in {"minutes_ago", "hours_ago", "days_ago"}:
        n = int(match.group("n"))
        delta = {
            "minutes_ago": dt.timedelta(minutes=n),
            "hours_ago": dt.timedelta(hours=n),
            "days_ago": dt.timedelta(days=n),
        }[kind]
        target = anchor - delta
        return (
            target,
            target,
            "minute" if kind != "days_ago" else "day",
            "A time stated relative to the record's timestamp.",
            0.8,
        )
    if kind == "weekday":
        name = match.group("day").lower()
        target_day = DAY_NAMES.split("|").index(name)
        delta = (target_day - midnight.weekday()) % 7
        target = midnight + dt.timedelta(days=delta)
        # A bare weekday name is ambiguous: it could mean the coming occurrence or the
        # last one. We report the coming occurrence and say so.
        return (
            target,
            target + dt.timedelta(days=1),
            "day",
            "A weekday name without a week reference; shown as the next occurrence.",
            0.6,
        )
    if kind == "relative_meeting":
        inner = match.group("when")
        when_match = re.match(r"last\s+week|earlier\s+last\s+week", inner, re.I)
        if when_match:
            week_start = midnight - dt.timedelta(days=midnight.weekday() + 7)
            return (
                week_start,
                week_start + dt.timedelta(days=7),
                "week",
                "Refers to a discussion in the week before the record. Exact date not stated.",
                0.6,
            )
        return None
    return None


def find_backreferences(text: str) -> list[tuple[int, int, str]]:
    """Locate phrases that point at a discussion DigitalMind may not have.

    Returns ``(start, end, matched_text)``. Used by the Scenario B logic: when a record
    refers back to a prior discussion, we search for it and, if nothing matches, say
    "no matching discussion was found in the records analysed" - never "it never
    happened".
    """
    out: list[tuple[int, int, str]] = []
    claimed: list[tuple[int, int]] = []
    for pattern in BACKREFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            if any(not (match.end() <= s or match.start() >= e) for s, e in claimed):
                continue
            claimed.append((match.start(), match.end()))
            out.append((match.start(), match.end(), match.group(0)))
    out.sort(key=lambda t: t[0])
    return out

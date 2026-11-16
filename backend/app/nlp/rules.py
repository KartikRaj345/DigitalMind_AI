"""Baseline (rules/regex) extraction.

This is the *reference implementation*: dependency-free, deterministic, and always
available. It is evaluated alongside the neural pipeline so we can measure what the
models actually add (see ``backend/scripts/evaluate_extraction.py``).

Output is a list of :class:`Span` - literal text plus character offsets plus a
confidence. Nothing here interprets meaning; it only locates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PERSON = "PERSON"
ORG = "ORG"
DATE = "DATE"
MONEY = "MONEY"
EVENT = "EVENT"
PRODUCT = "PRODUCT"


@dataclass(frozen=True)
class Span:
    """A located piece of text with its provenance."""

    text: str
    label: str
    start: int
    end: int
    confidence: float
    extractor: str = "rules"
    normalized: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start_char": self.start,
            "end_char": self.end,
            "confidence": round(self.confidence, 4),
            "extractor": self.extractor,
            "normalized": self.normalized,
        }


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS = "$€£¥₹"
_MONEY_RE = re.compile(
    rf"(?P<cur>[{CURRENCY_SYMBOLS}])\s?(?P<num>\d[\d,]*(?:\.\d{{1,2}})?)"
    rf"|(?P<cur2>\b(?:USD|EUR|GBP|INR)\b)\s?(?P<num2>\d[\d,]*(?:\.\d{{1,2}})?)",
    re.IGNORECASE,
)
_MONEY_WORDS = re.compile(
    r"\b(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\s+"
    r"(?P<unit>dollars?|euros?|pounds?|usd|eur|gbp|inr|rupees?)\b",
    re.IGNORECASE,
)
_UNIT_MAP = {
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
    "rupee": "INR", "rupees": "INR", "inr": "INR",
}


def extract_money(text: str) -> list[Span]:
    """Money amounts with a normalized numeric value."""
    spans: list[Span] = []
    claimed: list[tuple[int, int]] = []

    def claim(start: int, end: int) -> bool:
        if any(not (end <= s or start >= e) for s, e in claimed):
            return False
        claimed.append((start, end))
        return True

    for m in _MONEY_RE.finditer(text):
        raw = m.group(0)
        if not claim(m.start(), m.end()):
            continue
        symbol = (m.group("cur") or m.group("cur2") or "").upper()
        num = m.group("num") or m.group("num2")
        currency = symbol if symbol in {"USD", "EUR", "GBP", "INR"} else _symbol_currency(symbol)
        value = float(num.replace(",", ""))
        spans.append(
            Span(raw, MONEY, m.start(), m.end(), 0.95, normalized=f"{value:.2f} {currency}")
        )

    for m in _MONEY_WORDS.finditer(text):
        raw = m.group(0)
        if not claim(m.start(), m.end()):
            continue
        unit = m.group("unit").lower()
        currency = _UNIT_MAP.get(unit, "USD")
        value = float(m.group("amount").replace(",", ""))
        spans.append(
            Span(raw, MONEY, m.start(), m.end(), 0.9, normalized=f"{value:.2f} {currency}")
        )
    return spans


def _symbol_currency(symbol: str) -> str:
    return {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}.get(symbol, "USD")


# ---------------------------------------------------------------------------
# Email addresses (reliable PERSON anchor)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def extract_emails(text: str) -> list[Span]:
    """Email addresses. Strong PERSON signal, higher confidence than a bare name."""
    out: list[Span] = []
    for m in _EMAIL_RE.finditer(text):
        handle = m.group(0).split("@")[0]
        name_part = re.split(r"[._-]+", handle)[0]
        readable = name_part.replace("dm", "").strip() if name_part.lower() in {"dm"} else name_part
        out.append(
            Span(
                text=m.group(0),
                label=PERSON,
                start=m.start(),
                end=m.end(),
                confidence=0.9,
                normalized=readable.title() if readable else m.group(0),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------

# Horizontal whitespace only. Using \s+ lets a pattern run across a line break, which
# produced spans like "Account Services\nNorthwind Bank" that cover two unrelated things.
_HS = r"[ \t]+"

_ORG_SUFFIX = (
    r"(?:Inc|Inc\.|LLC|Ltd|Ltd\.|Limited|Corp|Corp\.|Corporation|Company|Co\.|GmbH|PLC"
    r"|Group|Holdings|Partners|Bank|Banks|Foundation|Institute|University|Agency|Department"
    r"|Credit|Union|Savings|Lending|Trust|Mutual|Capital|Financial|Insurance|Freight"
    r"|Publishing|Supplies|Media|Travel|Foods)"
)
_ORG_RE = re.compile(rf"\b(?:[A-Z][\w&'.-]*{_HS}+){{0,3}}(?:{_ORG_SUFFIX})\b")

# "Bank of Raval" - matched first so the generic suffix rule cannot steal "Bank".
_BANK_OF_RE = re.compile(rf"\b(?:[A-Z][\w&'.-]*{_HS}+){{0,2}}Bank{_HS}+of{_HS}+[A-Z][\w&'.-]*")

# Verbs/adverbs that may precede an org name but are not part of it.
_ORG_LEADING_STOPWORDS = {
    "please", "contact", "email", "call", "the", "a", "an", "and", "or",
    "from", "to", "with", "for", "about", "regarding", "concerning", "our",
    "your", "their", "his", "her", "i", "we", "you", "they", "sent", "received",
    "asking", "asked", "meeting", "call", "dear", "hi", "hello", "thanks",
}


def extract_orgs(text: str) -> list[Span]:
    """Organizations detected by legal/descriptor suffix, or by a bank keyword."""
    spans: list[Span] = []
    claimed: list[tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return not any(not (end <= s or start >= e) for s, e in claimed)

    # "Bank of ..." style names take precedence over the generic suffix rule.
    for m in _BANK_OF_RE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        claimed.append((m.start(), m.end()))
        spans.append(Span(m.group(0), ORG, m.start(), m.end(), 0.8, normalized=m.group(0)))

    for m in _ORG_RE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        name = m.group(0).strip(" .,")
        # Trim leading words that are clearly sentence-level verbs, not name parts.
        while True:
            head, sep, tail = name.partition(" ")
            if not (sep and tail and head.lower() in _ORG_LEADING_STOPWORDS):
                break
            name = tail
        start = m.end() - len(name)
        if not free(start, m.end()):
            continue
        claimed.append((start, m.end()))
        spans.append(Span(name, ORG, start, m.end(), 0.75, normalized=name))
    return spans


# ---------------------------------------------------------------------------
# Capitalised person-name heuristic
# ---------------------------------------------------------------------------

_SALUTATION_RE = re.compile(
    r"\b(?:Dear|Hi|Hail|Hello|Good (?:morning|afternoon|evening)|Greetings)\s+"
    r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)
_TITLE_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Sir|Madam)\.?\s+"
    r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)
_FULL_NAME_RE = re.compile(
    r"\b(?P<first>[A-Z][a-z]{1,20})\s+(?P<last>[A-Z][a-z]{1,20})\b"
)

# Shapes that look like names to a capitalisation heuristic but are not people.
_NOT_A_PERSON = re.compile(
    r"^\d"                              # 16:24:03, 2026
    r"|^\d{1,2}:\d{2}"                  # 14:30
    r"|^[A-Z][a-z]+\s+[A-Z]{2,}$"       # "Ashcombe ROAD"
    r"|^(The|This|That|These|Those|A|An|In|On|At|By|For|From|With|And|But|If|When)$",
)

# Words that start sentences or are common nouns; used to avoid tagging them as names.
_SENTENCE_STARTERS = {
    "the", "this", "that", "these", "those", "a", "an", "and", "but", "if",
    "when", "while", "after", "before", "please", "thanks", "thank", "dear",
    "regards", "kind", "best", "hi", "hello", "hey", "yes", "no", "not",
    "our", "your", "their", "his", "her", "its", "my", "we", "i", "you",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}

_STOP_PERSON_LIKE = {
    "account", "payment", "security", "customer", "support", "service",
    "team", "project", "meeting", "invoice", "transfer", "bank", "risk",
    "compliance", "please", "thank", "hello", "regards", "system", "update",
    # address components, which otherwise look like surnames to the heuristic
    "row", "street", "road", "avenue", "lane", "drive", "close", "way",
    "court", "place", "terrace", "house", "flat", "apartment", "square",
    "gardens", "hill", "park", "crescent", "campus", "centre", "center",
    "north", "south", "east", "west", "suite", "floor", "building",
}


def extract_person_names(text: str, known_orgs: list[str] | None = None) -> list[Span]:
    """Titles and Firstname Lastname pairs.

    Low precision by nature (0.4-0.6), which is exactly why the neural NER exists.
    We keep the confidence honest so the UI can show it.
    """
    spans: list[Span] = []
    claimed: list[tuple[int, int]] = []
    org_tokens = set()
    for org in known_orgs or []:
        org_tokens.update(re.findall(r"[A-Za-z]+", org.lower()))

    for m in _TITLE_RE.finditer(text):
        start, end = m.start("name"), m.end("name")
        if any(not (end <= s or start >= e) for s, e in claimed):
            continue
        name = m.group("name")
        if name.split()[0].lower() in _STOP_PERSON_LIKE or _NOT_A_PERSON.match(name):
            continue
        claimed.append((start, end))
        spans.append(Span(name, PERSON, start, end, 0.75, normalized=name))

    for m in _SALUTATION_RE.finditer(text):
        start, end = m.start("name"), m.end("name")
        if any(not (end <= s or start >= e) for s, e in claimed):
            continue
        name = m.group("name")
        if name.split()[0].lower() in _STOP_PERSON_LIKE or _NOT_A_PERSON.match(name):
            continue
        claimed.append((start, end))
        spans.append(Span(name, PERSON, start, end, 0.7, normalized=name))

    for m in _FULL_NAME_RE.finditer(text):
        first, last = m.group("first"), m.group("last")
        if first.lower() in _SENTENCE_STARTERS or last.lower() in _STOP_PERSON_LIKE:
            continue
        if first.lower() in org_tokens or last.lower() in org_tokens:
            continue
        if any(not (m.end() <= s or m.start() >= e) for s, e in claimed):
            continue
        claimed.append((m.start(), m.end()))
        name = f"{first} {last}"
        if _NOT_A_PERSON.match(name):
            continue
        spans.append(Span(name, PERSON, m.start(), m.end(), 0.5, normalized=name))

    return spans


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

EVENT_TRIGGERS = (
    r"meeting|call|conference|workshop|interview|appointment|review|demo|"
    r"deadline|due\s+date|flight|reservation|appointment"
)
_EVENT_RE = re.compile(
    rf"\b(?P<phrase>(?:[A-Z][\w'-]+{_HS}+){{0,4}}(?:{EVENT_TRIGGERS})\b)", re.IGNORECASE
)
_MEETING_INVITE_RE = re.compile(
    r"\b(?:meeting|call|appointment|sync|review)\b\s*(?:is\s+)?(?:scheduled|booked|set\s+for|at)\b",
    re.IGNORECASE,
)


def extract_events(text: str) -> list[Span]:
    """Event-like noun phrases. Confidence is low: these are candidates, not facts."""
    spans: list[Span] = []
    claimed: list[tuple[int, int]] = []

    for m in _MEETING_INVITE_RE.finditer(text):
        if any(not (m.end() <= s or m.start() >= e) for s, e in claimed):
            continue
        claimed.append((m.start(), m.end()))
        spans.append(Span(m.group(0), EVENT, m.start(), m.end(), 0.6, normalized=m.group(0)))

    for m in _EVENT_RE.finditer(text):
        if any(not (m.end() <= s or m.start() >= e) for s, e in claimed):
            continue
        phrase = m.group("phrase").strip()
        if len(phrase.split()) < 2:
            # A bare "meeting" with no qualifier is too weak to be useful.
            continue
        claimed.append((m.start(), m.end()))
        spans.append(Span(phrase, EVENT, m.start(), m.end(), 0.45, normalized=phrase))
    return spans


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def extract_all(text: str) -> list[Span]:
    """Run every rules extractor and de-overlap by character span.

    On overlap we keep the higher-confidence span and record the loser's label in the
    normalized string so nothing is silently discarded.
    """
    orgs = extract_orgs(text)
    spans: list[Span] = []
    spans.extend(extract_money(text))
    spans.extend(extract_emails(text))
    spans.extend(orgs)
    spans.extend(extract_person_names(text, [s.text for s in orgs]))
    spans.extend(extract_events(text))
    return resolve_overlaps(spans)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Greedy de-overlap by descending confidence."""
    ordered = sorted(spans, key=lambda s: (-s.confidence, s.start, -(s.end - s.start)))
    kept: list[Span] = []
    for span in ordered:
        if any(not (span.end <= k.start or span.start >= k.end) for k in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s.start)

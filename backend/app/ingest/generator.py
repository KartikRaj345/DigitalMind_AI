"""Seeded synthetic dataset generation.

Produces two things from one deterministic generator (``seed`` controls everything):

1. **Demo corpus** - the hand-authored Scenario A/B narrative plus benign look-alikes.
   This is what ``--demo`` loads into the app.
2. **Evaluation corpus** - 300+ records with gold entity spans, gold risk labels and
   gold back-reference targets, split train/dev/test. Used by every eval script.

All names, organizations and amounts are fictional. Any resemblance to a real bank or
person is coincidental; the generator deliberately avoids real institution names.

Design rules for the generator:

* Templates are drawn per record type so the corpus resembles the six supported types.
* Benign look-alikes are generated with the *same surface features* as the risky messages
  (deadlines, amounts, "account" vocabulary, urgency words) so precision is actually
  tested. This is the single most important property of the dataset.
* Risk labels are multi-label: a record can be urgent but legitimate.
* Gold entity spans are recorded as exact offsets so NER can be scored with strict IoU.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Priya", "Marcus", "Ana", "Tobias", "Lena", "Rafael", "Nadia", "Colin",
    "Simone", "Hiro", "Beatrice", "Omar", "Greta", "Diego", "Yusuf", "Clara",
    "Mateo", "Ingrid", "Ravi", "Sofia", "Anders", "Leila", "Kwame", "Freya",
]
LAST_NAMES = [
    "Raman", "Delgado", "Novak", "Okafor", "Lindqvist", "Fontaine", "Whitfield",
    "Bianchi", "Kowalski", "Ferreira", "Nakamura", "Abadi", "Mbeki", "Sorensen",
    "Castellanos", "Petrov", "Halvorsen", "Villanueva", "Ashworth", "Bergstrom",
]
ORGANIZATIONS = [
    "Northwind Bank", "Harbourline Credit Union", "Cedar Peak Savings",
    "Meridian Trust", "Lakeside Mutual", "Bright Anchor Lending",
]
VENDORS = [
    "Atlas Freight", "Quill Publishing", "Verdant Foods", "Ironbridge Supplies",
    "Solace Travel", "Copperline Media",
]
PROJECTS = [
    "Project Atlas", "Project Beacon", "Project Cedar", "Project Lantern",
    "Project Meridian",
]
TEAMS = ["Operations", "Finance", "Compliance", "Logistics", "Platform"]

SOURCE_TYPES = ["email", "message", "document", "note", "calendar", "task"]

RISK_LABELS = ["urgency", "threat_language", "emotional_pressure", "payment_request"]


# ---------------------------------------------------------------------------
# Template fragments
# ---------------------------------------------------------------------------

# Each fragment: (text, labels it contributes, whether it is a "benign look-alike"
# feature). Labels are exactly the four risk indicators.
URGENCY_TEMPLATES = [
    "This must be resolved within 10 minutes.",
    "We need your response in the next 15 minutes to avoid a hold.",
    "Please respond immediately.",
    "Treat this as urgent - the deadline is today at 17:00.",
    "We have 30 minutes left on this window.",
]
URGENCY_BENIGN = [
    "The support team's response target is 30 minutes, per our SLA.",
    "Our quarterly review is scheduled for today at 17:00.",
    "Please review the attached draft before the 15:00 standup tomorrow.",
    "This item is due in the next sprint; no escalation is required.",
]
THREAT_TEMPLATES = [
    "Your account will be closed today unless action is taken.",
    "Failure to comply will result in immediate suspension of your account.",
    "We are terminating your access if this is not settled.",
    "Service will be permanently disabled if payment is not received.",
]
THREAT_BENIGN = [
    "Accounts without verified addresses are closed after 30 days, per policy.",
    "Access is suspended automatically when a password expires.",
    "Our policy states that dormant accounts are closed after 24 months.",
]
EMOTION_TEMPLATES = [
    "We are extremely worried about what this may mean for your family.",
    "I am terrified we will lose everything if this is not fixed.",
    "This is devastating - please act, we have no other options.",
    "You must understand how serious this is, we are desperate.",
]
EMOTION_BENIGN = [
    "I am sorry for the inconvenience this may have caused.",
    "We appreciate your patience while we sort this out.",
    "Apologies again for the confusion in the last invoice.",
]
PAYMENT_TEMPLATES = [
    "Please transfer $4,250.00 to the verification account immediately.",
    "Wire 1,200 euros to account 8831 4472 before the cutoff.",
    "Send $980.75 to the new supplier bank details provided below.",
    "A payment of 12,500 dollars is required to release the shipment.",
]
PAYMENT_BENIGN = [
    "Your scheduled payment of $240.00 will be collected on the 3rd as usual.",
    "The invoice total of 3,120.40 euros is due on 18 April, per the attached statement.",
    "Payment of $75.00 for the annual subscription renews automatically.",
]

TOPICS = [
    "the account verification review",
    "the Q2 supplier reconciliation",
    "the updated supplier bank details",
    "the pending invoice for the Harbourline account",
    "the shipment release documentation",
]

# Absolute dates appear in a minority of records on purpose: the date-normalization
# evaluation needs enough of them to be measurable, but a corpus that is mostly absolute
# dates would hide how the system handles relative wording.
ABSOLUTE_DATE_PHRASES = [
    "The review is scheduled for {date}.",
    "We agreed the deadline is {date}.",
    "The reconciliation window closes on {date}.",
    "Please diarise {date} for the supplier call.",
    "Documentation is due by {date}.",
]

ABSOLUTE_DATES = [
    "4 March 2026", "March 4, 2026", "2026-03-11", "18 April 2026",
    "12/03/2026", "27 February 2026", "March 2026", "2026-04-02",
    "9 March 2026", "30/04/2026",
]

TIME_PHRASES = [
    "at 14:30", "at 09:00", "at 17:45", "at 08:15",
]

BENIGN_BODIES = [
    "Sharing the notes from our {topic}. {extra} Let me know if anything is missing.",
    "Following up on {topic}. {extra} No action needed on your side.",
    "For reference, {extra} I have attached the summary we discussed.",
    "Quick update on {topic}: {extra} I will circulate the final version tomorrow.",
    "Update on {topic}. {extra} The next checkpoint is {absdate}{time}.",
    "Hi {sender},\n\n{topic}: {extra} The next checkpoint is {absdate}{time}.\n\nThanks,\n{sender}",
]

MEETING_TITLES = [
    "Account verification review",
    "Supplier reconciliation sync",
    "Quarterly planning",
    "Compliance walkthrough",
    "Operations handover",
]

DOC_TITLES = [
    "Account information summary",
    "Supplier payment terms",
    "Verification checklist",
    "Onboarding record extract",
]

TASK_TITLES = [
    "Confirm supplier bank details by phone",
    "Send verification documents",
    "Reconcile Q2 invoices",
    "Update account recovery contacts",
]


@dataclass
class GoldSpan:
    text: str
    label: str
    start: int
    end: int


@dataclass
class GoldRecord:
    """A generated record plus its gold annotations."""

    id: str
    source_type: str
    title: str
    body: str
    occurred_at: dt.datetime
    participants: list[str] = field(default_factory=list)
    risk_labels: list[str] = field(default_factory=list)
    entity_spans: list[GoldSpan] = field(default_factory=list)
    backreference: bool = False
    scenario: str = ""
    benign_lookalike: bool = False

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "title": self.title,
            "body": self.body,
            "occurred_at": self.occurred_at.isoformat(),
            "participants": self.participants,
            "risk_labels": self.risk_labels,
            "entity_spans": [
                {"text": s.text, "label": s.label, "start": s.start, "end": s.end}
                for s in self.entity_spans
            ],
            "backreference": self.backreference,
            "scenario": self.scenario,
            "benign_lookalike": self.benign_lookalike,
        }


# ---------------------------------------------------------------------------
# Span-aware building
# ---------------------------------------------------------------------------


class BodyBuilder:
    """Assembles a body while recording exact offsets for the pieces we annotate."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._spans: list[GoldSpan] = []
        self._length = 0

    def add(self, text: str) -> "BodyBuilder":
        self._parts.append(text)
        self._length += len(text)
        return self

    def add_entity(self, text: str, label: str) -> "BodyBuilder":
        start = self._length
        self.add(text)
        self._spans.append(GoldSpan(text, label, start, self._length))
        return self

    def build(self) -> tuple[str, list[GoldSpan]]:
        return "".join(self._parts), list(self._spans)


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _email_for(name: str, org: str) -> str:
    slug = name.lower().replace(" ", ".")
    domain = org.lower().replace(" ", "").replace(",", "")
    return f"{slug}@{domain}.example"


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------


def _risky_record(rng: random.Random, rid: str, when: dt.datetime) -> GoldRecord:
    """A message carrying at least two risk indicators (the attack pattern)."""
    person = _person(rng)
    org = rng.choice(ORGANIZATIONS)
    topic = rng.choice(TOPICS)

    b = BodyBuilder()
    b.add("Dear ")
    b.add_entity(person, "PERSON")
    b.add(",\n\n")
    b.add(f"This is regarding {topic}. ")
    b.add(rng.choice(URGENCY_TEMPLATES) + " ")
    b.add(rng.choice(EMOTION_TEMPLATES) + " ")
    b.add(rng.choice(PAYMENT_TEMPLATES) + " ")
    b.add(rng.choice(THREAT_TEMPLATES) + " ")
    b.add(f"Reference: {rng.choice(PROJECTS)}, logged {rng.choice(ABSOLUTE_DATES)}.")
    b.add("Please reply to this message only.\n\nRegards,\n")
    b.add_entity("Account Services", "ORG")
    b.add(f"\n{org}")

    labels: list[str] = []
    for pool, label in (
        (URGENCY_TEMPLATES, "urgency"),
        (EMOTION_TEMPLATES, "emotional_pressure"),
        (PAYMENT_TEMPLATES, "payment_request"),
        (THREAT_TEMPLATES, "threat_language"),
    ):
        if rng.random() < 0.85:
            labels.append(label)

    body, spans = b.build()
    spans.extend(_locate_entities(body, org))
    spans.extend(_locate_money(body))
    spans.sort(key=lambda s: s.start)
    return GoldRecord(
        id=rid,
        source_type="message" if rng.random() < 0.6 else "email",
        title=f"Action required - {topic}",
        body=body,
        occurred_at=when,
        participants=[person, "Account Services", org],
        risk_labels=sorted(set(labels)),
        entity_spans=spans,
        scenario="generated_risk",
    )


def _benign_lookalike(rng: random.Random, rid: str, when: dt.datetime) -> GoldRecord:
    """A legitimate record with the *same* surface features as a risky one.

    This is the false-positive control: if the model cannot separate these from the
    attack records, the evaluation is meaningless.
    """
    person = _person(rng)
    org = rng.choice(ORGANIZATIONS)
    topic = rng.choice(TOPICS)

    b = BodyBuilder()
    b.add("Hello ")
    b.add_entity(person, "PERSON")
    b.add(",\n\n")
    b.add(f"Some notes on {topic}. ")
    b.add(rng.choice(URGENCY_BENIGN) + " ")
    b.add(rng.choice(THREAT_BENIGN) + " ")
    b.add(rng.choice(PAYMENT_BENIGN) + " ")
    b.add(rng.choice(EMOTION_BENIGN) + " ")
    b.add_entity(org, "ORG")
    b.add(" remains the account holder of record.\n\nThanks,\n")
    b.add_entity(_person(rng), "PERSON")

    body, spans = b.build()
    spans.extend(_locate_money(body))
    spans.sort(key=lambda s: s.start)
    return GoldRecord(
        id=rid,
        source_type=rng.choice(["email", "message", "document"]),
        title=f"Re: {topic}",
        body=body,
        occurred_at=when,
        participants=[person, org],
        risk_labels=[],
        entity_spans=spans,
        scenario="generated_benign_lookalike",
        benign_lookalike=True,
    )


def _plain_record(rng: random.Random, rid: str, when: dt.datetime) -> GoldRecord:
    person = _person(rng)
    org = rng.choice(ORGANIZATIONS)
    topic = rng.choice(TOPICS)
    team = rng.choice(TEAMS)
    extra = rng.choice(
        [
            "the figures in section 3 have been updated",
            "the shipping window moved by two days",
            "the second reviewer is on leave until the 14th",
            "we archived the older copies",
        ]
    )
    stype = rng.choice(SOURCE_TYPES)
    absdate = rng.choice(ABSOLUTE_DATES)
    when_time = rng.choice(TIME_PHRASES)
    second = ""

    if stype == "calendar":
        title = rng.choice(MEETING_TITLES)
        second = _person(rng)
        body = (
            f"Meeting: {title}\n"
            f"Attendees: {person}, {second}\n"
            f"Agenda: {topic}; review of {extra}.\n"
            f"Scheduled for {absdate}{when_time}.\n"
            f"Location: Room 4B (or the usual video link)."
        )
        participants = [person, second]
    elif stype == "task":
        title = rng.choice(TASK_TITLES)
        body = (
            f"{title}. Related to {topic}. Owner: {person}. Due: {absdate}.\n"
            f"Note that {extra}."
        )
        participants = [person]
    elif stype == "document":
        title = rng.choice(DOC_TITLES)
        body = (
            f"{title}\n\n"
            f"Account holder: {person}\n"
            f"Institution: {org}\n"
            f"Reference: {rng.choice(PROJECTS)}\n"
            f"Effective date: {absdate}\n"
            f"Notes: {topic}; {extra}."
        )
        participants = [person, org]
    else:
        # Every template must include the sampled person. When the team-update templates
        # omitted them, two records that drew the same (topic, extra, date) produced
        # byte-identical bodies, which put duplicates on both sides of the train/test
        # split and left those records with no gold span for their person.
        title = f"{team} update"
        body = rng.choice(BENIGN_BODIES).format(
            topic=topic,
            extra=extra,
            absdate=absdate,
            time=when_time,
            sender=person,
        )
        participants = [person]

    attendees = second if stype == "calendar" else ""
    spans = _locate_entities(body, person, org, attendees)
    return GoldRecord(
        id=rid,
        source_type=stype,
        title=title,
        body=body,
        occurred_at=when,
        participants=participants,
        risk_labels=[],
        entity_spans=spans,
        scenario="generated_plain",
    )


def _locate_money(body: str) -> list[GoldSpan]:
    """Locate every money amount using the same regexes the app uses.

    Re-using the app's own money patterns for gold annotation is a deliberate, disclosed
    choice: it makes the MONEY label's gold set exact. It also means the MONEY F1 measures
    *whether the pipeline finds the amounts the templates contain*, not whether the amount
    regex is correct - the latter is covered by unit tests in ``test_nlp_rules.py``.
    """
    from ..nlp import rules

    return [
        GoldSpan(s.text, "MONEY", s.start, s.end) for s in rules.extract_money(body)
    ]


def _locate_entities(body: str, *needles: str) -> list[GoldSpan]:
    """Locate every occurrence of each entity so the gold set is not under-annotated.

    Annotating only the first occurrence would make every later mention a "false positive"
    and would understate measured precision for reasons that have nothing to do with the
    model.
    """
    spans: list[GoldSpan] = []
    for needle in needles:
        if not needle:
            continue
        label = "ORG" if needle in ORGANIZATIONS + VENDORS else "PERSON"
        start = 0
        while True:
            idx = body.find(needle, start)
            if idx < 0:
                break
            end = idx + len(needle)
            span = GoldSpan(needle, label, idx, end)
            if not any(not (span.end <= s.start or span.start >= s.end) for s in spans):
                spans.append(span)
            start = idx + 1
    return sorted(spans, key=lambda s: s.start)


def _backref_record(rng: random.Random, rid: str, when: dt.datetime) -> GoldRecord:
    """Scenario B: asserts a prior discussion. Only some of these have a match."""
    person = _person(rng)
    org = rng.choice(ORGANIZATIONS)
    topic = rng.choice(TOPICS)
    b = BodyBuilder()
    b.add("Hi ")
    b.add_entity(person, "PERSON")
    b.add(",\n")
    b.add("As we discussed last week, ")
    b.add(f"we agreed to move the {topic} review to a later date. ")
    b.add("Please confirm you are still happy with that plan.")
    body, _ = b.build()
    spans = _locate_entities(body, person)
    return GoldRecord(
        id=rid,
        source_type=rng.choice(["message", "email"]),
        title=f"Re: {topic}",
        body=body,
        occurred_at=when,
        participants=[person, org],
        risk_labels=[],
        entity_spans=spans,
        backreference=True,
        scenario="generated_backref",
    )


def generate_corpus(
    n: int = 320,
    seed: int = 20260310,
    start: dt.datetime | None = None,
    span_days: int = 120,
    max_retries: int = 40,
) -> list[GoldRecord]:
    """Generate ``n`` records with gold labels. Fully deterministic given ``seed``.

    Records are de-duplicated on their body text. Two templates can legitimately produce
    the same words when the same entity/topic/date combination is drawn twice, and a
    duplicate that lands in both train and test is train/test leakage: the model can score
    a perfect hit by memorising one string rather than learning anything. Resampling until
    the body is new is the fix, and it keeps determinism because the RNG stream stays
    sequential.
    """
    rng = random.Random(seed)
    base = start or dt.datetime(2026, 3, 1, 9, 0, tzinfo=dt.timezone.utc)

    # Composition. The risky fraction is deliberately modest so precision is testable.
    n_risk = max(1, int(n * 0.20))
    n_lookalike = max(1, int(n * 0.30))
    n_backref = max(1, int(n * 0.10))

    seen_bodies: set[str] = set()
    records: list[GoldRecord] = []

    def emit(builder, rid: str, prefix: str) -> None:
        """Generate one record, resampling until its body is unique."""
        for attempt in range(max_retries):
            when = base + dt.timedelta(
                days=rng.randrange(span_days), minutes=rng.randrange(0, 24 * 60)
            )
            record = builder(rng, f"{prefix}{attempt:03d}" if attempt else rid, when)
            key = record.body.strip()
            if key in seen_bodies:
                continue
            seen_bodies.add(key)
            records.append(record)
            return
        raise RuntimeError(
            f"could not generate a unique body for {rid} after {max_retries} attempts; "
            "increase n, widen the template pools, or lower the dedup strictness"
        )

    for i in range(n_risk):
        emit(_risky_record, f"risk-{i:04d}", f"risk-{i:04d}-")

    for i in range(n_lookalike):
        emit(_benign_lookalike, f"lookalike-{i:04d}", f"lookalike-{i:04d}-")

    for i in range(n_backref):
        emit(_backref_record, f"backref-{i:04d}", f"backref-{i:04d}-")

    for i in range(max(0, n - len(records))):
        emit(_plain_record, f"plain-{i:04d}", f"plain-{i:04d}-")

    records.sort(key=lambda r: (r.occurred_at, r.id))
    return records


def split_records(
    records: list[GoldRecord], seed: int = 20260310, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> dict[str, list[GoldRecord]]:
    """Stratified split by scenario so each set keeps the risky/benign mix.

    A plain random split would put all of one scenario in one fold and make the dev set
    useless for threshold selection.
    """
    rng = random.Random(seed)
    by_scenario: dict[str, list[GoldRecord]] = {}
    for rec in records:
        by_scenario.setdefault(rec.scenario, []).append(rec)

    out: dict[str, list[GoldRecord]] = {"train": [], "dev": [], "test": []}
    names = ("train", "dev", "test")
    for scenario, group in sorted(by_scenario.items()):
        group = sorted(group, key=lambda r: r.id)
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * ratios[0]))
        n_dev = int(round(n * ratios[1]))
        out["train"].extend(group[:n_train])
        out["dev"].extend(group[n_train : n_train + n_dev])
        out["test"].extend(group[n_train + n_dev :])

    for key in out:
        out[key].sort(key=lambda r: (r.occurred_at, r.id))
    return out


def write_jsonl(records: list[GoldRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.as_dict(), ensure_ascii=False) + "\n")
    return path


def write_dataset(out_dir: Path, n: int = 320, seed: int = 20260310) -> dict[str, Path]:
    records = generate_corpus(n=n, seed=seed)
    splits = split_records(records, seed=seed)
    written = {
        "all": write_jsonl(records, out_dir / "all.jsonl"),
        "train": write_jsonl(splits["train"], out_dir / "train.jsonl"),
        "dev": write_jsonl(splits["dev"], out_dir / "dev.jsonl"),
        "test": write_jsonl(splits["test"], out_dir / "test.jsonl"),
    }
    meta = {
        "seed": seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "total": len(records),
        "positive_rate": {
            k: sum(1 for r in v if r.risk_labels) / len(v) if v else 0.0
            for k, v in splits.items()
        },
        "scenarios": {
            k: sorted({r.scenario for r in v}) for k, v in splits.items()
        },
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return written

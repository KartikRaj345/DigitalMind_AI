"""Risk lexicon and rule-based risk detection (the baseline).

Four indicators are detected **independently** and reported separately:

* ``urgency``            - deadlines, time pressure
* ``threat_language``    - consequences stated as certain ("your account will be closed")
* ``emotional_pressure`` - fear/pleading/appalling language aimed at the recipient
* ``payment_request``    - a demand that money move

There is deliberately **no** combined score in this module. Combining happens later, in
:mod:`app.risk.assess`, and only with an explicit formula that the UI shows.

Lexicon entries carry a weight and a note. The note is shown to the user next to the
matched span, so "why was this flagged?" always has an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

URGENCY = "urgency"
THREAT = "threat_language"
EMOTION = "emotional_pressure"
PAYMENT = "payment_request"

RISK_LABELS = (URGENCY, THREAT, EMOTION, PAYMENT)


@dataclass(frozen=True)
class LexEntry:
    pattern: str
    label: str
    weight: float
    note: str
    flags: int = re.IGNORECASE

    def regex(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


# ---------------------------------------------------------------------------
# Urgency
# ---------------------------------------------------------------------------

URGENCY_ENTRIES = [
    LexEntry(
        r"\bwithin\s+(\d{1,3}\s+)?(minutes?|hours?)\b", URGENCY, 0.9,
        "A deadline is expressed in minutes or hours from now.",
    ),
    LexEntry(
        r"\bin\s+(\d{1,3}\s+)?(minutes?|hours?)\b", URGENCY, 0.7,
        "A short time window is given.",
    ),
    LexEntry(
        r"\b(immediately|right away|asap|at once|without delay)\b", URGENCY, 0.8,
        "Immediate action is demanded.",
    ),
    LexEntry(
        r"\b(urgent|urgently|asap|critical|emergency|final notice|last chance)\b",
        URGENCY, 0.65,
        "The message is marked urgent.",
    ),
    LexEntry(
        r"\b(respond|reply|act|confirm|pay|transfer|settle)\s+(now|today|before)\b",
        URGENCY, 0.6,
        "Action is demanded before a stated point in time.",
    ),
    LexEntry(
        r"\bonly\s+\d{1,3}\s+(minutes?|hours?)\s+(left|remaining)\b", URGENCY, 0.85,
        "A countdown is stated.",
    ),
    LexEntry(
        r"\b(deadline|cut[- ]?off|expires?|expiring)\b", URGENCY, 0.5,
        "A deadline is referenced.",
    ),
    LexEntry(
        r"\b(pending|suspended)\s+(for|until|pending)\b", URGENCY, 0.4,
        "A hold is referenced.",
    ),
    LexEntry(
        r"\bwill\s+be\s+(closed|suspended|disabled|terminated|revoked)\b", URGENCY, 0.5,
        "A future state change is asserted.",
    ),
]

# Legitimate contexts. Presence of any of these lowers urgency confidence, because
# "urgent" in a genuine reminder usually comes with a real, checkable channel.
URGENCY_MITIGATORS = [
    LexEntry(
        r"\b(no charge|no fee|there is no (charge|fee)|without charge)\b", URGENCY, -0.4,
        "The message states that no payment is required.",
    ),
    LexEntry(
        r"\b(we will never ask|never ask you)\b", URGENCY, -0.4,
        "The message explicitly denies asking for money.",
    ),
    LexEntry(
        r"\b(scheduled|scheduled for|response target|service level)\b", URGENCY, -0.25,
        "The timing refers to a planned or policy-driven event.",
    ),
    LexEntry(
        r"\b(per our|according to our|as set out in|per the)\b", URGENCY, -0.15,
        "The statement references existing policy rather than a new demand.",
    ),
]

# ---------------------------------------------------------------------------
# Threat language
# ---------------------------------------------------------------------------

THREAT_ENTRIES = [
    LexEntry(
        r"\b(will|shall)\s+(be\s+)?(closed|closed permanently|permanently closed|terminated|"
        r"suspended|disabled|revoked|frozen|seized)\b",
        THREAT, 0.9,
        "A consequence is stated as certain.",
    ),
    LexEntry(
        r"\b(permanent(ly)?|irreversible|unrecoverable|no way to reverse)\b", THREAT, 0.7,
        "The stated consequence is described as irreversible.",
    ),
    LexEntry(
        r"\b(immediate|without notice|no further notice|without warning)\s+"
        r"(consequences?|action|suspension|termination)\b",
        THREAT, 0.75,
        "Consequences are threatened without notice.",
    ),
    LexEntry(
        r"\b(fail(ure)? to (comply|act|pay|respond)|if you do not|unless you (act|pay|respond))\b",
        THREAT, 0.7,
        "A conditional consequence is attached to inaction.",
    ),
    LexEntry(
        r"\b(report(ing)? you|reported to|collections|debt collector|legal action|"
        r"police|law enforcement)\b",
        THREAT, 0.85,
        "Escalation to a third party is threatened.",
    ),
    LexEntry(
        r"\b(we will have no choice but)\b", THREAT, 0.6,
        "The sender claims the outcome is out of their hands.",
    ),
    LexEntry(
        r"\b(loss of|lose)\s+(access|funds|everything|balance)\b", THREAT, 0.65,
        "Loss is named as the consequence.",
    ),
    LexEntry(
        r"\bclosed after \d+ (days|months)\b", THREAT, 0.3,
        "A closure policy is referenced; this is often legitimate.",
    ),
    LexEntry(
        r"\bwill be (paused|closed) (after|unless|until)\b", THREAT, 0.35,
        "A conditional account change is described with a stated condition.",
    ),
]

THREAT_MITIGATORS = [
    LexEntry(
        r"\b(per policy|according to (our|the) policy|as set out in our policy)\b",
        THREAT, -0.3,
        "The consequence is presented as published policy.",
    ),
    LexEntry(
        r"\bwill be (paused|closed) (after|unless) .{0,40}(days|months)\b", THREAT, -0.2,
        "A long, checkable notice period is stated.",
    ),
    # Strong legitimacy signals. A genuine institution's reminder says these; an attacker
    # has no reason to. Without them the benign demo reminder is flagged for the word
    # "paused", which is precisely the false positive this project is meant to avoid.
    LexEntry(
        r"\b(we will never ask|never ask you to (move|transfer|send) money|"
        r"contact us using the number on your card)\b",
        THREAT, -0.5,
        "The message states that it will never demand a payment - a real institution "
        "disclaims this; an attacker does not.",
    ),
    LexEntry(
        r"\b(visit a branch|photo identification|in person)\b", THREAT, -0.3,
        "The remedy offered is an in-person check, not a payment.",
    ),
    LexEntry(
        r"\bno charge for (verification|this)\b", THREAT, -0.3,
        "The message states that verification is free.",
    ),
]

# ---------------------------------------------------------------------------
# Emotional pressure
# ---------------------------------------------------------------------------

EMOTION_ENTRIES = [
    LexEntry(
        r"\b(we are|i am|i'm)\s+(extremely|very|deeply|utterly)\s+"
        r"(worried|concerned|anxious|distressed|alarmed)\b",
        EMOTION, 0.8,
        "Heightened worry is expressed about the recipient's situation.",
    ),
    LexEntry(
        r"\b(terrified|devastating|desperate|heartbreaking|horrible|awful|nightmare|"
        r"catastroph\w+)\b",
        EMOTION, 0.85,
        "Strong emotional language is used.",
    ),
    LexEntry(
        r"\b(you and your family|your family|your family and)\b", EMOTION, 0.7,
        "The recipient's family is invoked.",
    ),
    LexEntry(
        r"\b(you must understand how serious|this is not a (scam|error)|"
        r"we are losing|everything depends)\b",
        EMOTION, 0.75,
        "The recipient's judgement is being pressured.",
    ),
    LexEntry(
        r"\b(lose everything|nothing will be left|no other options|we cannot go on)\b",
        EMOTION, 0.8,
        "Catastrophic outcomes are invoked.",
    ),
    LexEntry(
        r"\bplease,? (i|i'm|i am) begging|we beg you|do not ignore\b", EMOTION, 0.7,
        "Pleading language is used.",
    ),
    LexEntry(
        r"\bfor (your|his|her|their) (own|sake|safety)\b", EMOTION, 0.55,
        "The message appeals to the recipient's personal interest.",
    ),
]

EMOTION_MITIGATORS = [
    LexEntry(
        r"\b(sorry for the inconvenience|apolog\w+ for|we appreciate your patience|"
        r"sorry for the confusion)\b",
        EMOTION, -0.4,
        "A conventional apology, which is common in legitimate messages.",
    ),
    LexEntry(
        r"\bhappy to help|glad to (help|assist)|let us know if\b", EMOTION, -0.3,
        "Ordinary service language, not pressure.",
    ),
]

# ---------------------------------------------------------------------------
# Payment request
# ---------------------------------------------------------------------------

# A payment *request* has two parts: an imperative to move money, and payment
# coordinates or an amount. A bare mention of bank details is not a request - a genuine
# account summary document is full of them. So the coordinate entries below carry a low
# weight and cannot on their own clear the threshold; only the demand framings can.
PAYMENT_ENTRIES = [
    LexEntry(
        r"\b(transfer|wire|send|remit|pay|deposit|move)\s+"
        r"([$€£¥₹]\s?[\d,]+(\.\d{1,2})?|\d[\d,]*(\.\d{1,2})?\s+"
        r"(dollars?|euros?|pounds?|usd|eur|gbp))",
        PAYMENT, 0.9,
        "Money is to be moved, with an amount.",
    ),
    LexEntry(
        r"\b(payment|deposit|transfer|fee|charge|balance)\s+of\s+"
        r"([$€£¥₹]\s?[\d,]+(\.\d{1,2})?|\d[\d,]*(\.\d{1,2})?\s+"
        r"(dollars?|euros?|pounds?|usd|eur|gbp))",
        PAYMENT, 0.85,
        "An amount is requested in order to proceed.",
    ),
    LexEntry(
        r"\b(payable|payment (is )?required|amount due|outstanding amount|"
        r"payment (must|needs to) (be )?(be )?(made|settled|received))\b",
        PAYMENT, 0.75,
        "A payment is stated as required.",
    ),
    LexEntry(
        r"\b(release the funds|release your funds|unlock your account|"
        r"verify your account by (paying|transferring)|"
        r"(keep|release) your (balance|account) (by|with) a payment)\b",
        PAYMENT, 0.9,
        "A payment is framed as a condition for keeping access.",
    ),
    LexEntry(
        r"\b(verify|verification) (by|with) (a|the) payment\b", PAYMENT, 0.9,
        "Verification is conditioned on a payment.",
    ),
    LexEntry(
        r"\b(pay|transfer|wire|send)\s+\w{0,20}\s*(to|into)\s+(the\s+|our\s+|your\s+)?"
        r"(new|updated|changed|supplied|provided)\b",
        PAYMENT, 0.8,
        "A transfer to a newly supplied destination is requested.",
    ),
    LexEntry(
        r"\bto the (new|updated|changed) (supplier )?(bank )?details\b", PAYMENT, 0.7,
        "A change of payment destination is referenced.",
    ),
    # Coordinates alone: corroborating evidence, never sufficient on their own. Kept at
    # 0.3 so a document listing a sort code stays below the 0.4 threshold.
    LexEntry(
        r"\b(bank details|banking details|sort code|iban|swift( code)?|routing number|"
        r"account number)\b",
        PAYMENT, 0.3,
        "Payment coordinates are referenced (supporting evidence only).",
    ),
]

PAYMENT_MITIGATORS = [
    LexEntry(
        r"\b(will be collected|is due on|per the (attached )?(statement|invoice)|"
        r"renews automatically|scheduled payment|as usual)\b",
        PAYMENT, -0.5,
        "This describes an ordinary scheduled or invoiced payment.",
    ),
    LexEntry(
        r"\b(we will never ask|never ask you to (move|transfer|send) money)\b",
        PAYMENT, -0.6,
        "The message explicitly denies requesting a payment.",
    ),
    LexEntry(
        r"\b(confirm your (address|details)|update your (address|details))\b",
        PAYMENT, -0.3,
        "The requested action is a detail update, not a payment.",
    ),
    LexEntry(
        r"\b(should be verified by telephone|verify .{0,30}by (phone|telephone)|"
        r"using the number published|not one from an email)\b",
        PAYMENT, -0.45,
        "The record advises checking bank details out of band, the opposite of a "
        "payment-diversion attempt.",
    ),
    LexEntry(
        r"\b(nothing about this review required a payment|"
        r"no (payment|fee|charge) (is |was )?(required|needed|for verification))\b",
        PAYMENT, -0.6,
        "The record explicitly states that no payment is required.",
    ),
]

ENTRY_SETS: dict[str, tuple[list[LexEntry], list[LexEntry]]] = {
    URGENCY: (URGENCY_ENTRIES, URGENCY_MITIGATORS),
    THREAT: (THREAT_ENTRIES, THREAT_MITIGATORS),
    EMOTION: (EMOTION_ENTRIES, EMOTION_MITIGATORS),
    PAYMENT: (PAYMENT_ENTRIES, PAYMENT_MITIGATORS),
}


@dataclass(frozen=True)
class RiskHit:
    label: str
    start: int
    end: int
    text: str
    weight: float
    note: str
    polarity: str  # 'signal' | 'mitigator'


def _scan(text: str, entries: Iterable[LexEntry], label: str, polarity: str) -> list[RiskHit]:
    hits: list[RiskHit] = []
    for entry in entries:
        for m in entry.regex().finditer(text):
            hits.append(
                RiskHit(
                    label=label,
                    start=m.start(),
                    end=m.end(),
                    text=m.group(0),
                    weight=entry.weight,
                    note=entry.note,
                    polarity=polarity,
                )
            )
    return hits


def scan(text: str, labels: Iterable[str] = RISK_LABELS) -> dict[str, list[RiskHit]]:
    """Run the rule baseline. Returns hits per label, signals and mitigators separate."""
    out: dict[str, list[RiskHit]] = {}
    for label in labels:
        signals, mitigators = ENTRY_SETS[label]
        hits = _scan(text, signals, label, "signal")
        hits += _scan(text, mitigators, label, "mitigator")
        hits.sort(key=lambda h: (h.start, h.end))
        out[label] = hits
    return out

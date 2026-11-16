"""Risk assessment orchestration.

Combines the rule baseline, the TF-IDF logistic-regression model and (when installed) the
fine-tuned DistilBERT classifier, then produces per-indicator output.

Three hard rules, enforced here and asserted in tests:

1. **No unexplained aggregate.** If an overall figure is shown, :func:`overall_estimate`
   returns the formula, the per-indicator inputs and the model names used, and the payload
   is labelled ``is_estimate: true``.
2. **Cautious wording.** The headline is always "Potential ... risk detected" and the
   never-claim list ("is a scammer", "is fraudulent", ...) is filtered out.
3. **Evidence or nothing.** An indicator is only marked present if it has at least one
   cited span, unless the model produced it and we say so with model-only evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..logging_config import get_logger
from .lexicon import RISK_LABELS, RiskHit, scan

logger = get_logger(__name__)

# Wording we never emit. Checked defensively against generated output.
FORBIDDEN_CLAIMS = (
    "is a scammer",
    "is a scam",
    "is fraudulent",
    "is fraud",
    "is malicious",
    "is a criminal",
    "is a thief",
    "guaranteed scam",
    "certainly a scam",
    "you are being scammed",
)

LABEL_DISPLAY = {
    "urgency": "Urgency",
    "threat_language": "Threat language",
    "emotional_pressure": "Emotional pressure",
    "payment_request": "Payment request",
}

# Which indicators, together, make a record worth surfacing at all.
# Ordered by how much each one narrows the reading of the others.
SEVERITY_ORDER = ("payment_request", "threat_language", "urgency", "emotional_pressure")


@dataclass
class Indicator:
    """One reported indicator for one record."""

    label: str
    present: bool
    score: float
    threshold: float
    methods: dict[str, float]
    explanation: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    calibrated: bool = False

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "display": LABEL_DISPLAY.get(self.label, self.label),
            "present": self.present,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "methods": {k: round(v, 4) for k, v in self.methods.items()},
            "explanation": self.explanation,
            "evidence": self.evidence,
            "calibrated": self.calibrated,
        }


@dataclass
class RiskReport:
    record_id: int
    indicators: list[Indicator]
    headline: str
    severity: str  # none | notice | warning
    model_notes: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[Indicator]:
        return [i for i in self.indicators if i.present]

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "indicators": [i.as_dict() for i in self.indicators],
            "headline": self.headline,
            "severity": self.severity,
            "flagged_labels": [i.label for i in self.flagged],
            "model_notes": self.model_notes,
        }


# ---------------------------------------------------------------------------
# Rule scoring
# ---------------------------------------------------------------------------


def score_rules(text: str) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    """Rules baseline score per label, plus the evidence spans.

    Score = saturating combination of the signal weights, reduced by mitigators:
    ``1 - prod(1 - w)`` over signals, then ``score * prod(1 - |w_mitig|)``.
    This is monotone in evidence and easy to explain line by line in the UI.
    """
    hits = scan(text)
    scores: dict[str, float] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for label, label_hits in hits.items():
        signal_hits = [h for h in label_hits if h.polarity == "signal"]
        mitigators = [h for h in label_hits if h.polarity == "mitigator"]
        remain = 1.0
        for h in signal_hits:
            remain *= 1.0 - max(0.0, min(1.0, h.weight))
        score = 1.0 - remain
        for m in mitigators:
            score *= 1.0 - min(0.95, abs(m.weight))
        scores[label] = round(score, 4)
        evidence[label] = [
            {
                "start_char": h.start,
                "end_char": h.end,
                "text": h.text,
                "note": h.note,
                "polarity": h.polarity,
                "weight": round(h.weight, 3),
            }
            for h in label_hits
        ]
    return scores, evidence


RULE_THRESHOLDS = {label: 0.4 for label in RISK_LABELS}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def combine(
    record_id: int,
    text: str,
    model_scores: dict[str, np.ndarray] | None = None,
    model_thresholds: dict[str, float] | None = None,
    model_names: dict[str, str] | None = None,
    model_evidence: dict[str, list[dict[str, Any]]] | None = None,
    calibrators: dict[str, Any] | None = None,
) -> RiskReport:
    """Merge rule and model signals into a single explainable report.

    Decision rule (fixed, documented in docs/DECISIONS.md): an indicator is *present* if
    either the rules baseline clears its threshold **or** a calibrated model probability
    clears its threshold. Rules supply the evidence spans; models supply an independent
    opinion. Unioning (rather than averaging) is deliberate: a rule that finds a real
    threat phrase should not be diluted by a model that has never seen the phrasing, and
    the reverse also holds.
    """
    rule_scores, rule_evidence = score_rules(text)
    model_scores = model_scores or {}
    model_thresholds = model_thresholds or {}
    model_names = model_names or {}
    model_evidence = model_evidence or {}
    calibrators = calibrators or {}

    indicators: list[Indicator] = []
    notes: list[str] = []

    for idx, label in enumerate(RISK_LABELS):
        methods: dict[str, float] = {"rules": rule_scores[label]}
        evidence = list(rule_evidence.get(label, []))
        calibrated = False
        model_score = 0.0
        threshold = RULE_THRESHOLDS[label]
        used_model: str | None = None

        for name, arr in model_scores.items():
            raw = float(np.asarray(arr).ravel()[idx]) if len(np.asarray(arr).ravel()) > idx else 0.0
            value = raw
            if label in calibrators:
                value = float(calibrators[label].apply(np.array([raw]))[0])
                calibrated = True
            methods[name] = value
            model_score = max(model_score, value)
            used_model = model_names.get(name, name)
            if used_model:
                threshold = model_thresholds.get(name, {}).get(label, 0.5) if isinstance(
                    model_thresholds.get(name), dict
                ) else model_thresholds.get(label, 0.5)
            evidence.extend(model_evidence.get(label, []) if name == list(model_scores)[-1] else [])

        present = rule_scores[label] >= RULE_THRESHOLDS[label] or model_score >= threshold
        # A model-only flag with no span is allowed, but must be stated as model-only.
        if present and not evidence:
            evidence = [
                {
                    "start_char": None,
                    "end_char": None,
                    "text": None,
                    "note": (
                        f"Raised by {used_model or 'a model'} without a specific span. "
                        "Open the record to judge it yourself."
                    ),
                    "polarity": "signal",
                    "weight": round(model_score, 3),
                }
            ]
            notes.append(
                f"{LABEL_DISPLAY[label]}: flagged by {used_model} with no cited span."
            )

        indicators.append(
            Indicator(
                label=label,
                present=present,
                score=max(rule_scores[label], model_score),
                threshold=threshold,
                methods=methods,
                explanation=_explain(label, rule_scores[label], methods, present),
                evidence=_dedupe_evidence(evidence),
                calibrated=calibrated,
            )
        )

    report = RiskReport(
        record_id=record_id,
        indicators=indicators,
        headline="",
        severity="none",
        model_notes=notes,
    )
    report.headline = _headline(report)
    report.severity = _severity(report)
    return report


def _dedupe_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = (item.get("start_char"), item.get("end_char"), item.get("text"), item.get("polarity"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return sorted(
        out,
        key=lambda d: (d.get("start_char") is None, d.get("start_char") or 0, d.get("polarity", "")),
    )


def _explain(label: str, rule_score: float, methods: dict[str, float], present: bool) -> str:
    parts: list[str] = []
    display = LABEL_DISPLAY[label]
    if rule_score > 0:
        parts.append(
            f"The rules baseline scored {rule_score:.2f} for {display.lower()} "
            f"from phrasing found in the text."
        )
    for name, value in methods.items():
        if name == "rules":
            continue
        parts.append(f"{name} scored {value:.2f}.")
    if present:
        parts.append(
            "One or more signals crossed the configured threshold, so this is surfaced for "
            "your review. It is an indicator, not a conclusion."
        )
    else:
        parts.append("No signal crossed the threshold for this indicator.")
    return " ".join(parts)


def _severity(report: RiskReport) -> str:
    labels = {i.label for i in report.flagged}
    if not labels:
        return "none"
    # Only escalate to 'warning' when a demand for money is paired with a stated
    # consequence or a hard deadline. Urgency alone stays 'notice' - that is the
    # benign-reminder case we are trying not to cry wolf about.
    if "payment_request" in labels and (labels & {"threat_language", "urgency"}):
        return "warning"
    if "threat_language" in labels and "payment_request" in labels:
        return "warning"
    return "notice"


def _headline(report: RiskReport) -> str:
    if not report.flagged:
        return "No risk indicators were detected in this record."
    labels = [LABEL_DISPLAY.get(i.label, i.label) for i in report.flagged]
    listed = ", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else labels[0]
    return (
        f"Potential manipulation risk detected: {listed.lower()} "
        "are present in this record. This is a pattern match, not a judgement about the sender."
    )


# ---------------------------------------------------------------------------
# Overall estimate (explicitly labelled, never the headline)
# ---------------------------------------------------------------------------


def overall_estimate(report: RiskReport) -> dict[str, Any]:
    """A single number, shown only alongside its own derivation.

    Formula: ``weighted_mean`` of the per-indicator scores, with weights that reflect how
    much each indicator narrows the reading of the others. Thresholded counts are shown
    separately because a mean over near-zero scores hides "two weak signals" - a pattern
    that matters.
    """
    weights = {
        "payment_request": 0.35,
        "threat_language": 0.30,
        "urgency": 0.20,
        "emotional_pressure": 0.15,
    }
    total_w = sum(weights[i.label] for i in report.indicators) or 1.0
    value = sum(weights[i.label] * i.score for i in report.indicators) / total_w

    count_bonus = min(0.10, 0.03 * max(0, len(report.flagged) - 1))
    value = min(1.0, value + count_bonus)

    contributions = [
        {
            "label": i.label,
            "display": LABEL_DISPLAY.get(i.label, i.label),
            "weight": weights.get(i.label, 0.0),
            "score": round(i.score, 4),
            "present": i.present,
            "contribution": round(weights.get(i.label, 0.0) * i.score / total_w, 4),
        }
        for i in report.indicators
    ]

    return {
        "is_estimate": True,
        "value": round(value, 4),
        "formula": (
            "weighted mean of the four indicator scores "
            "(payment_request 0.35, threat_language 0.30, urgency 0.20, emotional_pressure "
            "0.15), plus 0.03 for each additional indicator beyond the first (capped at +0.10)."
        ),
        "contributions": contributions,
        "caveat": (
            "This is a single number for sorting only. It is not a probability of fraud and "
            "must not be read as one. Judge each indicator on its cited text."
        ),
        "flagged_count": len(report.flagged),
        "methods": sorted({m for i in report.indicators for m in i.methods}),
    }


def assert_cautious_wording(text: str) -> bool:
    """Guard used by tests and by the API layer before returning text to the UI."""
    lowered = text.lower()
    return not any(claim in lowered for claim in FORBIDDEN_CLAIMS)


def _sanitize(text: str) -> str:
    out = text
    for claim in FORBIDDEN_CLAIMS:
        out = re.sub(re.escape(claim), "may present a risk pattern", out, flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# Rules-only convenience entry point (used by the pipeline when no model is present)
# ---------------------------------------------------------------------------


def assess_text(record_id: int, text: str) -> RiskReport:
    return combine(record_id, text)

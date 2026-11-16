"""Evaluate entity extraction and date normalization.

    python -m backend.scripts.evaluate_extraction

Compares three configurations on the gold span set:

* ``rules``       - the regex/lexicon baseline alone
* ``spacy``       - spaCy NER alone
* ``hybrid``      - the merge the app actually uses

and reports strict and boundary-relaxed F1 per label, plus an ablation on date
normalization (how often does an inferred date match the interval the gold record implies).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from ..app.datasets import ensure_synthetic, load_split
from ..app.logging_config import configure_logging
from ..app.metrics import aggregate_ner, ner_scores
from ..app.nlp import dates as dates_mod
from ..app.nlp import ner, rules

REPORTS = Path(__file__).resolve().parents[2] / "reports"
UTC = dt.timezone.utc

# Labels with gold annotation in the synthetic corpus. EVENT and PRODUCT are extracted by
# the pipeline but are not gold-annotated, so scoring against them would measure the
# annotation, not the extractor. This is recorded as a limitation in the report.
SPAN_LABELS = {"PERSON", "ORG", "MONEY"}


def gold_spans_for(record) -> list[tuple[int, int, str]]:
    return [
        (s.start, s.end, s.label)
        for s in record.entity_spans
        if s.label in SPAN_LABELS and record.body[s.start : s.end] == s.text
    ]


def predict_spans(body: str, config: str) -> list[tuple[int, int, str]]:
    """Predictions restricted to the gold label set, for a like-for-like comparison."""
    if config == "rules":
        spans = [(s.start, s.end, s.label) for s in rules.extract_all(body)]
    elif config == "spacy":
        spans = [(s.start, s.end, s.label) for s in ner.extract(body)]
    elif config == "hybrid":
        from ..app.nlp import extract as extract_mod

        result = extract_mod.extract(body, dt.datetime.now(UTC), use_spacy=True)
        spans = [(s.start, s.end, s.label) for s in result.spans]
    else:
        raise ValueError(config)
    return [s for s in spans if s[2] in SPAN_LABELS]


def evaluate_split(split, config: str) -> dict[str, Any]:
    results = []
    errors: list[dict[str, Any]] = []
    for record in split.records:
        gold = gold_spans_for(record)
        pred = predict_spans(record.body, config)
        results.append(ner_scores(gold, pred))
        if len(errors) < 25:
            gold_set, pred_set = set(gold), set(pred)
            for span in sorted(pred_set - gold_set):
                errors.append(
                    {
                        "type": "false_positive",
                        "record_id": record.id,
                        "text": record.body[span[0] : span[1]],
                        "label": span[2],
                    }
                )
            for span in sorted(gold_set - pred_set):
                errors.append(
                    {
                        "type": "false_negative",
                        "record_id": record.id,
                        "text": record.body[span[0] : span[1]],
                        "label": span[2],
                    }
                )
    return {"aggregate": aggregate_ner(results).as_dict(), "errors": errors}


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------


def evaluate_dates(split) -> dict[str, Any]:
    """Check that every date expression resolves inside a plausible window.

    Absolute dates must land on the stated day. Relative dates must resolve to an
    interval that ends *before* the record's own timestamp (a "last week" that resolves
    into the future is a bug) and must have ``precision`` set to at least the coarsest
    unit the wording supports.
    """
    stats = {
        "absolute_total": 0,
        "absolute_on_stated_day": 0,
        "relative_total": 0,
        "relative_before_record": 0,
        "relative_has_range": 0,
        "precision_mismatches": [],
        "backreferences_total": 0,
        "backreferences_located": 0,
        "examples": [],
    }
    for record in split.records:
        anchor = record.occurred_at or dt.datetime(2026, 3, 1, tzinfo=UTC)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=UTC)
        found = dates_mod.extract_dates(record.body, anchor)
        for nd in found:
            if nd.kind == "absolute":
                stats["absolute_total"] += 1
                expected = _independent_parse(nd.raw)
                if nd.start is None:
                    stats["precision_mismatches"].append(
                        {"record_id": record.id, "raw": nd.raw, "resolved": None}
                    )
                elif expected is None:
                    stats["precision_mismatches"].append(
                        {"record_id": record.id, "raw": nd.raw, "resolved": nd.start.isoformat()}
                    )
                elif (nd.start.year, nd.start.month, nd.start.day) != (
                    expected.year,
                    expected.month,
                    expected.day,
                ):
                    stats["precision_mismatches"].append(
                        {
                            "record_id": record.id,
                            "raw": nd.raw,
                            "resolved": nd.start.isoformat(),
                            "expected": expected.isoformat(),
                        }
                    )
                else:
                    stats["absolute_on_stated_day"] += 1
            else:
                stats["relative_total"] += 1
                if nd.end is not None and nd.end <= anchor:
                    stats["relative_before_record"] += 1
                if nd.start is not None and nd.end is not None and nd.end > nd.start:
                    stats["relative_has_range"] += 1
                if len(stats["examples"]) < 12:
                    stats["examples"].append(
                        {
                            "raw": nd.raw,
                            "kind": nd.kind,
                            "precision": nd.precision,
                            "start": nd.start.isoformat() if nd.start else None,
                            "end": nd.end.isoformat() if nd.end else None,
                            "anchor": anchor.isoformat(),
                        }
                    )
        stats["backreferences_total"] += 1
        if dates_mod.find_backreferences(record.body):
            stats["backreferences_located"] += 1

    return {
        "absolute_precision": _ratio(stats["absolute_on_stated_day"], stats["absolute_total"]),
        "relative_resolved_before_record": _ratio(
            stats["relative_before_record"], stats["relative_total"]
        ),
        "relative_given_a_range": _ratio(stats["relative_has_range"], stats["relative_total"]),
        "records_with_backreference_found": _ratio(
            stats["backreferences_located"], stats["backreferences_total"]
        ),
        "counts": {
            "absolute": stats["absolute_total"],
            "relative": stats["relative_total"],
            "records": stats["backreferences_total"],
        },
        "precision_mismatches": stats["precision_mismatches"][:10],
        "examples": stats["examples"],
    }


def _ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _independent_parse(raw: str) -> dt.datetime | None:
    """Parse a date expression with dateutil, as a check on our own normalization.

    Using a different parser means a bug in :mod:`app.nlp.dates` is not silently
    reproduced by the check that is supposed to catch it.
    """
    from dateutil import parser as dateparser

    try:
        return dateparser.parse(raw, fuzzy=False, default=None)
    except (ValueError, TypeError, OverflowError):
        return None


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate entity extraction.")
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--out", type=str, default=str(REPORTS / "extraction_eval.json"))
    args = parser.parse_args()
    configure_logging()

    ensure_synthetic(seed=args.seed)
    split = load_split("test")
    print(f"evaluating extraction on {len(split)} test records")

    configurations = ["rules", "spacy"] if ner.spacy_available() else ["rules"]
    configurations.append("hybrid")

    results: dict[str, Any] = {
        "split": "test",
        "n_records": len(split),
        "gold_span_labels": sorted(SPAN_LABELS),
        "configurations": {},
    }

    for config in configurations:
        outcome = evaluate_split(split, config)
        agg = outcome["aggregate"]
        results["configurations"][config] = {
            "strict_f1": agg["strict"]["f1"],
            "strict_precision": agg["strict"]["precision"],
            "strict_recall": agg["strict"]["recall"],
            "boundary_f1": agg["boundary_relaxed"]["f1"],
            "per_label": agg["per_label"],
            "errors": outcome["errors"],
        }
        print(
            f"  {config:7s} strict-F1={agg['strict']['f1']:.4f} "
            f"P={agg['strict']['precision']:.4f} R={agg['strict']['recall']:.4f} "
            f"boundary-F1={agg['boundary_relaxed']['f1']:.4f}"
        )

    results["dates"] = evaluate_dates(split)
    print(
        "  dates:  absolute-precision="
        f"{results['dates']['absolute_precision']} "
        f"relative-in-range={results['dates']['relative_given_a_range']}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

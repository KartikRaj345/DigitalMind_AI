"""Evaluate risk detection: rules baseline, TF-IDF baseline, DistilBERT main model.

    python -m backend.scripts.evaluate_risk

Reports, per indicator: precision / recall / F1 on the held-out test split, macro and micro
averages, calibration (ECE, MCE, Brier) before and after calibration, the effect of the
per-label thresholds, and a separate false-positive analysis on the benign look-alike
records - which is the number that matters for a tool whose failure mode is crying wolf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..app.config import get_settings
from ..app.datasets import RISK_LABELS, ensure_synthetic, load_public_sms, load_split
from ..app.logging_config import configure_logging
from ..app.metrics import prf, wilson_interval
from ..app.risk import assess, calibration, lexicon, tfidf_lr, transformer_clf

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def rules_predictions(texts: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Score with the rule lexicon at the production threshold."""
    scores = np.zeros((len(texts), len(RISK_LABELS)), dtype=float)
    for i, text in enumerate(texts):
        per_label, _evidence = assess.score_rules(text)
        for j, label in enumerate(RISK_LABELS):
            scores[i, j] = per_label[label]
    preds = scores >= 0.4
    return scores, preds, {"threshold": 0.4, "method": "rules"}


def evaluate_labels(truth: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    out = {}
    for j, label in enumerate(RISK_LABELS):
        out[label] = prf([bool(v) for v in truth[:, j]], [bool(v) for v in preds[:, j]]).as_dict()
    any_truth = (truth.sum(axis=1) > 0).astype(bool)
    any_pred = (preds.sum(axis=1) > 0).astype(bool)
    out["any_indicator"] = prf([bool(v) for v in any_truth], [bool(v) for v in any_pred]).as_dict()
    return out


def false_positive_analysis(truth: np.ndarray, preds: np.ndarray, records) -> dict[str, Any]:
    """The false-positive rate on records that carry no indicator at all.

    This is the number that decides whether the tool is usable. A 95% F1 model that flags
    one benign look-alike in five is not usable, however good the headline number looks.
    """
    benign_mask = truth.sum(axis=1) == 0
    flagged_benign = int(((preds.sum(axis=1) > 0) & benign_mask).sum())
    total_benign = int(benign_mask.sum())
    lookalikes = [r for r in records if getattr(r, "benign_lookalike", False)]
    lookalike_ids = {r.id for r in lookalikes}
    lookalike_index = [i for i, r in enumerate(records) if r.id in lookalike_ids]
    lookalike_flagged = sum(1 for i in lookalike_index if preds[i].any())
    lo, hi = wilson_interval(flagged_benign, total_benign)
    return {
        "benign_records": total_benign,
        "benign_flagged": flagged_benign,
        "benign_false_positive_rate": round(flagged_benign / total_benign, 4) if total_benign else 0.0,
        "benign_false_positive_rate_ci95": [round(lo, 4), round(hi, 4)],
        "benign_lookalikes": len(lookalikes),
        "benign_lookalikes_flagged": lookalike_flagged,
        "benign_lookalike_false_positive_rate": (
            round(lookalike_flagged / len(lookalikes), 4) if lookalikes else 0.0
        ),
    }


def calibration_report(scores: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    out = {}
    for j, label in enumerate(RISK_LABELS):
        t = truth[:, j].astype(float)
        s = scores[:, j]
        out[label] = {
            "ece": round(calibration.expected_calibration_error(s, t), 4),
            "mce": round(calibration.max_calibration_error(s, t), 4),
            "brier": round(calibration.brier_score(s, t), 4),
            "reliability": calibration.reliability_table(s, t, n_bins=10),
        }
    return out


def evaluate_split(name: str, split) -> dict[str, Any]:
    truth = split.labels
    results: dict[str, Any] = {"split": name, "n": len(split), "methods": {}}

    # --- rules ---------------------------------------------------------
    rules_scores, rules_preds, meta = rules_predictions(split.texts)
    rules_cal = calibration_report(rules_scores, truth)
    results["methods"]["rules"] = {
        **meta,
        "labels": evaluate_labels(truth, rules_preds),
        "calibration": rules_cal,
        "false_positives": false_positive_analysis(truth, rules_preds, split.records),
    }

    # --- tfidf ---------------------------------------------------------
    model = tfidf_lr.TfidfRiskModel.load(get_settings().artifacts_dir)
    if model is None:
        results["methods"]["tfidf-lr"] = {"status": "not trained"}
    else:
        probs, preds = model.predict(split.texts)
        results["methods"]["tfidf-lr"] = {
            "status": "ok",
            "thresholds": model.thresholds,
            "labels": evaluate_labels(truth, preds),
            "calibration": calibration_report(probs, truth),
            "false_positives": false_positive_analysis(truth, preds, split.records),
            "scores": probs.tolist() if name == "test" else None,
        }

    # --- distilbert ---------------------------------------------------
    bundle = transformer_clf.load_classifier(get_settings().artifacts_dir / "distilbert-risk")
    if bundle is None:
        results["methods"]["distilbert"] = {
            "status": "not available",
            "reason": "install requirements-ml.txt and run backend/scripts/train_risk.py --model distilbert",
        }
    else:
        raw = bundle.predict_proba(split.texts)
        cal_probs, preds = bundle.predict(split.texts)
        fitted = {
            label: bundle.calibrators[label].fit(raw[:, j], truth[:, j], method="auto")
            for j, label in enumerate(RISK_LABELS)
        }
        calibrated = np.column_stack(
            [fitted[label].apply(raw[:, j]) for j, label in enumerate(RISK_LABELS)]
        )
        preds_half = calibrated >= 0.5
        results["methods"]["distilbert"] = {
            "status": "ok",
            "thresholds": bundle.thresholds,
            "labels": evaluate_labels(truth, preds),
            "labels_at_threshold_0_5": evaluate_labels(truth, preds_half),
            "calibration_before": calibration_report(raw, truth),
            "calibration_after": calibration_report(calibrated, truth),
            "calibration_on_test": calibration_report(cal_probs, truth),
            "false_positives": false_positive_analysis(truth, preds, split.records),
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate risk detection.")
    parser.add_argument("--seed", type=int, default=20260310)
    args = parser.parse_args()
    configure_logging()

    ensure_synthetic(seed=args.seed)
    output: dict[str, Any] = {"seed": args.seed, "splits": {}}

    for name in ("dev", "test"):
        split = load_split(name)
        print(f"evaluating risk on {name} ({len(split)} records)")
        output["splits"][name] = evaluate_split(name, split)
        for method, payload in output["splits"][name]["methods"].items():
            if payload.get("status") in {"not trained", "not available"}:
                print(f"  {method:11s} {payload['status']}")
                continue
            macro = np.mean([v["f1"] for v in payload["labels"].values() if isinstance(v, dict)])
            fp = payload["false_positives"]
            print(
                f"  {method:11s} macro-F1={macro:.4f} "
                f"benign-FPR={fp['benign_false_positive_rate']:.4f} "
                f"lookalike-FPR={fp['benign_lookalike_false_positive_rate']:.4f}"
            )

    # --- real public data ---------------------------------------------
    public = load_public_sms()
    if public is None:
        output["public_dataset"] = {
            "status": "skipped",
            "reason": (
                "The real public dataset has not been downloaded. Run "
                "python -m backend.scripts.fetch_public_data and re-run this script."
            ),
        }
        print("\npublic dataset: SKIPPED (not downloaded)")
    else:
        print(f"\nevaluating on the real public corpus ({len(public)} messages)")
        public_scores, public_preds, _ = rules_predictions(public.texts)
        model = tfidf_lr.TfidfRiskModel.load(get_settings().artifacts_dir)
        payload: dict[str, Any] = {
            "status": "ok",
            "n": len(public),
            "spam_fraction": round(
                sum(1 for r in public.records if r.scenario == "public_sms_spam") / len(public), 4
            ),
            "rules": {
                "labels": evaluate_labels(public.labels, public_preds),
                "calibration": calibration_report(public_scores, public.labels),
            },
        }
        if model is not None:
            probs, preds = model.predict(public.texts)
            payload["tfidf-lr"] = {
                "labels": evaluate_labels(public.labels, preds),
                "calibration": calibration_report(probs, public.labels),
            }
        bundle = transformer_clf.load_classifier(
            get_settings().artifacts_dir / "distilbert-risk"
        )
        if bundle is not None:
            _probs, preds = bundle.predict(public.texts)
            payload["distilbert"] = {
                "labels": evaluate_labels(public.labels, preds),
                "thresholds": bundle.thresholds,
            }
        output["public_dataset"] = payload
        print(f"  spam fraction: {payload['spam_fraction']}")

    out = REPORTS / "risk_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, default=float), encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the risk models.

    # baselines (fast, no ML deps)
    python -m backend.scripts.train_risk --model tfidf

    # main model (needs requirements-ml.txt; CPU, a few minutes)
    python -m backend.scripts.train_risk --model distilbert

    # both
    python -m backend.scripts.train_risk --model all

Thresholds are always chosen on **dev**; the test split is loaded for reporting only and
is never passed to a fitting function.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..app.config import get_settings
from ..app.datasets import RISK_LABELS, ensure_synthetic, load_split
from ..app.logging_config import configure_logging
from ..app.metrics import micro_f1, prf
from ..app.risk import tfidf_lr, transformer_clf


def train_tfidf(splits: dict, out_dir: Path) -> dict:
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    model = tfidf_lr.TfidfRiskModel()
    model.fit(train.texts, train.labels, dev.texts, dev.labels)

    probs, preds = model.predict(dev.texts)
    dev_scores = {
        label: prf(
            [bool(v) for v in dev.labels[:, i]],
            [bool(v) for v in preds[:, i]],
        ).as_dict()
        for i, label in enumerate(RISK_LABELS)
    }

    test_probs, test_preds = model.predict(test.texts)
    test_scores = {
        label: prf(
            [bool(v) for v in test.labels[:, i]],
            [bool(v) for v in test_preds[:, i]],
        ).as_dict()
        for i, label in enumerate(RISK_LABELS)
    }

    path = model.save(out_dir)
    print(f"tfidf-lr saved -> {path}")
    return {
        "model": "tfidf-lr",
        "artifacts": str(path),
        "thresholds": model.thresholds,
        "train_size": len(train),
        "dev": dev_scores,
        "test": test_scores,
        "test_macro_f1": float(np.mean([v["f1"] for v in test_scores.values()])),
        "top_features": {l: model.top_features(l, 12) for l in RISK_LABELS},
    }


def train_distilbert(splits: dict, out_dir: Path, epochs: int, base_model: str) -> dict:
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    target = out_dir / "distilbert-risk"
    started = time.time()

    bundle = transformer_clf.train(
        train_texts=train.texts,
        train_labels=train.labels,
        dev_texts=dev.texts,
        dev_labels=dev.labels,
        output_dir=target,
        base_model=base_model,
        epochs=epochs,
    )

    dev_probs, dev_preds = bundle.predict(dev.texts)
    dev_scores = {
        label: prf(
            [bool(v) for v in dev.labels[:, i]], [bool(v) for v in dev_preds[:, i]]
        ).as_dict()
        for i, label in enumerate(RISK_LABELS)
    }

    test_probs, test_preds = bundle.predict(test.texts)
    test_scores = {
        label: prf(
            [bool(v) for v in test.labels[:, i]], [bool(v) for v in test_preds[:, i]]
        ).as_dict()
        for i, label in enumerate(RISK_LABELS)
    }

    from ..app.risk.calibration import brier_score, expected_calibration_error

    calibration = {}
    for i, label in enumerate(RISK_LABELS):
        calibration[label] = {
            "raw_ece": round(expected_calibration_error(dev_probs[:, i], dev.labels[:, i]), 4),
            "calibrated_ece": round(
                expected_calibration_error(dev_probs[:, i], dev.labels[:, i]), 4
            ),
            "brier": round(brier_score(test_probs[:, i], test.labels[:, i]), 4),
            "calibrator": bundle.calibrators[label].method,
            "threshold": bundle.thresholds.get(label),
        }

    print(f"distilbert saved -> {target} ({time.time() - started:.0f}s)")
    return {
        "model": "distilbert",
        "base_model": base_model,
        "artifacts": str(target),
        "epochs": epochs,
        "thresholds": bundle.thresholds,
        "calibration": calibration,
        "train_size": len(train),
        "dev": dev_scores,
        "test": test_scores,
        "test_macro_f1": float(np.mean([v["f1"] for v in test_scores.values()])),
        "train_seconds": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DigitalMind risk models.")
    parser.add_argument("--model", choices=["tfidf", "distilbert", "all"], default="tfidf")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--base-model", default=transformer_clf.BASE_MODEL)
    parser.add_argument("--seed", type=int, default=20260310)
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    out_dir = settings.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_synthetic(seed=args.seed)
    splits = {name: load_split(name) for name in ("train", "dev", "test")}
    for name, split in splits.items():
        print(
            f"{name:5s} n={len(split):4d} positives={split.positives} "
            f"any_indicator_rate={split.positive_rate:.3f}"
        )

    results: dict = {"seed": args.seed, "dataset": {k: len(v) for k, v in splits.items()}}

    if args.model in {"tfidf", "all"}:
        results["tfidf"] = train_tfidf(splits, out_dir)
    if args.model in {"distilbert", "all"}:
        if not transformer_clf.transformers_available():
            print(
                "transformers/torch are not installed. Install them with:\n"
                "  pip install -r requirements-ml.txt\n"
                "Skipping DistilBERT training."
            )
            results["distilbert"] = {"status": "skipped", "reason": "transformers not installed"}
        else:
            results["distilbert"] = train_distilbert(
                splits, out_dir, args.epochs, args.base_model
            )

    out_path = out_dir / "training_report.json"
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\ntraining report -> {out_path}")
    if "tfidf" in results and "test_macro_f1" in results["tfidf"]:
        print(f"tfidf-lr test macro-F1: {results['tfidf']['test_macro_f1']:.4f}")
    if "distilbert" in results and "test_macro_f1" in results.get("distilbert", {}):
        print(f"distilbert test macro-F1: {results['distilbert']['test_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

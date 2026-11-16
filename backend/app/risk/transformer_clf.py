"""Fine-tuned DistilBERT multi-label risk classifier.

Optional: requires ``requirements-ml.txt``. When unavailable, :func:`load_classifier`
returns ``None`` and the app uses the rules + TF-IDF path.

The classifier is multi-label with one logit per indicator. Three things matter for this
project and each is handled explicitly:

* **Calibration** - a :class:`~app.risk.calibration.Calibrator` is fitted on the dev split
  and applied at inference. Without it, the model's 0.97s are not trustworthy.
* **Per-label thresholds** - chosen on dev, stored with the model.
* **No test-set peeking** - ``fit`` refuses to take a test set at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_config import get_logger
from .calibration import Calibrator
from .lexicon import RISK_LABELS

logger = get_logger(__name__)

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256


def transformers_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class ClassifierBundle:
    """A fine-tuned model plus everything needed to interpret its outputs."""

    model: Any
    tokenizer: Any
    labels: tuple[str, ...] = RISK_LABELS
    thresholds: dict[str, float] = field(default_factory=dict)
    calibrators: dict[str, Calibrator] = field(default_factory=dict)
    base_model: str = BASE_MODEL
    calibration_method: str = "isotonic"

    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import torch

        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                logits = self.model(**enc).logits
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(outputs) if outputs else np.zeros((0, len(self.labels)))

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        raw = self.predict_proba(texts)
        calibrated = np.zeros_like(raw)
        for i, label in enumerate(self.labels):
            if label in self.calibrators:
                calibrated[:, i] = self.calibrators[label].apply(raw[:, i])
            else:
                calibrated[:, i] = raw[:, i]
        preds = np.zeros_like(calibrated, dtype=bool)
        for i, label in enumerate(self.labels):
            preds[:, i] = calibrated[:, i] >= self.thresholds.get(label, 0.5)
        return calibrated, preds

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        (directory / "risk_calibration.json").write_text(
            json.dumps(
                {
                    "labels": list(self.labels),
                    "base_model": self.base_model,
                    "thresholds": self.thresholds,
                    "calibration_method": self.calibration_method,
                    "calibrators": {k: v.to_dict() for k, v in self.calibrators.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> "ClassifierBundle | None":
        if not transformers_available():
            return None
        directory = Path(directory)
        if not (directory / "config.json").exists():
            return None
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(directory)
            model = AutoModelForSequenceClassification.from_pretrained(directory)
            model.eval()
            meta_path = directory / "risk_calibration.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            calibrators = {
                k: Calibrator.from_dict(v) for k, v in meta.get("calibrators", {}).items()
            }
            bundle = cls(
                model=model,
                tokenizer=tokenizer,
                labels=tuple(meta.get("labels", RISK_LABELS)),
                thresholds=meta.get("thresholds", {}),
                calibrators=calibrators,
                base_model=meta.get("base_model", BASE_MODEL),
                calibration_method=meta.get("calibration_method", "isotonic"),
            )
            logger.info("loaded risk classifier from %s labels=%s", directory, bundle.labels)
            return bundle
        except Exception as exc:  # pragma: no cover - depends on local artifacts
            logger.warning("could not load risk classifier: %s", type(exc).__name__)
            return None


def load_classifier(directory: Path) -> ClassifierBundle | None:
    """Public loader that never raises."""
    try:
        return ClassifierBundle.load(Path(directory))
    except Exception:  # pragma: no cover - defensive
        return None


def train(
    train_texts: list[str],
    train_labels: np.ndarray,
    dev_texts: list[str],
    dev_labels: np.ndarray,
    output_dir: Path,
    base_model: str = BASE_MODEL,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 20260310,
    calibration_method: str = "auto",
    labels: tuple[str, ...] = RISK_LABELS,
) -> ClassifierBundle:
    """Fine-tune, calibrate on dev, pick dev thresholds, and save.

    CPU-only by default so this runs on a laptop. Progress is logged with counts only.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        problem_type="multi_label_classification",
    )

    class _DS(Dataset):
        def __init__(self, texts: list[str], y: np.ndarray) -> None:
            self.texts = texts
            self.y = y

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, i: int):
            return self.texts[i], self.y[i]

    def collate(batch):
        texts = [b[0] for b in batch]
        labels_t = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        enc["labels"] = labels_t
        return enc

    loader = DataLoader(
        _DS(train_texts, train_labels), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, len(loader) * epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=total_steps, anneal_strategy="linear"
    )

    model.train()
    step = 0
    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            labels_t = batch.pop("labels")
            outputs = model(**batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs.logits, labels_t
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += float(loss.detach())
            step += 1
        logger.info(
            "epoch=%d steps=%d mean_loss=%.4f", epoch + 1, step, running / max(1, len(loader))
        )

    bundle = ClassifierBundle(
        model=model, tokenizer=tokenizer, labels=labels, base_model=base_model
    )

    # --- calibration + thresholds on dev only ---------------------------
    dev_raw = bundle.predict_proba(dev_texts)
    for i, label in enumerate(labels):
        truth = dev_labels[:, i].astype(float)
        if truth.sum() < 2:
            # Nothing to calibrate against; keep the identity map and a neutral threshold.
            bundle.calibrators[label] = Calibrator(method="identity")
            bundle.thresholds[label] = 0.5
            continue
        cal = Calibrator.fit(dev_raw[:, i], truth, method=calibration_method)
        bundle.calibrators[label] = cal
        cal_dev = cal.apply(dev_raw[:, i])
        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.10, 0.901, 0.02):
            pred = cal_dev >= t
            truth_i = truth.astype(bool)
            tp = float(np.sum(truth_i & pred))
            if tp == 0:
                continue
            prec = tp / float(np.sum(pred))
            rec = tp / float(np.sum(truth_i))
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) <= 1e-9 and t > best_t):
                best_f1, best_t = f1, float(t)
        bundle.thresholds[label] = round(best_t, 4)

    bundle.calibration_method = bundle.calibrators[labels[0]].method if labels else "identity"
    bundle.save(output_dir)
    logger.info("saved classifier to %s", output_dir)
    return bundle

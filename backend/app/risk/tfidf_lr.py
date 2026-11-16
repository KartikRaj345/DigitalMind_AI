"""TF-IDF + one-vs-rest logistic regression: the classical risk baseline.

Included because it is the honest reference point. If a fine-tuned DistilBERT does not
beat this on macro-F1, the extra dependency is not earning its keep, and the report says
so.

Multi-label: one binary logistic regression per indicator, each with its own decision
threshold chosen on the **dev** split (never on test). Thresholds matter more than they
look: at 0.5 the benign look-alikes in the synthetic corpus produce a flood of false
positives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .lexicon import RISK_LABELS


@dataclass
class ThresholdChoice:
    label: str
    threshold: float
    dev_f1: float


@dataclass
class TfidfRiskModel:
    """Fitted vectorizer + per-label classifiers."""

    vectorizer: TfidfVectorizer | None = None
    classifiers: dict[str, LogisticRegression] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    labels: tuple[str, ...] = RISK_LABELS

    # -- training --------------------------------------------------------
    def fit(
        self,
        texts: list[str],
        label_matrix: np.ndarray,
        dev_texts: list[str] | None = None,
        dev_labels: np.ndarray | None = None,
        seed: int = 20260310,
        tune_thresholds: bool = True,
    ) -> "TfidfRiskModel":
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
        )
        X = self.vectorizer.fit_transform(texts)
        for i, label in enumerate(self.labels):
            clf = LogisticRegression(
                max_iter=2000, C=4.0, class_weight="balanced", random_state=seed
            )
            clf.fit(X, label_matrix[:, i])
            self.classifiers[label] = clf

        self.thresholds = {label: 0.5 for label in self.labels}
        if tune_thresholds and dev_texts is not None and dev_labels is not None:
            Xd = self.vectorizer.transform(dev_texts)
            self.thresholds = self.tune_thresholds(Xd, dev_labels)
        return self

    def tune_thresholds(self, X, dev_labels: np.ndarray) -> dict[str, float]:
        """Pick the per-label threshold that maximises F1 on dev.

        We search only 0.10-0.90: below 0.10 the model flags almost everything, above
        0.90 it becomes so conservative the demo looks broken.
        """
        out: dict[str, float] = {}
        for i, label in enumerate(self.labels):
            probs = self.classifiers[label].predict_proba(X)[:, 1]
            truth = dev_labels[:, i]
            best_f1, best_t = -1.0, 0.5
            for t in np.arange(0.10, 0.901, 0.02):
                pred = probs >= t
                f1 = _f1(truth, pred)
                # Prefer the higher threshold on ties: fewer false alarms by default.
                if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) <= 1e-9 and t > best_t):
                    best_f1, best_t = f1, float(t)
            out[label] = round(best_t, 4)
        return out

    # -- inference -------------------------------------------------------
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.vectorizer is not None, "model is not fitted"
        X = self.vectorizer.transform(texts)
        cols = [self.classifiers[label].predict_proba(X)[:, 1] for label in self.labels]
        return np.vstack(cols).T

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(texts)
        preds = np.zeros_like(probs, dtype=bool)
        for i, label in enumerate(self.labels):
            preds[:, i] = probs[:, i] >= self.thresholds[label]
        return probs, preds

    def top_features(self, label: str, k: int = 15) -> list[tuple[str, float]]:
        """Highest-weight ngrams for a label. Shown in the UI as 'why this fired'."""
        if self.vectorizer is None or label not in self.classifiers:
            return []
        names = np.asarray(self.vectorizer.get_feature_names_out())
        coefs = self.classifiers[label].coef_[0]
        order = np.argsort(coefs)[::-1]
        return [(str(names[i]), float(coefs[i])) for i in order[:k]]

    # -- persistence -----------------------------------------------------
    def save(self, directory: Path) -> Path:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifiers": self.classifiers,
                "thresholds": self.thresholds,
                "labels": list(self.labels),
            },
            directory / "tfidf_risk.joblib",
        )
        (directory / "tfidf_risk.json").write_text(
            json.dumps(
                {
                    "labels": list(self.labels),
                    "thresholds": self.thresholds,
                    "top_features": {l: self.top_features(l, 20) for l in self.labels},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory / "tfidf_risk.joblib"

    @classmethod
    def load(cls, directory: Path) -> "TfidfRiskModel | None":
        import joblib

        path = directory / "tfidf_risk.joblib"
        if not path.exists():
            return None
        blob = joblib.load(path)
        model = cls(labels=tuple(blob["labels"]))
        model.vectorizer = blob["vectorizer"]
        model.classifiers = blob["classifiers"]
        model.thresholds = blob["thresholds"]
        return model


def _f1(truth: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum((truth == 1) & pred))
    fp = float(np.sum((truth == 0) & pred))
    fn = float(np.sum((truth == 1) & ~pred))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

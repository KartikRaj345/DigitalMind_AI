"""Probability calibration.

A fine-tuned classifier's sigmoid outputs are typically over-confident. Since this
project reports "potential manipulation risk" to a human, a score of 0.99 that fires 40%
of the time on benign reminders is worse than useless: it erodes trust in every other
flag. So calibration is not optional here.

Implemented in-tree (no `calibration` package - see requirements-ml.txt):

* :func:`isotonic_fit` / :func:`isotonic_apply` - non-parametric, monotone, good with
  enough dev data.
* :func:`platt_fit` / :func:`platt_apply` - logistic regression on the logit, robust in
  the low-data regime.
* :func:`expected_calibration_error` - the ECE we report.
* :func:`reliability_table` - binned confidences vs empirical accuracy, used by the
  report so a reader can see the shape of the miscalibration.

All functions operate on plain numpy arrays and are tested directly, so the ECE numbers
in ``reports/`` are reproducible without torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Calibrator:
    """A fitted monotone mapping from raw score to calibrated probability."""

    method: str  # 'isotonic' | 'platt' | 'identity'
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    slope: float = 1.0
    intercept: float = 0.0

    # -- fitting ---------------------------------------------------------
    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        method: str = "auto",
        eps: float = 1e-6,
    ) -> "Calibrator":
        scores = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        if scores.size == 0:
            return cls(method="identity")
        if method == "auto":
            # Isotonic needs enough positives to fit a curve; otherwise fall back to Platt.
            n_pos = float(labels.sum())
            method = "isotonic" if n_pos >= 20 and len(scores) >= 60 else "platt"
        if method == "isotonic":
            return cls._fit_isotonic(scores, labels)
        if method == "platt":
            return cls._fit_platt(scores, labels)
        return cls(method="identity")

    @classmethod
    def _fit_isotonic(cls, scores: np.ndarray, labels: np.ndarray) -> "Calibrator":
        """Pool-adjacent-violators on (score, label) sorted by score."""
        order = np.argsort(scores, kind="stable")
        s = scores[order]
        y = labels[order]

        # Average y within tied scores first, otherwise PAVA can create steps at ties.
        uniq, inv = np.unique(s, return_inverse=True)
        sums = np.zeros(uniq.size)
        counts = np.zeros(uniq.size)
        np.add.at(sums, inv, y)
        np.add.at(counts, inv, 1.0)
        means = sums / counts
        weights = counts

        # PAVA with weights: monotone non-decreasing values.
        values: list[float] = []
        wts: list[float] = []
        starts: list[int] = []
        for i, (v, w) in enumerate(zip(means, weights)):
            values.append(float(v))
            wts.append(float(w))
            starts.append(i)
            while len(values) > 1 and values[-2] > values[-1]:
                w2, v2 = wts.pop(), values.pop()
                s2 = starts.pop()
                w1, v1 = wts.pop(), values.pop()
                s1 = starts.pop()
                total = w1 + w2
                values.append((v1 * w1 + v2 * w2) / total)
                wts.append(total)
                starts.append(min(s1, s2))

        cal = cls(method="isotonic")
        cal.thresholds = uniq[np.array(starts, dtype=int)]
        cal.values = np.clip(np.array(values, dtype=float), 0.0, 1.0)
        return cal

    @classmethod
    def _fit_platt(cls, scores: np.ndarray, labels: np.ndarray) -> "Calibrator":
        """Logistic regression on the logit of the raw score, fitted by Newton steps."""
        eps = 1e-6
        p = np.clip(scores, eps, 1 - eps)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        X = np.hstack([x, np.ones((x.shape[0], 1))])
        w = np.zeros(2)
        for _ in range(100):
            z = X @ w
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
            grad = X.T @ (s - labels)
            hess = (X * (s * (1 - s))[:, None]).T @ X + 1e-8 * np.eye(2)
            try:
                delta = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:  # pragma: no cover - numerical edge
                break
            w -= delta
            if np.max(np.abs(delta)) < 1e-9:
                break
        return cls(method="platt", slope=float(w[0]), intercept=float(w[1]))

    # -- application -----------------------------------------------------
    def apply(self, scores: np.ndarray | float) -> np.ndarray | float:
        s = np.asarray(scores, dtype=float)
        if self.method == "identity" or self.thresholds.size == 0 and self.method != "platt":
            return np.clip(s, 0.0, 1.0) if s.ndim else float(np.clip(s, 0.0, 1.0))
        if self.method == "platt":
            eps = 1e-6
            p = np.clip(s, eps, 1 - eps)
            z = np.log(p / (1 - p)) * self.slope + self.intercept
            out = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        else:
            out = np.interp(s, self.thresholds, self.values, left=self.values[0],
                            right=self.values[-1])
        return out if s.ndim else float(out)

    __call__ = apply

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "thresholds": self.thresholds.tolist(),
            "values": self.values.tolist(),
            "slope": self.slope,
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        return cls(
            method=data.get("method", "identity"),
            thresholds=np.asarray(data.get("thresholds", []), dtype=float),
            values=np.asarray(data.get("values", []), dtype=float),
            slope=float(data.get("slope", 1.0)),
            intercept=float(data.get("intercept", 0.0)),
        )


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Standard ECE with a weighted average of |accuracy - confidence| per bin.

    ``strategy='quantile'`` gives equal-count bins, which is the better estimator when the
    score distribution is heavily skewed towards 0 - the normal case here.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    if scores.size == 0:
        return 0.0
    n_bins = max(1, min(n_bins, scores.size))
    if strategy == "quantile":
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            return float(abs(scores.mean() - labels.mean()))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, len(edges) - 2)
    ece = 0.0
    n = scores.size
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = scores[mask].mean()
        acc = labels[mask].mean()
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def max_calibration_error(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    if scores.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, max(2, n_bins) + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, len(edges) - 2)
    worst = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() < 5:  # ignore near-empty bins
            continue
        worst = max(worst, abs(labels[mask].mean() - scores[mask].mean()))
    return float(worst)


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    if scores.size == 0:
        return 0.0
    return float(np.mean((scores - labels) ** 2))


def reliability_table(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Per-bin confidence vs empirical accuracy, for the report."""
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, max(2, n_bins) + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, len(edges) - 2)
    rows: list[dict] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        rows.append(
            {
                "bin_low": round(float(edges[b]), 4),
                "bin_high": round(float(edges[b + 1]), 4),
                "count": count,
                "mean_confidence": round(float(scores[mask].mean()), 4) if count else None,
                "empirical_rate": round(float(labels[mask].mean()), 4) if count else None,
            }
        )
    return rows

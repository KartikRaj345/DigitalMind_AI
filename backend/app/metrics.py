"""Shared metrics for the evaluation suite.

Everything is implemented here rather than pulled from ``sklearn.metrics`` so the numbers
in ``reports/`` are auditable line by line, and so the multi-label conventions are ours
and stated in the report:

* **NER**: strict span match (exact character offsets). A predicted span counts as correct
  only if its offsets match a gold span exactly. Partial overlap is reported separately as
  ``boundary_f1`` because it is the difference between "found the name" and "found the name
  with the right boundaries".
* **Risk**: per-label precision/recall/F1, plus macro and micro averages. A label counts as
  positive if any indicator fires.
* **Entity resolution**: pairwise precision/recall/F1 over the co-membership relation.
* **Retrieval**: Recall@k, MRR, nDCG@k with graded relevance from the gold record set.
* **Calibration**: ECE, MCE, Brier - see :mod:`app.risk.calibration`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Generic classification
# ---------------------------------------------------------------------------


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    support: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def prf(truth: Sequence[bool], pred: Sequence[bool]) -> PRF:
    tp = sum(1 for t, p in zip(truth, pred) if t and p)
    fp = sum(1 for t, p in zip(truth, pred) if not t and p)
    fn = sum(1 for t, p in zip(truth, pred) if t and not p)
    tn = sum(1 for t, p in zip(truth, pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp + fn, tp, fp, fn, tn)


def macro_f1(per_label: dict[str, PRF]) -> float:
    values = [v.f1 for v in per_label.values()]
    return sum(values) / len(values) if values else 0.0


def micro_f1(per_label: dict[str, PRF]) -> float:
    tp = sum(v.tp for v in per_label.values())
    fp = sum(v.fp for v in per_label.values())
    fn = sum(v.fn for v in per_label.values())
    if not (tp + fp + fn):
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def accuracy(truth: Sequence[bool], pred: Sequence[bool]) -> float:
    if not truth:
        return 0.0
    return sum(1 for t, p in zip(truth, pred) if t == p) / len(truth)


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

Span = tuple[int, int, str]  # (start, end, label)


@dataclass
class NerResult:
    strict: PRF
    boundary: PRF
    exact_text: float
    per_label: dict[str, PRF] = field(default_factory=dict)
    label_accuracy: float = 0.0
    n_gold: int = 0
    n_pred: int = 0

    def as_dict(self) -> dict:
        return {
            "strict": self.strict.as_dict(),
            "boundary_relaxed": self.boundary.as_dict(),
            "exact_text_match": round(self.exact_text, 4),
            "label_accuracy": round(self.label_accuracy, 4),
            "n_gold": self.n_gold,
            "n_pred": self.n_pred,
            "per_label": {k: v.as_dict() for k, v in sorted(self.per_label.items())},
        }


def ner_scores(gold: Sequence[Span], pred: Sequence[Span]) -> NerResult:
    """Strict (exact offsets + label) and boundary-relaxed (offset overlap) scoring."""
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    strict = PRF(
        precision=tp / len(pred_set) if pred_set else 0.0,
        recall=tp / len(gold_set) if gold_set else 0.0,
        f1=(2 * tp / (len(pred_set) + len(gold_set))) if (pred_set or gold_set) else 0.0,
        support=len(gold_set),
        tp=tp,
        fp=fp,
        fn=fn,
    )

    # Boundary-relaxed: any overlap counts, regardless of label agreement.
    matched_gold: set[int] = set()
    boundary_tp = 0
    for s, e, _label in pred:
        for i, (gs, ge, _gl) in enumerate(gold):
            if i in matched_gold:
                continue
            if s < ge and gs < e:
                matched_gold.add(i)
                boundary_tp += 1
                break
    boundary = PRF(
        precision=boundary_tp / len(pred_set) if pred_set else 0.0,
        recall=boundary_tp / len(gold_set) if gold_set else 0.0,
        f1=(2 * boundary_tp / (len(pred_set) + len(gold_set))) if (pred_set or gold_set) else 0.0,
        support=len(gold_set),
        tp=boundary_tp,
        fp=len(pred_set) - boundary_tp,
        fn=len(gold_set) - boundary_tp,
    )

    exact_text = 0.0
    if pred and gold:
        for s, e, _ in pred:
            if any(gs == s and ge == e for gs, ge, _ in gold):
                exact_text += 1
        exact_text /= len(pred)

    per_label: dict[str, PRF] = {}
    labels = {lbl for _s, _e, lbl in gold} | {lbl for _s, _e, lbl in pred}
    label_correct = 0
    for label in sorted(labels):
        g = {x for x in gold if x[2] == label}
        p = {x for x in pred if x[2] == label}
        inter = len(g & p)
        label_correct += inter
        per_label[label] = PRF(
            precision=inter / len(p) if p else 0.0,
            recall=inter / len(g) if g else 0.0,
            f1=2 * inter / (len(p) + len(g)) if (p or g) else 0.0,
            support=len(g),
            tp=inter,
            fp=len(p) - inter,
            fn=len(g) - inter,
        )

    return NerResult(
        strict=strict,
        boundary=boundary,
        exact_text=exact_text,
        per_label=per_label,
        label_accuracy=label_correct / len(pred_set) if pred_set else 0.0,
        n_gold=len(gold_set),
        n_pred=len(pred_set),
    )


def aggregate_ner(results: Iterable[NerResult]) -> NerResult:
    """Micro-average several NerResults by pooling counts."""
    results = list(results)
    if not results:
        return NerResult(strict=PRF(0, 0, 0, 0), boundary=PRF(0, 0, 0, 0), exact_text=0.0)

    def pooled(kind: str) -> PRF:
        tp = sum(getattr(getattr(r, kind), "tp") for r in results)
        fp = sum(getattr(getattr(r, kind), "fp") for r in results)
        fn = sum(getattr(getattr(r, kind), "fn") for r in results)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return PRF(precision, recall, f1, tp + fn, tp, fp, fn)

    n_pred = sum(r.n_pred for r in results)
    exact_text = (
        sum(r.exact_text * r.n_pred for r in results) / n_pred if n_pred else 0.0
    )
    labels = sorted({lbl for r in results for lbl in r.per_label})
    per_label = {
        label: pooled_label(results, label, label) for label in labels
    }
    return NerResult(
        strict=pooled("strict"),
        boundary=pooled("boundary"),
        exact_text=exact_text,
        per_label=per_label,
        n_gold=sum(r.n_gold for r in results),
        n_pred=n_pred,
    )


def pooled_label(results: list[NerResult], pred_attr: str, label: str) -> PRF:
    tp = sum(r.per_label[label].tp for r in results if label in r.per_label)
    fp = sum(r.per_label[label].fp for r in results if label in r.per_label)
    fn = sum(r.per_label[label].fn for r in results if label in r.per_label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp + fn, tp, fp, fn)


# ---------------------------------------------------------------------------
# Entity resolution (pairwise)
# ---------------------------------------------------------------------------


@dataclass
class PairwiseResult:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict:
        return {
            "pairwise_precision": round(self.precision, 4),
            "pairwise_recall": round(self.recall, 4),
            "pairwise_f1": round(self.f1, 4),
            "true_positive_pairs": self.true_positives,
            "false_positive_pairs": self.false_positives,
            "false_negative_pairs": self.false_negatives,
        }


def pairwise_prf(gold_clusters: Sequence[set[str]], pred_clusters: Sequence[set[str]]) -> PairwiseResult:
    """Pairwise precision/recall/F1 over cluster co-membership.

    The B-cubed family converges to this; pairwise is the standard for entity resolution
    and is what we report, with the caveat noted in the report that it over-rewards large
    clusters.
    """

    def pairs(clusters: Sequence[set[str]]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for cluster in clusters:
            items = sorted(cluster)
            for i, a in enumerate(items):
                for b in items[i + 1 :]:
                    out.add((a, b))
        return out

    gold = pairs(gold_clusters)
    pred = pairs(pred_clusters)
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PairwiseResult(precision, recall, f1, tp, fp, fn)


def b_cubed(gold_clusters: Sequence[set[str]], pred_clusters: Sequence[set[str]]) -> dict[str, float]:
    """B-cubed precision/recall/F1. Reported alongside pairwise because pairwise is
    biased toward big clusters and this is not."""
    gold_of: dict[str, str] = {}
    pred_of: dict[str, str] = {}
    for i, cluster in enumerate(gold_clusters):
        for item in cluster:
            gold_of[item] = f"g{i}"
    for i, cluster in enumerate(pred_clusters):
        for item in cluster:
            pred_of[item] = f"p{i}"

    keys = sorted(set(gold_of) | set(pred_of))
    precisions: list[float] = []
    recalls: list[float] = []
    for key in keys:
        g = next((c for c in gold_clusters if key in c), {key})
        p = next((c for c in pred_clusters if key in c), {key})
        overlap = len(g & p)
        precisions.append(overlap / len(p) if p else 0.0)
        recalls.append(overlap / len(g) if g else 0.0)

    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cluster_counts(clusters: Sequence[set[str]]) -> dict[str, int]:
    sizes: dict[int, int] = defaultdict(int)
    for cluster in clusters:
        sizes[len(cluster)] += 1
    return {str(k): v for k, v in sorted(sizes.items())}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    num_queries: int

    def as_dict(self) -> dict:
        return {
            "recall_at_k": {str(k): round(v, 4) for k, v in sorted(self.recall_at_k.items())},
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": {str(k): round(v, 4) for k, v in sorted(self.ndcg_at_k.items())},
            "num_queries": self.num_queries,
        }


def retrieval_scores(
    run: Sequence[Sequence[str]], gold: Sequence[set[str]], ks: Sequence[int] = (1, 5, 10)
) -> RetrievalResult:
    """Recall@k, MRR and nDCG@k over ranked record-id lists.

    Relevance is binary here: a retrieved record is relevant if it is in the gold set for
    that query. That is a real limitation - "which records mention Project X" has one right
    answer set, but a topical query has many - and it is stated in the report.
    """
    if not run:
        return RetrievalResult({k: 0.0 for k in ks}, 0.0, {k: 0.0 for k in ks}, 0)

    max_k = max(ks)
    recalls: dict[int, float] = {k: 0.0 for k in ks}
    ndcgs: dict[int, float] = {k: 0.0 for k in ks}
    rr_total = 0.0

    for ranked, relevant in zip(run, gold):
        if not relevant:
            continue
        seen: set[str] = set()
        deduped: list[str] = []
        for rid in ranked:
            if rid not in seen:
                seen.add(rid)
                deduped.append(rid)

        for k in ks:
            top = deduped[:k]
            hits = len([r for r in top if r in relevant])
            recalls[k] += hits / len(relevant)
            dcg = sum(
                1.0 / math.log2(i + 2) for i, r in enumerate(top) if r in relevant
            )
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
            ndcgs[k] += dcg / ideal if ideal else 0.0

        for i, rid in enumerate(deduped):
            if rid in relevant:
                rr_total += 1.0 / (i + 1)
                break

    n = len(gold)
    return RetrievalResult(
        recall_at_k={k: v / n for k, v in recalls.items()},
        mrr=rr_total / n,
        ndcg_at_k={k: v / n for k, v in ndcgs.items()},
        num_queries=n,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float], statistic=sum, n_boot: int = 1000, seed: int = 20260310, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean or sum of per-example scores."""
    import numpy as np

    if len(values) < 2:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = statistic(arr[idx], axis=1) / arr.size
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return (lo, hi)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Used for demo-corpus assertions."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

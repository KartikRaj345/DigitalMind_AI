"""Error analysis across all components.

    python -m backend.scripts.error_analysis

Aggregates the errors the other evaluation scripts found and groups them by *cause*, not by
component. The point is to name the failure modes a maintainer would actually have to fix,
and to separate "the model is wrong" from "the gold annotation is incomplete".
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..app.datasets import ensure_synthetic, load_public_sms, load_split
from ..app.logging_config import configure_logging

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _load(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def categorize_extraction_error(text: str, label: str, kind: str) -> str:
    """Name *why* a span was wrong, so the categories are actionable."""
    stripped = text.strip()
    lowered = stripped.lower()

    if re.match(r"^\d", stripped):
        return "numeric token tagged as an entity"
    if lowered.startswith(("project ", "q1", "q2", "q3", "q4")):
        return "project/period code tagged as a name"
    if re.match(r"^[a-z]\.\s+[A-Z][a-z]+$", stripped) or stripped.endswith("."):
        return "initial or trailing punctuation in the span"
    if len(stripped.split()) >= 3:
        return "multi-word span that is not a single entity"
    if kind == "false_positive" and label == "ORG":
        return "organization suffix heuristic over-firing"
    if kind == "false_positive" and label == "PERSON":
        return "capitalisation heuristic over-firing"
    if kind == "false_negative":
        return "entity missed (boundary or type disagreement)"
    return "other"


def analyze_extraction(report: dict[str, Any]) -> dict[str, Any]:
    errors = report["configurations"]["hybrid"]["errors"]
    by_cause: dict[str, int] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for error in errors:
        cause = categorize_extraction_error(error["text"], error["label"], error["type"])
        by_cause[cause] += 1
        if len(examples[cause]) < 5:
            examples[cause].append(f"[{error['label']}] {error['text']!r}")
    return {
        "n_errors_analysed": len(errors),
        "by_cause": dict(by_cause.most_common()),
        "examples": dict(examples),
        "note": (
            "Several of these are not model errors. PROJECT/period codes and numeric tokens "
            "are extracted with a low confidence and are not annotated as entities in the "
            "gold set, so they appear as false positives here."
        ),
    }


def analyze_risk(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("dev", "test"):
        methods = report["splits"][split]["methods"]
        for name, payload in methods.items():
            if payload.get("status") in {"not trained", "not available"}:
                continue
            fp = payload["false_positives"]
            out[f"{split}/{name}"] = {
                "macro_f1": round(
                    sum(v["f1"] for k, v in payload["labels"].items() if k != "any_indicator")
                    / 4,
                    4,
                ),
                "benign_false_positive_rate": fp["benign_false_positive_rate"],
                "benign_false_positive_rate_ci95": fp["benign_false_positive_rate_ci95"],
                "benign_lookalike_false_positive_rate": fp[
                    "benign_lookalike_false_positive_rate"
                ],
            }

    interpretations = [
        {
            "observation": "TF-IDF + logistic regression scores higher than the rule baseline.",
            "explanation": (
                "The classifier memorised our own templates. The benign look-alikes reuse the "
                "same sentence frames as the attack records with a different middle clause, "
                "which is exactly the discrimination a bag-of-ngrams model is good at and a "
                "human is not fooled by. Treat the TF-IDF numbers as an upper bound that "
                "depends on the generator, not as evidence of generalisation."
            ),
            "mitigation": (
                "The real public corpus is scored separately in risk_eval.json; that number is "
                "the one to trust about wording we did not write."
            ),
        },
        {
            "observation": "The rule baseline produces false positives on genuine urgent reminders.",
            "explanation": (
                "'your account will be closed' and 'immediately' both appear in legitimate bank "
                "reminders. The mitigator list catches some of these and not all of them: a "
                "reminder that omits the 'no charge' sentence loses its mitigation."
            ),
            "mitigation": (
                "The API reports each indicator separately with its evidence, so a user can see "
                "that the flag came from a policy sentence rather than a demand. Severity only "
                "reaches 'warning' when a payment demand is paired with a threat or a deadline."
            ),
        },
    ]
    out["interpretation"] = interpretations
    return out


def analyze_retrieval(report: dict[str, Any]) -> dict[str, Any]:
    by_name = {c["name"]: c for c in report["configurations"]}
    observations = []
    if {"bm25", "hybrid_rrf"} <= set(by_name):
        bm25 = by_name["bm25"]["overall"]["recall_at_k"]["5"]
        hybrid = by_name["hybrid_rrf"]["overall"]["recall_at_k"]["5"]
        observations.append(
            {
                "observation": f"Recall@5: BM25 {bm25:.3f} vs hybrid RRF {hybrid:.3f}.",
                "explanation": (
                    "Most queries in this benchmark are entity names lifted from the record, "
                    "which is the strongest possible case for exact lexical matching and the "
                    "weakest possible case for embeddings. Fusing a weaker channel with equal "
                    "weight therefore costs recall on this corpus."
                ),
                "mitigation": (
                    "The per-kind breakdown shows where fusion does help. Keep the two channels "
                    "separately selectable in Settings, and let the paraphrase queries - where "
                    "BM25 has no lexical overlap at all - be the reason hybrid is the default."
                ),
            }
        )
    paraphrase = by_name.get("hybrid_rrf", {}).get("by_query_kind", {}).get("paraphrase")
    if paraphrase:
        observations.append(
            {
                "observation": (
                    f"Paraphrase queries: Recall@1 {paraphrase['recall_at_k']['1']:.3f} but "
                    f"MRR {paraphrase['mrr']:.3f}."
                ),
                "explanation": (
                    "The right record is usually found, but several records are equally "
                    "relevant so the first position is nearly a coin flip. Binary relevance "
                    "makes nDCG look poor for the same reason."
                ),
                "mitigation": (
                    "Treat MRR as the meaningful number for topical queries and Recall@k for "
                    "'which records mention X'. Both are reported."
                ),
            }
        )
    return {
        "embedding_backend": report["embedding_backend"],
        "interpretation": observations,
        "caveats": report["caveats"],
    }


def analyze_public_data() -> dict[str, Any]:
    split = load_public_sms()
    if split is None:
        return {"status": "skipped"}
    ham = sum(1 for r in split.records if r.scenario == "public_sms_ham")
    spam = len(split) - ham
    projected = {label: 0 for label in ("urgency", "threat_language", "emotional_pressure", "payment_request")}
    for record in split.records:
        for label in record.risk_labels:
            projected[label] += 1
    return {
        "status": "ok",
        "n": len(split),
        "spam": spam,
        "ham": ham,
        "spam_fraction": round(spam / len(split), 4),
        "indicator_projection": projected,
        "interpretation": [
            "The projection onto our indicators is deliberately conservative: only 'urgency' "
            "survives on real spam text. Real SMS spam rarely makes an explicit threat or an "
            "explicit emotional appeal - it relies on scarcity and a prize instead.",
            "That is a genuine finding, not a bug. It means our four indicators cover the "
            "long-form business-email manipulation in Scenario A far better than they cover "
            "consumer SMS fraud, and the report should not claim otherwise.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Error analysis.")
    parser.add_argument("--seed", type=int, default=20260310)
    args = parser.parse_args()
    configure_logging()

    ensure_synthetic(seed=args.seed)
    load_split("test")  # ensure the corpus exists

    output: dict[str, Any] = {"generated_from": []}

    extraction = _load("extraction_eval.json")
    if extraction:
        output["extraction"] = analyze_extraction(extraction)
        output["generated_from"].append("extraction_eval.json")

    risk = _load("risk_eval.json")
    if risk:
        output["risk"] = analyze_risk(risk)
        output["generated_from"].append("risk_eval.json")

    retrieval = _load("retrieval_eval.json")
    if retrieval:
        output["retrieval"] = analyze_retrieval(retrieval)
        output["generated_from"].append("retrieval_eval.json")

    resolution = _load("entity_resolution_eval.json")
    if resolution:
        primary = resolution["primary"]
        output["entity_resolution"] = {
            "pairwise_f1": primary["pairwise"]["pairwise_f1"],
            "distractor_false_merge_rate": resolution["distractors"][
                "distractor_false_merge_rate"
            ],
            "interpretation": [
                {
                    "observation": (
                        "Recall is lower than precision: "
                        f"P={primary['pairwise']['pairwise_precision']:.3f} vs "
                        f"R={primary['pairwise']['pairwise_recall']:.3f}."
                    ),
                    "explanation": (
                        "The resolver is deliberately conservative. A missed merge leaves two "
                        "entities the user can inspect; a wrong merge silently asserts that two "
                        "different people are the same, which is the more damaging error in a "
                        "graph the user is meant to trust."
                    ),
                    "mitigation": (
                        "Clusters below the cohesion threshold are reported to the user as "
                        "possibly-merged rather than being merged silently."
                    ),
                }
            ],
        }
        output["generated_from"].append("entity_resolution_eval.json")

    output["public_data"] = analyze_public_data()

    out = REPORTS / "error_analysis.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: (v.get("n_errors_analysed") if isinstance(v, dict) else v)
         for k, v in output.items() if k != "generated_from"},
        indent=2,
    ))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

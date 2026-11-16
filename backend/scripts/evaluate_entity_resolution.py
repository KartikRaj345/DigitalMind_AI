"""Evaluate entity resolution.

    python -m backend.scripts.evaluate_entity_resolution

The synthetic corpus mentions each of its entities once, so scoring on it directly would
be vacuous (every cluster has one member). Instead this script builds a **surface-form
benchmark**: for every fictional person and organization in the corpus vocabulary it
generates the variants real text actually produces - different case, a title, a middle
initial, an email handle, punctuation drift, a common misspelling - and asks whether the
resolver puts them in one cluster while keeping *distractor* entities apart.

Distractors are the point. "Ana Delgado" / "Ana M. Delgado" / "Marco Delgado" share a
surname and must not merge. A resolver that scores 1.0 because nothing ever looks alike is
not measuring anything.

Scoring: pairwise P/R/F1 (the entity-resolution standard) and B-cubed, plus a threshold
sweep and a containment ablation so the operating point is shown, not asserted.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path
from typing import Any

from ..app.datasets import ensure_synthetic
from ..app.entity import resolve as resolve_mod
from ..app.ingest.generator import FIRST_NAMES, LAST_NAMES, ORGANIZATIONS, VENDORS
from ..app.logging_config import configure_logging
from ..app.metrics import b_cubed, pairwise_prf

REPORTS = Path(__file__).resolve().parents[2] / "reports"


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def person_variants(rng: random.Random, first: str, last: str, email_domain: str) -> list[str]:
    handle = f"{first.lower()}.{last.lower()}@{email_domain}"
    initial = f"{first[0]}. {last}"
    misspelled = _misspell(rng, last)
    return [
        f"{first} {last}",              # canonical
        f"{first} {last}".upper(),      # ALL CAPS header
        f"{first.lower()} {last.lower()}",
        f"Dr {first} {last}",           # title
        initial,                        # middle initial
        handle,                         # email address
        f"{first} {misspelled}",        # typo in the surname
        f"{first} {last}.",             # sentence-final punctuation
    ]


def _misspell(rng: random.Random, word: str) -> str:
    """A plausible typo: transpose, drop, or double a character."""
    if len(word) < 4:
        return word
    mode = rng.randrange(3)
    i = rng.randrange(1, len(word) - 2)
    if mode == 0:
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if mode == 1:
        return word[:i] + word[i + 1 :]
    return word[:i] + word[i] + word[i:]


def org_variants(rng: random.Random, name: str, email_domain: str) -> list[str]:
    """Variants of an organization name.

    Note the address variant uses the organization name as the local part
    (``northwind.bank@northwind.example``) rather than a role mailbox: a role mailbox such
    as ``no-reply@...`` carries no evidence that it belongs to the organization, so
    including one in a gold cluster would be labelling an impossible inference as correct.
    """
    words = name.split()
    slug = ".".join(strip_accents(w).lower() for w in words)
    abbreviation = " ".join([w[0] + "." for w in words[:-1]] + [words[-1]]) if len(words) > 1 else name
    misspelt = " ".join(words[:-1] + [_misspell(rng, words[-1])]) if len(words) > 1 else name
    return [
        name,
        name.upper(),
        name.lower(),
        f"the {name}",
        f"{name} Customer Service",          # containment: extra role tokens
        f"{slug}@{email_domain}",            # address whose handle is the org name
        abbreviation,                        # "Cedar Peak Sav." style
        misspelt,                            # one typo in the final token
    ]


def build_benchmark(seed: int = 20260310, n_persons: int = 40, n_orgs: int = 12) -> dict[str, Any]:
    """Gold clusters plus distractor pairs that must stay apart."""
    rng = random.Random(seed)
    firsts = rng.sample(FIRST_NAMES, min(n_persons, len(FIRST_NAMES)))
    lasts = rng.sample(LAST_NAMES, min(n_persons // 2, len(LAST_NAMES)))

    gold: list[set[str]] = []
    canonicals: list[tuple[str, str]] = []  # (canonical form, surname-ish key)

    for i, first in enumerate(firsts):
        last = lasts[i % len(lasts)]
        domain = f"{strip_accents(last).lower()}.example"
        forms = person_variants(rng, first, last, domain)
        gold.append({f"p{i}:{f}" for f in forms})
        canonicals.append((f"{first} {last}", last))

    for j, org in enumerate(ORGANIZATIONS[:n_orgs] + VENDORS[:n_orgs]):
        domain = f"{strip_accents(org).lower().replace(' ', '').replace(',', '')}.example"
        forms = org_variants(rng, org, domain)
        gold.append({f"o{j}:{f}" for f in forms})
        canonicals.append((org, org.split()[-1]))

    # Distractors: two *existing* canonicals that share a surname-ish token and must stay
    # in separate clusters. Built from the clusters themselves so both forms really exist.
    by_key: dict[str, list[str]] = {}
    for form, key in canonicals:
        by_key.setdefault(strip_accents(key).lower(), []).append(form)
    distractors: list[tuple[str, str]] = []
    for key, forms in sorted(by_key.items()):
        if len(forms) >= 2:
            distractors.append((forms[0], forms[1]))
        if len(forms) >= 3:
            distractors.append((forms[0], forms[2]))

    return {"gold_clusters": gold, "distractors": distractors, "seed": seed}


# ---------------------------------------------------------------------------
# Running the resolver
# ---------------------------------------------------------------------------


def run_resolver(
    benchmark: dict[str, Any], thresholds: tuple[float, float] | None = None
) -> tuple[list[set[str]], dict[str, str]]:
    """Resolve every variant.

    Returns ``(multi-member clusters in gold key space, membership of every gold item)``.
    The membership map covers singletons too, which is what the distractor check needs.
    """
    original = None
    if thresholds is not None:
        original = (resolve_mod.STRONG_THRESHOLD, resolve_mod.WEAK_THRESHOLD)
        resolve_mod.STRONG_THRESHOLD, resolve_mod.WEAK_THRESHOLD = thresholds
    try:
        norm_to_items: dict[tuple[str, str], set[str]] = {}
        mentions: list[resolve_mod.Mention] = []
        idx = 0
        for cluster in benchmark["gold_clusters"]:
            for item in sorted(cluster):
                prefix, _, form = item.partition(":")
                label = "PERSON" if prefix == "p" else "ORG"
                norm = resolve_mod.normalize(form, label)
                norm_to_items.setdefault((norm, label), set()).add(item)
                mentions.append(
                    resolve_mod.Mention(key=f"m{idx}", record_id=1, text=form, label=label)
                )
                idx += 1

        clusters = resolve_mod.cluster_mentions(mentions)

        predicted: list[set[str]] = []
        membership: dict[str, str] = {}
        for i, cluster in enumerate(clusters):
            members: set[str] = set()
            for key in cluster.mentions:
                norm, _, label = key.rpartition("::")
                members |= norm_to_items.get((norm, label), set())
            if members:
                membership[f"cluster{i}"] = ",".join(sorted(members))
            for item in members:
                membership[item] = f"cluster{i}"
            if len(members) > 1:
                predicted.append(members)
        return predicted, membership
    finally:
        if original is not None:
            resolve_mod.STRONG_THRESHOLD, resolve_mod.WEAK_THRESHOLD = original


def score(benchmark: dict[str, Any], predicted: list[set[str]]) -> dict[str, Any]:
    gold = benchmark["gold_clusters"]
    pairwise = pairwise_prf(gold, predicted)
    bcubed = b_cubed(gold, predicted)
    return {
        "pairwise": pairwise.as_dict(),
        "b_cubed": {k: round(v, 4) for k, v in bcubed.items()},
        "n_gold_clusters": len(gold),
        "n_pred_clusters": len(predicted),
        "mean_gold_cluster_size": round(
            sum(len(c) for c in gold) / len(gold) if gold else 0.0, 2
        ),
    }


def distractor_report(benchmark: dict[str, Any], membership: dict[str, str]) -> dict[str, Any]:
    """How many must-not-merge pairs ended up merged?"""
    merged = 0
    checked = 0
    examples: list[dict[str, Any]] = []
    for a_form, b_form in benchmark["distractors"]:
        a_key = next((k for k in membership if k.endswith(f":{a_form}")), None)
        b_key = next((k for k in membership if k.endswith(f":{b_form}")), None)
        if a_key is None or b_key is None:
            continue
        checked += 1
        if membership[a_key] == membership[b_key]:
            merged += 1
            if len(examples) < 6:
                examples.append({"a": a_form, "b": b_form, "merged_into": membership[a_key]})
    return {
        "distractor_pairs_checked": checked,
        "distractor_pairs_merged": merged,
        "distractor_false_merge_rate": round(merged / checked, 4) if checked else 0.0,
        "examples": examples,
    }


def threshold_sweep(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for strong, weak in [(1.01, 1.01), (0.99, 0.95), (0.95, 0.90), (0.92, 0.86), (0.88, 0.80), (0.85, 0.75)]:
        predicted, membership = run_resolver(benchmark, (strong, weak))
        outcome = score(benchmark, predicted)
        rows.append(
            {
                "strong_threshold": strong,
                "weak_threshold": weak,
                "pairwise_f1": outcome["pairwise"]["pairwise_f1"],
                "pairwise_precision": outcome["pairwise"]["pairwise_precision"],
                "pairwise_recall": outcome["pairwise"]["pairwise_recall"],
                "false_positive_pairs": outcome["pairwise"]["false_positive_pairs"],
                "false_negative_pairs": outcome["pairwise"]["false_negative_pairs"],
                "b_cubed_f1": outcome["b_cubed"]["f1"],
                "distractor_false_merge_rate": distractor_report(benchmark, membership)[
                    "distractor_false_merge_rate"
                ],
            }
        )
    return rows


def ablate_containment(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Measure what the token-containment and email-handle rules each contribute."""
    from rapidfuzz import fuzz

    original = resolve_mod.similarity

    def fuzzy_only(a: str, b: str, label: str = ""):
        na, nb = resolve_mod.normalize(a, label), resolve_mod.normalize(b, label)
        if not na or not nb:
            return 0.0, "none"
        if na == nb:
            return 1.0, "exact"
        score = max(fuzz.token_set_ratio(na, nb) / 100.0, fuzz.partial_ratio(na, nb) / 100.0 * 0.97)
        return (score, "fuzzy") if score >= 0.86 else (0.0, "none")

    real_email_handle = resolve_mod.email_handle

    def no_handle(a: str, b: str, label: str = ""):
        """Keep containment, drop the email-handle shortcut."""
        resolve_mod.email_handle = lambda name: None
        try:
            return original(a, b, label)
        finally:
            resolve_mod.email_handle = real_email_handle

    out: dict[str, Any] = {"full": score(benchmark, run_resolver(benchmark)[0])}

    resolve_mod.similarity = fuzzy_only
    try:
        out["without_containment_and_handle"] = score(benchmark, run_resolver(benchmark)[0])
    finally:
        resolve_mod.similarity = original

    resolve_mod.similarity = no_handle
    try:
        out["without_email_handle"] = score(benchmark, run_resolver(benchmark)[0])
    finally:
        resolve_mod.similarity = original

    for key in ("without_containment_and_handle", "without_email_handle"):
        out[f"{key}_delta_f1"] = round(
            out["full"]["pairwise"]["pairwise_f1"] - out[key]["pairwise"]["pairwise_f1"], 4
        )
    return out


def qualitative_cases() -> list[dict[str, Any]]:
    """Named cases a reader can check by eye."""
    cases: list[tuple[list[str], str]] = [
        (["Northwind Bank", "northwind bank", "Northwind Bank Customer Verification"], "org_case_and_containment"),
        (["Northwind Bank", "Northwind-secure"], "org_lookalike_domain"),
        (["Priya Raman", "priya.raman@studio.example", "Priya R. Raman"], "person_email_handle"),
        (["Ana Delgado", "Ana M. Delgado", "Marco Delgado"], "shared_surname_must_not_merge"),
        (["Colin Whitfield", "Colin"], "person_given_name_only"),
        (["Harbourline Credit Union", "the Harbourline Credit Union"], "leading_article"),
        (["Rafael Fontaine", "Rafel Fontaine"], "single_character_typo"),
    ]
    out = []
    for forms, case in cases:
        mentions = [
            resolve_mod.Mention(
                key=f"m{i}", record_id=1, text=f, label="ORG" if "Bank" in f or "Union" in f else "PERSON"
            )
            for i, f in enumerate(forms)
        ]
        clusters = resolve_mod.cluster_mentions(mentions)
        out.append(
            {
                "case": case,
                "input": forms,
                "n_clusters": len(clusters),
                "clusters": [
                    {
                        "canonical": c.canonical,
                        "members": sorted(c.surface_forms),
                        "cohesion": round(c.cohesion, 3),
                        "merges": [m.as_dict() for m in c.merges],
                    }
                    for c in clusters
                ],
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate entity resolution.")
    parser.add_argument("--seed", type=int, default=20260310)
    args = parser.parse_args()
    configure_logging()

    ensure_synthetic(seed=args.seed)
    benchmark = build_benchmark(seed=args.seed)

    predicted, membership = run_resolver(benchmark)
    primary = score(benchmark, predicted)
    distractors = distractor_report(benchmark, membership)

    print(
        f"pairwise F1 = {primary['pairwise']['pairwise_f1']:.4f} "
        f"(P={primary['pairwise']['pairwise_precision']:.4f}, "
        f"R={primary['pairwise']['pairwise_recall']:.4f})"
    )
    print(f"B-cubed  F1 = {primary['b_cubed']['f1']:.4f}")
    print(
        f"distractor false-merge rate = {distractors['distractor_false_merge_rate']:.4f} "
        f"({distractors['distractor_pairs_merged']}/{distractors['distractor_pairs_checked']})"
    )

    print("\nthreshold sweep (strong / weak):")
    sweep = threshold_sweep(benchmark)
    for row in sweep:
        marker = (
            "  <- operating point"
            if (row["strong_threshold"], row["weak_threshold"])
            == (resolve_mod.STRONG_THRESHOLD, resolve_mod.WEAK_THRESHOLD)
            else ""
        )
        print(
            f"  {row['strong_threshold']:.2f} / {row['weak_threshold']:.2f}  "
            f"F1={row['pairwise_f1']:.4f}  "
            f"FP-pairs={row['false_positive_pairs']}  "
            f"FN-pairs={row['false_negative_pairs']}  "
            f"distractor-merge={row['distractor_false_merge_rate']:.4f}{marker}"
        )

    ablation = ablate_containment(benchmark)
    print("\nablations (pairwise F1):")
    for key in ("full", "without_containment_and_handle", "without_email_handle"):
        print(f"  {key:32s} {ablation[key]['pairwise']['pairwise_f1']:.4f}")

    output = {
        "benchmark": {
            "n_clusters": len(benchmark["gold_clusters"]),
            "n_items": sum(len(c) for c in benchmark["gold_clusters"]),
            "variant_kinds": [
                "canonical", "upper case", "lower case", "title prefix",
                "middle initial", "email handle", "surname typo", "trailing punctuation",
                "leading article", "role suffix", "lookalike domain", "misspelled org token",
            ],
            "seed": args.seed,
        },
        "primary": primary,
        "distractors": distractors,
        "threshold_sweep": sweep,
        "ablation": ablation,
        "qualitative": qualitative_cases(),
        "thresholds_in_use": {
            "strong": resolve_mod.STRONG_THRESHOLD,
            "weak": resolve_mod.WEAK_THRESHOLD,
            "min_cohesion": resolve_mod.MIN_CLUSTER_COHESION,
        },
    }

    out = REPORTS / "entity_resolution_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

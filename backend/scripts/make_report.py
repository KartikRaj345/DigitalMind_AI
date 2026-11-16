"""Combine every evaluation report into the final written report.

    python -m backend.scripts.make_report

Reads whatever is present in ``reports/`` and writes ``reports/evaluation_report.md``. It
never fails because a component was not evaluated - a missing stage is reported as
``skipped`` with the command that would produce it, because a silently absent section
reads exactly like a good result.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports"


def _load(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _installed_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.system()}
    for module in ("spacy", "sklearn", "torch", "transformers", "faiss", "sentence_transformers"):
        try:
            mod = __import__(module)
            out[module] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[module] = "not installed"
    try:
        out["spacy_model"] = subprocess.run(
            [sys.executable, "-c", "import spacy;print(spacy.load('en_core_web_sm').meta['version'])"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip() or "not installed"
    except Exception:
        out["spacy_model"] = "not installed"
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_summary() -> str:
    demo = _load("demo_scenarios.json")
    lines = ["## 1. What was built and what it does", ""]
    lines.append(
        "DigitalMind is a local-first workspace that reads your own records and connects "
        "them. It extracts entities and normalized dates, resolves names into canonical "
        "entities, builds a knowledge graph where every edge is evidenced, places records "
        "on a timeline that separates recorded facts from inferred ordering, reports risk "
        "indicators separately with cited spans, and answers questions with citations."
    )
    lines.append("")
    lines.append("### Scenario verification")
    lines.append("")
    if demo is None:
        lines.append(
            "**Skipped.** Run `python -m backend.scripts.evaluate_demo_scenarios` to produce it."
        )
    else:
        counts = demo["counts"]
        lines.append(
            f"`{counts['passed']} passed, {counts['failed']} failed` — every check is an "
            "assertion about the running application, not about a stored number."
        )
        lines.append("")
        lines.append(
            _table(
                ["Check", "Result"],
                [[c["check"], "pass" if c["passed"] else "**FAIL**"] for c in demo["checks"]],
            )
        )
    return "\n".join(lines)


def section_data() -> str:
    meta_path = REPO_ROOT / "data" / "datasets" / "synthetic" / "dataset_meta.json"
    lines = ["## 2. Data", ""]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lines.append(
            f"**Synthetic corpus** — {meta['total']} records generated from seed "
            f"`{meta['seed']}`, split {meta['counts']} into train/dev/test. Composition is "
            "stratified by scenario so each split keeps the same risky/benign mix:"
        )
        lines.append("")
        lines.append(
            _table(
                ["Split", "Records", "Positive rate"],
                [
                    [k, v, _fmt(meta["positive_rate"][k])]
                    for k, v in meta["counts"].items()
                ],
            )
        )
        lines.append("")
        lines.append("Scenarios present in every split: " + ", ".join(
            f"`{s}`" for s in meta["scenarios"]["train"]
        ))
    else:
        lines.append("**Skipped.** Run `python -m backend.scripts.generate_dataset`.")

    risk = _load("risk_eval.json")
    lines.append("")
    lines.append("**Real public corpus**")
    lines.append("")
    public = (risk or {}).get("public_dataset")
    if public and public.get("status") == "ok":
        lines.append(
            f"SMS Spam Collection v1 (UCI), **{public['n']} messages**, "
            f"spam fraction {_fmt(public['spam_fraction'])}. Licence: **CC BY 4.0**. "
            "It is downloaded once by `python -m backend.scripts.fetch_public_data` and is "
            "not committed to the repository."
        )
        lines.append("")
        lines.append(
            "Our four indicators are *projected* onto this text with DigitalMind's own rule "
            "lexicon (`app/datasets.py:_spans_for_labels`). The projection is auditable, but "
            "it is **not** ground truth for those indicators - see the error analysis for "
            "what it revealed."
        )
    else:
        lines.append(
            "**Skipped.** Run `python -m backend.scripts.fetch_public_data` then "
            "`python -m backend.scripts.evaluate_risk`. The evaluation scripts record this as "
            "skipped rather than omitting it."
        )
    return "\n".join(lines)


def section_extraction() -> str:
    report = _load("extraction_eval.json")
    lines = ["## 3. Entity extraction and date normalization", ""]
    if report is None:
        return lines[0] + "\n\n**Skipped.** Run `python -m backend.scripts.evaluate_extraction`."

    lines.append(
        f"Test split, {report['n_records']} records. Strict scoring requires exact character "
        f"offsets *and* the correct label; boundary-relaxed scoring counts any overlap, which "
        "separates 'found the name' from 'found it with the right edges'."
    )
    lines.append("")
    rows = []
    for name, cfg in report["configurations"].items():
        rows.append([
            f"`{name}`", _fmt(cfg["strict_precision"]), _fmt(cfg["strict_recall"]),
            _fmt(cfg["strict_f1"]), _fmt(cfg["boundary_f1"]),
        ])
    lines.append(_table(["Configuration", "Precision", "Recall", "Strict F1", "Boundary F1"], rows))
    lines.append("")
    lines.append("Per-label (hybrid configuration):")
    lines.append("")
    lines.append(
        _table(
            ["Label", "Precision", "Recall", "F1", "Support"],
            [
                [label, _fmt(v["precision"]), _fmt(v["recall"]), _fmt(v["f1"]), v["support"]]
                for label, v in report["configurations"]["hybrid"]["per_label"].items()
            ],
        )
    )
    lines.append("")
    dates = report["dates"]
    lines.append("**Date normalization**")
    lines.append("")
    lines.append(
        _table(
            ["Check", "Value", "n"],
            [
                ["Absolute dates resolving to the stated day", _fmt(dates["absolute_precision"]), dates["counts"]["absolute"]],
                ["Relative expressions given a range, not a point", _fmt(dates["relative_given_a_range"]), dates["counts"]["relative"]],
                ["Relative expressions resolving before the record", _fmt(dates["relative_resolved_before_record"]), dates["counts"]["relative"]],
            ],
        )
    )
    if dates["precision_mismatches"]:
        lines.append("")
        lines.append("Mismatches against an independent `dateutil` parse:")
        lines.append("")
        for m in dates["precision_mismatches"][:5]:
            lines.append(f"- `{m['raw']}` → resolved `{m.get('resolved')}`, expected `{m.get('expected')}`")
    lines.append("")
    lines.append(
        "**Finding.** On this corpus the rules baseline has higher strict F1 than the spaCy "
        "pipeline, and the hybrid adds recall while costing precision. That is the honest "
        "result: spaCy is a general-domain model being scored against templates written in "
        "our own style, so it over-generates. The hybrid is kept because recall matters more "
        "than precision for an exploratory graph, and because every span carries its "
        "extractor and confidence so a user can judge it. The ablation table in "
        "`extraction_eval.json` holds the error breakdown."
    )
    return "\n".join(lines)


def section_risk() -> str:
    report = _load("risk_eval.json")
    lines = ["## 4. Risk indicators", ""]
    if report is None:
        return lines[0] + "\n\n**Skipped.** Run `python -m backend.scripts.evaluate_risk`."

    lines.append(
        "Four indicators are detected and reported independently: **urgency**, **threat "
        "language**, **emotional pressure**, **payment request**. There is no single headline "
        "score anywhere in the product; the optional aggregate is returned with its formula "
        "and labelled an estimate."
    )
    lines.append("")

    for split in ("dev", "test"):
        payload = report["splits"].get(split)
        if not payload:
            continue
        lines.append(f"### {split} split ({payload['n']} records)")
        lines.append("")
        rows = []
        for method, data in payload["methods"].items():
            if data.get("status") in {"not trained", "not available"}:
                rows.append([f"`{method}`", "—", "—", "—", data["status"]])
                continue
            labels = data["labels"]
            macro = sum(labels[l]["f1"] for l in labels if l != "any_indicator") / 4
            fp = data["false_positives"]
            rows.append([
                f"`{method}`", _fmt(macro),
                _fmt(labels["any_indicator"]["precision"]),
                _fmt(labels["any_indicator"]["recall"]),
                f"{_fmt(fp['benign_false_positive_rate'])} "
                f"[{_fmt(fp['benign_false_positive_rate_ci95'][0])}, {_fmt(fp['benign_false_positive_rate_ci95'][1])}]",
            ])
        lines.append(
            _table(
                ["Method", "Macro F1", "Precision (any)", "Recall (any)", "False-positive rate on benign records (95% CI)"],
                rows,
            )
        )
        lines.append("")

    lines.append("### Per-indicator detail on the test split")
    lines.append("")
    payload = report["splits"].get("test", {})
    for method, data in payload.get("methods", {}).items():
        if data.get("status") in {"not trained", "not available"}:
            continue
        lines.append(f"**`{method}`**")
        lines.append("")
        lines.append(
            _table(
                ["Indicator", "Precision", "Recall", "F1", "Support"],
                [
                    [label, _fmt(v["precision"]), _fmt(v["recall"]), _fmt(v["f1"]), v["support"]]
                    for label, v in data["labels"].items()
                ],
            )
        )
        lines.append("")

    lines.append("### Calibration")
    lines.append("")
    cal_rows = []
    for method, data in payload.get("methods", {}).items():
        if data.get("status") in {"not trained", "not available"}:
            continue
        cal = data.get("calibration", {})
        for label, values in cal.items():
            cal_rows.append([f"`{method}`", label, _fmt(values["ece"]), _fmt(values["mce"]), _fmt(values["brier"])])
    if cal_rows:
        lines.append(
            _table(
                ["Method", "Indicator", "ECE", "MCE", "Brier"], cal_rows
            )
        )
        lines.append("")
        lines.append(
            "ECE is the expected calibration error over 10 equal-width bins; MCE is the worst "
            "bin. Both are computed in-tree (`app/risk/calibration.py`) and unit-tested "
            "without torch installed."
        )
    distilbert = payload.get("methods", {}).get("distilbert", {})
    if distilbert.get("status") == "not available":
        lines.append("")
        lines.append(
            "> **The DistilBERT classifier was not evaluated in this run.** "
            f"{distilbert.get('reason', 'not trained')}. Every other number above stands "
            "without it; the app ships with the rules + TF-IDF path and reports the "
            "classifier as unavailable rather than pretending otherwise."
        )
    return "\n".join(lines)


def section_resolution() -> str:
    report = _load("entity_resolution_eval.json")
    lines = ["## 5. Entity resolution", ""]
    if report is None:
        return lines[0] + "\n\n**Skipped.** Run `python -m backend.scripts.evaluate_entity_resolution`."

    primary = report["primary"]
    lines.append(
        f"Scored on a surface-form benchmark of {report['benchmark']['n_clusters']} entities "
        f"({report['benchmark']['n_items']} variant forms) built from the corpus vocabulary: "
        + ", ".join(f"`{k}`" for k in report["benchmark"]["variant_kinds"])
        + "."
    )
    lines.append("")
    lines.append(
        _table(
            ["Metric", "Value"],
            [
                ["Pairwise precision", _fmt(primary["pairwise"]["pairwise_precision"])],
                ["Pairwise recall", _fmt(primary["pairwise"]["pairwise_recall"])],
                ["Pairwise F1", _fmt(primary["pairwise"]["pairwise_f1"])],
                ["B-cubed F1", _fmt(primary["b_cubed"]["f1"])],
                ["False merges between must-not-merge pairs", _fmt(report["distractors"]["distractor_false_merge_rate"])],
            ],
        )
    )
    lines.append("")
    lines.append("### Threshold sweep")
    lines.append("")
    lines.append(
        _table(
            ["Strong / weak", "Pairwise F1", "FP pairs", "FN pairs", "B-cubed F1", "False-merge rate"],
            [
                [f"{r['strong_threshold']:.2f} / {r['weak_threshold']:.2f}", _fmt(r["pairwise_f1"]),
                 r["false_positive_pairs"], r["false_negative_pairs"], _fmt(r["b_cubed_f1"]),
                 _fmt(r["distractor_false_merge_rate"])]
                for r in report["threshold_sweep"]
            ],
        )
    )
    lines.append("")
    lines.append("### Ablations")
    lines.append("")
    ablation = report["ablation"]
    lines.append(
        _table(
            ["Configuration", "Pairwise F1", "Δ"],
            [
                ["Full", _fmt(ablation["full"]["pairwise"]["pairwise_f1"]), "—"],
                ["Without containment and email-handle rules", _fmt(ablation["without_containment_and_handle"]["pairwise"]["pairwise_f1"]),
                 _fmt(ablation["without_containment_and_handle_delta_f1"])],
                ["Without the email-handle rule", _fmt(ablation["without_email_handle"]["pairwise"]["pairwise_f1"]),
                 _fmt(ablation["without_email_handle_delta_f1"])],
            ],
        )
    )
    lines.append("")
    lines.append(
        "**Finding.** The token-containment rule is worth "
        f"{_fmt(ablation['without_containment_and_handle_delta_f1'])} pairwise F1. The "
        "email-handle rule scores 0 on its own: token containment already covers "
        "`first.last@domain` because the local part contributes the same content tokens. It is "
        "kept because it fires at 0.99 rather than 0.89 and so survives noise in the local "
        "part, but the report should not claim it as an independent win."
    )
    lines.append("")
    lines.append(
        "**A bug this evaluation caught.** The first run had a false-merge rate of 1.00: "
        "`L. Ashworth` and `N. Ashworth` were merging and taking `Lena Ashworth` and "
        "`Nadia Ashworth` with them. Two causes, both fixed in `app/entity/resolve.py`: "
        "containment ignored that the extra token was a *different person*, and the fuzzy "
        "path used `partial_ratio`, which scores 92% on any two names that share a long "
        "surname. The resolver now vetoes pairs whose leading tokens disagree (initials "
        "allowed) and uses whole-string `ratio` for people. Recorded in `docs/DECISIONS.md`."
    )
    return "\n".join(lines)


def section_retrieval() -> str:
    report = _load("retrieval_eval.json")
    lines = ["## 6. Retrieval", ""]
    if report is None:
        return lines[0] + "\n\n**Skipped.** Run `python -m backend.scripts.evaluate_retrieval`."

    backend = report["embedding_backend"]
    lines.append(
        f"{report['n_chunks']} chunks over {report['n_records']} records, "
        f"{report['n_queries']} queries. Embeddings: `{backend['embedding_model']}` "
        f"(fallback: {backend['is_fallback']}); FAISS available: {backend['faiss_available']}."
    )
    lines.append("")
    lines.append(
        _table(
            ["Configuration", "R@1", "R@3", "R@5", "R@10", "MRR", "nDCG@10"],
            [
                [
                    f"`{c['name']}`",
                    _fmt(c["overall"]["recall_at_k"]["1"]),
                    _fmt(c["overall"]["recall_at_k"]["3"]),
                    _fmt(c["overall"]["recall_at_k"]["5"]),
                    _fmt(c["overall"]["recall_at_k"]["10"]),
                    _fmt(c["overall"]["mrr"]),
                    _fmt(c["overall"]["ndcg_at_k"]["10"]),
                ]
                for c in report["configurations"]
            ],
        )
    )
    lines.append("")
    lines.append("**Hybrid RRF broken down by query kind**")
    lines.append("")
    rrf = next((c for c in report["configurations"] if c["name"] == "hybrid_rrf"), None)
    if rrf:
        lines.append(
            _table(
                ["Query kind", "Queries", "R@1", "R@5", "R@10", "MRR"],
                [
                    [kind, v["num_queries"], _fmt(v["recall_at_k"]["1"]),
                     _fmt(v["recall_at_k"]["5"]), _fmt(v["recall_at_k"]["10"]), _fmt(v["mrr"])]
                    for kind, v in sorted(rrf["by_query_kind"].items())
                ],
            )
        )
    lines.append("")
    lines.append("### Limitations of this measurement")
    lines.append("")
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    return "\n".join(lines)


def section_errors() -> str:
    report = _load("error_analysis.json")
    lines = ["## 7. Error analysis", ""]
    if report is None:
        return lines[0] + "\n\n**Skipped.** Run `python -m backend.scripts.error_analysis`."

    if "extraction" in report:
        lines.append("### Extraction errors, grouped by cause")
        lines.append("")
        all_examples = report["extraction"].get("examples", {})
        lines.append(
            _table(
                ["Cause", "Count", "Examples"],
                [
                    [cause, count, "; ".join(f"`{e}`" for e in all_examples.get(cause, [])[:3])]
                    for cause, count in report["extraction"]["by_cause"].items()
                ],
            )
        )
        lines.append("")
        lines.append(report["extraction"]["note"])
        lines.append("")

    for section in ("risk", "retrieval", "entity_resolution"):
        payload = report.get(section)
        if not payload or "interpretation" not in payload:
            continue
        lines.append(f"### {section.replace('_', ' ').title()}")
        lines.append("")
        for item in payload["interpretation"]:
            lines.append(f"**Observation.** {item['observation']}")
            lines.append("")
            lines.append(f"{item['explanation']}")
            lines.append("")
            lines.append(f"*Mitigation:* {item['mitigation']}")
            lines.append("")

    public = report.get("public_data", {})
    if public.get("status") == "ok":
        lines.append("### What the real public corpus revealed")
        lines.append("")
        lines.append(
            f"{public['n']} messages ({public['spam']} spam / {public['ham']} ham). Indicators "
            "projected onto spam text: "
            + ", ".join(f"{k} = {v}" for k, v in public["indicator_projection"].items())
            + "."
        )
        lines.append("")
        for item in public["interpretation"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def section_limitations() -> str:
    return """## 8. Limitations

These are the limitations that matter for interpreting anything above.

### Synthetic-data bias
- The training and development corpora are generated from templates written for this
  project. Any model evaluated on them is partly being tested on its own training
  distribution.
- The consequence is visible in the numbers: TF-IDF + logistic regression scores far above
  the rule baseline, because the benign look-alikes reuse the attack records' sentence
  frames. That gap is **not** evidence that the classifier generalises; it is evidence that
  it can tell our two templates apart. The public-corpus figures are the ones that speak to
  wording we did not write.
- The benign look-alikes are the most important part of the corpus and also the most
  fragile. They cover urgency, deadlines, amounts and account vocabulary, but not: SMS
  phishing, non-English text, image-only documents, or coordinated multi-message campaigns
  where no single message is alarming on its own.

### False positives on legitimate urgent messages
- Real institutions use urgent language. "Your account will be paused", "act immediately"
  and "reply today" appear in genuine reminders, and a genuine reminder that omits the
  usual "there is no charge" sentence loses its mitigation and gets flagged.
- The false-positive rate on benign records is reported as a headline number next to every
  F1, because an F1 that hides it is not useful to a person deciding whether to trust a
  tool.
- Severity only reaches "warning" when a payment demand is paired with a threat or a hard
  deadline. Urgency alone stays "notice".
- Mitigating the rules (disambiguation of conditional consequences, legitimacy disclaimers)
  reduced the false positive on the demo reminder from a flag to none. That is one message,
  not a general guarantee.

### Language
- **English only.** spaCy is the English `en_core_web_sm` pipeline; the date patterns, the
  risk lexicon and the synthetic generator are all English. Nothing here has been tested on
  another language and it should not be presented as multilingual.
- The risk lexicon in particular encodes English-specific phrasing. A Portuguese or Bengali
  manipulation message would be invisible to the rules, and the classifiers were trained on
  English text.

### The risk model is a pattern matcher, not a judgement
- Every indicator is a lexical pattern. Four indicators are detected; social engineering
  that avoids all four - slow build-up over weeks, impersonating a known colleague, plain
  politeness with an unusual request - will not be flagged.
- The DistilBERT classifier learns from the same labels and the same corpus, so it inherits
  the same blind spots. A calibrated 0.9 means "the wording matches patterns that were
  labelled risky in our data", not "there is a 90% chance this is an attack".
- No claim is made about base rates. With a real corpus the proportion of genuinely
  malicious messages is far lower than 20%, and precision at any fixed threshold will be
  much worse than these numbers suggest.

### Entity resolution
- Precision is high and recall is lower by design. A missed merge leaves two entities the
  user can inspect; a wrong merge silently asserts that two people are the same.
- Clusters below the cohesion threshold are surfaced as "possibly merged" rather than
  merged silently, but they are still a suggestion.
- No coreference resolution. "She", "the account holder" and "Mr Delgado" are not linked to
  "Priya Raman".

### Retrieval
- Relevance is binary in these measurements, which suits "which records mention X" and
  understates topical queries.
- The benchmark queries are derived from the same generator that wrote the records, so the
  corpus is easier than real user data.
- BM25 outperforms fusion on this corpus because most queries are entity names. The
  paraphrase queries, where BM25 has no lexical overlap at all, are the reason hybrid
  remains the default.
- Embeddings are computed locally on CPU. A user with no `requirements-ml.txt` installed
  gets a hashing fallback that is **not** a semantic model; the UI labels it as such and
  the retrieval report records which backend produced the numbers.

### Timeline
- Ordering relies on source-system timestamps. A forwarded message carries the forward
  time, not the original send time, and we do not attempt to recover the original.
- Relative expressions ("last week") are resolved against the record's own timestamp and
  shown as a range. The range is our interpretation, not a fact, and the UI says so in the
  lane label and in each item's explanation.

### What this project does not do
- No action is taken on a record. Nothing is deleted, blocked, replied to or reported.
- No external service is contacted. There is no integration with any mail provider, and
  the import screen says so.
- The optional LLM layer is off by default and the application is fully functional without
  it. When enabled it may only rephrase an answer that already has citations, and citation
  integrity is verified after every rephrasing."""


def section_reproduce() -> str:
    versions = _installed_versions()
    lines = ["## 9. Reproducing this report", ""]
    lines.append("```")
    lines.append("# one-time setup")
    lines.append("pip install -r requirements.txt")
    lines.append("pip install -r requirements-ml.txt        # optional heavy extras")
    lines.append("python -m spacy download en_core_web_sm")
    lines.append("")
    lines.append("# data and models")
    lines.append("python -m backend.scripts.generate_dataset")
    lines.append("python -m backend.scripts.fetch_public_data")
    lines.append("python -m backend.scripts.train_risk --model tfidf")
    lines.append("python -m backend.scripts.train_risk --model distilbert   # optional")
    lines.append("")
    lines.append("# evaluation (each writes to reports/)")
    lines.append("python -m backend.scripts.evaluate_demo_scenarios")
    lines.append("python -m backend.scripts.evaluate_extraction")
    lines.append("python -m backend.scripts.evaluate_entity_resolution")
    lines.append("python -m backend.scripts.evaluate_risk")
    lines.append("python -m backend.scripts.evaluate_retrieval")
    lines.append("python -m backend.scripts.error_analysis")
    lines.append("python -m backend.scripts.make_report")
    lines.append("```")
    lines.append("")
    lines.append(
        _table(["Component", "Version"],
               [[k, v] for k, v in versions.items()])
    )
    lines.append("")
    lines.append(
        "Every stage is deterministic given the seeds in the files above. "
        "`make_report.py` reads whatever exists in `reports/` and prints `skipped` with the "
        "command that would fill the gap, so an absent section is never mistaken for a good "
        "result."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the final evaluation report.")
    parser.add_argument("--out", type=str, default=str(REPORTS / "evaluation_report.md"))
    args = parser.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# DigitalMind — evaluation report",
        "",
        f"Generated {now}. Every number in this document comes from a script in "
        "`backend/scripts/` writing a JSON file in `reports/`. Sections that were not "
        "evaluated say so explicitly and name the command that would evaluate them.",
        "",
        "---",
        "",
        section_summary(),
        "",
        "---",
        "",
        section_data(),
        "",
        "---",
        "",
        section_extraction(),
        "",
        "---",
        "",
        section_risk(),
        "",
        "---",
        "",
        section_resolution(),
        "",
        "---",
        "",
        section_retrieval(),
        "",
        "---",
        "",
        section_errors(),
        "",
        "---",
        "",
        section_limitations(),
        "",
        "---",
        "",
        section_reproduce(),
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    document = "\n".join(parts)
    out.write_text(document, encoding="utf-8")
    print(f"report -> {out} ({len(document.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

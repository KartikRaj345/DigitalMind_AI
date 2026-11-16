"""Evaluate hybrid retrieval.

    python -m backend.scripts.evaluate_retrieval

Builds a query set with gold relevance sets and scores each configuration separately:

* ``bm25``          - the lexical baseline
* ``embeddings``    - the semantic channel (sentence-transformers, or the hashing fallback,
                      which is labelled as such in the output)
* ``hybrid_rrf``    - Reciprocal Rank Fusion, the production default
* ``hybrid_weighted`` - min-max weighted blend, for comparison
* ``no_fusion``     - the lexical channel alone, as the ablation reference

Relevance is binary (the record is or is not in the gold set), which suits
"which records mention X" but understates performance on topical queries. That limitation
is stated in the report rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from ..app.datasets import ensure_synthetic, load_split
from ..app.logging_config import configure_logging
from ..app.metrics import retrieval_scores
from ..app.retrieval import bm25_index, chunker, embeddings as emb_mod, hybrid

REPORTS = Path(__file__).resolve().parents[2] / "reports"
KS = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def build_queries(split) -> list[dict[str, Any]]:
    """Queries with gold record sets, derived from the generator's own structure.

    Four kinds, because one kind would flatter the retriever:
      1. ``entity``     - "Northwind Bank" (the record mentions it)
      2. ``phrase``     - a distinctive phrase lifted from one record
      3. ``topic``      - the scenario topic (many records share it)
      4. ``paraphrase`` - wording that does *not* appear verbatim, so only the semantic
                          channel can help
    """
    queries: list[dict[str, Any]] = []
    by_topic: dict[str, set[str]] = {}
    for record in split.records:
        if record.scenario:
            by_topic.setdefault(record.scenario, set()).add(record.id)

    for record in split.records:
        entities = [s.text for s in record.entity_spans if s.label in {"PERSON", "ORG"}]
        if entities and len(queries) < 120:
            queries.append(
                {
                    "query": entities[0],
                    "kind": "entity",
                    "gold": {record.id},
                    "note": f"record {record.id} mentions this entity",
                }
            )
        if record.scenario == "generated_risk" and len(queries) < 200:
            # A distinctive phrase: the payment line, which is unique to attack records.
            gold = {
                r.id
                for r in split.records
                if "verification account" in r.body or "release the funds" in r.body
            }
            if gold:
                queries.append(
                    {
                        "query": "transfer money to release the funds",
                        "kind": "topic",
                        "gold": gold,
                        "note": "records demanding money to release access",
                    }
                )
        if record.scenario == "generated_benign_lookalike" and len(queries) < 260:
            gold = {
                r.id
                for r in split.records
                if "remains the account holder of record" in r.body
            }
            if gold:
                queries.append(
                    {
                        "query": "confirming who the account holder of record is",
                        "kind": "paraphrase",
                        "gold": gold,
                        "note": "paraphrase: no query token appears verbatim",
                    }
                )
        if record.scenario == "generated_backref" and len(queries) < 320:
            gold = {
                r.id
                for r in split.records
                if "as we discussed" in r.body.lower() or "we agreed to move" in r.body.lower()
            }
            if gold:
                queries.append(
                    {
                        "query": "moving the review to a later date after an agreement",
                        "kind": "paraphrase",
                        "gold": gold,
                        "note": "paraphrase of a back-referencing message",
                    }
                )

    for scenario, ids in sorted(by_topic.items()):
        queries.append(
            {
                "query": scenario.replace("_", " "),
                "kind": "topic",
                "gold": ids,
                "note": f"all {scenario} records",
            }
        )
    return queries


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def build_corpus(split) -> dict[int, tuple[int, int, str, int, int]]:
    """Chunk the test split in memory and build both retrieval channels."""
    rows: list[tuple[int, int, int, str, int, int]] = []
    lookup: dict[int, tuple[int, int, str, int, int]] = {}
    chunk_id = 0
    for record in split.records:
        for i, chunk in enumerate(chunker.renumber(chunker.chunk_text(record.body))):
            chunk_id += 1
            rows.append((chunk_id, _record_index(record, split), i, chunk.text,
                         chunk.start_char, chunk.end_char))
            lookup[chunk_id] = (rows[-1][1], i, chunk.text, chunk.start_char, chunk.end_char)

    # Build into the process-wide singletons, because that is what the hybrid layer reads.
    lexical = bm25_index.get_default_index()
    lexical.build(rows)

    embedder = emb_mod.get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedder.encode([r[3] for r in rows])
    semantic = emb_mod.get_default_index()
    semantic.build(
        [(r[0], r[1], emb_mod.pack_vector(v)) for r, v in zip(rows, vectors)],
        model_name=getattr(embedder, "name", "unknown"),
    )
    return lookup


_INDEX: dict[str, int] = {}


def _record_index(record, split) -> int:
    if record.id not in _INDEX:
        _INDEX[record.id] = len(_INDEX) + 1
    return _INDEX[record.id]


# ---------------------------------------------------------------------------


def run_configuration(
    name: str, queries: list[dict[str, Any]], lookup, mode: str, fusion: str = "rrf"
) -> dict[str, Any]:
    """Score one retrieval configuration.

    Both the ranking and the gold sets use the integer corpus ids assigned during indexing,
    so they are directly comparable.
    """
    lexical = bm25_index.get_default_index()
    semantic = emb_mod.get_default_index()
    run: list[list[int]] = []
    gold: list[set[int]] = []
    per_kind: dict[str, list[list[int]]] = {}
    per_kind_gold: dict[str, list[set[int]]] = {}

    for q in queries:
        relevant = {_INDEX[r] for r in q["gold"] if r in _INDEX}
        if not relevant:
            continue
        result = hybrid.search(
            q["query"],
            chunk_lookup=lookup,
            lexical_index=lexical if mode in {"lexical", "hybrid"} else None,
            semantic_index=semantic if mode in {"semantic", "hybrid"} else None,
            embeddings_model=semantic.model_name,
            mode=mode,
            top_k=20,
            fusion=fusion,
        )
        ordered: list[int] = []
        for hit in result.hits:
            if hit.record_id not in ordered:
                ordered.append(hit.record_id)
        run.append(ordered)
        gold.append(relevant)
        per_kind.setdefault(q["kind"], []).append(ordered)
        per_kind_gold.setdefault(q["kind"], []).append(relevant)

    overall = retrieval_scores(run, gold, KS)
    by_kind = {
        kind: retrieval_scores(runs, per_kind_gold[kind], KS).as_dict()
        for kind, runs in per_kind.items()
    }

    return {
        "name": name,
        "mode": mode,
        "fusion": fusion if mode == "hybrid" else None,
        "queries_scored": len(run),
        "overall": overall.as_dict(),
        "by_query_kind": by_kind,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval.")
    parser.add_argument("--seed", type=int, default=20260310)
    args = parser.parse_args()
    configure_logging()

    ensure_synthetic(seed=args.seed)
    split = load_split("test")
    _INDEX.clear()

    lookup = build_corpus(split)
    queries = build_queries(split)
    print(f"built {len(lookup)} chunks from {len(split)} records; {len(queries)} queries")

    embedder = emb_mod.get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    backend_note = {
        "embedding_model": getattr(embedder, "name", "unknown"),
        "is_fallback": bool(getattr(embedder, "is_fallback", False)),
        "faiss_available": emb_mod.faiss_available(),
        "note": (
            "The hashing fallback embedder is NOT a semantic model. If it is in use, the "
            "semantic numbers below measure lexical overlap in dense form, not meaning, and "
            "install requirements-ml.txt to train against real embeddings."
            if getattr(embedder, "is_fallback", False)
            else "Real sentence-transformer embeddings, computed locally on CPU."
        ),
    }

    configurations = [
        ("bm25", "lexical", "rrf"),
        ("embeddings", "semantic", "rrf"),
        ("hybrid_rrf", "hybrid", "rrf"),
        ("hybrid_weighted", "hybrid", "weighted"),
    ]

    results: list[dict[str, Any]] = []
    for name, mode, fusion in configurations:
        outcome = run_configuration(name, queries, lookup, mode, fusion)
        results.append(outcome)
        overall = outcome["overall"]
        print(
            f"  {name:16s} R@1={overall['recall_at_k']['1']:.4f} "
            f"R@5={overall['recall_at_k']['5']:.4f} R@10={overall['recall_at_k']['10']:.4f} "
            f"MRR={overall['mrr']:.4f} nDCG@10={overall['ndcg_at_k']['10']:.4f}"
        )

    output = {
        "n_records": len(split),
        "n_chunks": len(lookup),
        "n_queries": len(queries),
        "query_kinds": sorted({q["kind"] for q in queries}),
        "embedding_backend": backend_note,
        "configurations": results,
        "caveats": [
            "Relevance is binary: a record is relevant or it is not. This suits 'which "
            "records mention X' and understates topical queries.",
            "Queries are derived from the same generator that wrote the records, so the "
            "corpus is easier than real user data would be.",
            "nDCG uses the same binary gains as Recall, so it mostly re-orders the Recall "
            "numbers rather than adding an independent view.",
        ],
    }

    out = REPORTS / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

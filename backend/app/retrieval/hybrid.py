"""Hybrid retrieval: BM25 + embeddings, fused with Reciprocal Rank Fusion.

Why RRF rather than a weighted score blend: BM25 scores are unbounded and corpus-dependent
while cosine similarities are bounded, so any linear combination needs per-corpus tuning
and breaks when the corpus changes. RRF only uses ranks, which makes the fusion
corpus-independent and robust - the property we want when a user adds one record.

Both fusion strategies are implemented so the evaluation can *measure* the choice:

* ``rrf``      - Reciprocal Rank Fusion (default), ``k=60``
* ``weighted`` - min-max normalised weighted sum of the two channels

The API returns per-channel results as well as the fused ranking, so a user can see
whether a hit came from exact wording, meaning, or both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..logging_config import get_logger
from .bm25_index import Bm25Index, LexicalHit
from .embeddings import SemanticHit, SemanticIndex

logger = get_logger(__name__)

RRF_K = 60


@dataclass
class FusedHit:
    record_id: int
    chunk_id: int
    text: str = ""
    start_char: int = 0
    end_char: int = 0
    ordinal: int = 0
    score: float = 0.0
    rank: int = 0
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None
    agreement: str = "fusion"  # both | lexical_only | semantic_only
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "ordinal": self.ordinal,
            "score": round(self.score, 6),
            "rank": self.rank,
            "lexical_rank": self.lexical_rank,
            "semantic_rank": self.semantic_rank,
            "lexical_score": None if self.lexical_score is None else round(self.lexical_score, 6),
            "semantic_score": None if self.semantic_score is None else round(self.semantic_score, 6),
            "agreement": self.agreement,
            "chunks": self.chunks,
        }


@dataclass
class HybridResult:
    query: str
    hits: list[FusedHit]
    mode: str
    channels: dict[str, int]
    degraded: list[str] = field(default_factory=list)
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "hits": [h.as_dict() for h in self.hits],
            "channels": self.channels,
            "degraded": self.degraded,
            "explanation": self.explanation,
        }


def rrf_fuse(
    lexical: list[LexicalHit], semantic: list[SemanticHit], k: int = RRF_K, top_k: int = 20
) -> list[tuple[int, float, int | None, int | None]]:
    """Reciprocal Rank Fusion over chunk ids.

    Returns ``(chunk_id, score, lexical_rank, semantic_rank)`` sorted by score.
    """
    scores: dict[int, float] = {}
    lex_rank: dict[int, int] = {}
    sem_rank: dict[int, int] = {}

    for hit in lexical:
        lex_rank[hit.chunk_id] = hit.rank
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)
    for hit in semantic:
        sem_rank[hit.chunk_id] = hit.rank
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        (chunk_id, score, lex_rank.get(chunk_id), sem_rank.get(chunk_id))
        for chunk_id, score in ordered[:top_k]
    ]


def weighted_fuse(
    lexical: list[LexicalHit],
    semantic: list[SemanticHit],
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
    top_k: int = 20,
) -> list[tuple[int, float, int | None, int | None]]:
    """Min-max normalised weighted blend. Included for the ablation study."""

    def normalize(values: list[float]) -> dict[int, float]:
        if not values:
            return {}
        lo, hi = min(values), max(values)
        span = hi - lo
        if span <= 1e-12:
            return {i: (1.0 if hi > 0 else 0.0) for i in range(len(values))}
        return {i: (v - lo) / span for i, v in enumerate(values)}

    lex_norm = normalize([h.score for h in lexical])
    sem_norm = normalize([h.score for h in semantic])

    scores: dict[int, float] = {}
    lex_rank: dict[int, int] = {h.chunk_id: h.rank for h in lexical}
    sem_rank: dict[int, int] = {h.chunk_id: h.rank for h in semantic}
    for i, hit in enumerate(lexical):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + lexical_weight * lex_norm[i]
    for i, hit in enumerate(semantic):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + semantic_weight * sem_norm[i]

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        (chunk_id, score, lex_rank.get(chunk_id), sem_rank.get(chunk_id))
        for chunk_id, score in ordered[:top_k]
    ]


def search(
    query: str,
    chunk_lookup: dict[int, tuple[int, int, str, int, int]],
    lexical_index: Bm25Index | None = None,
    semantic_index: SemanticIndex | None = None,
    embeddings_model: str = "",
    mode: str = "hybrid",
    top_k: int = 20,
    allowed_records: set[int] | None = None,
    fusion: str = "rrf",
    rrf_k: int = RRF_K,
) -> HybridResult:
    """Run retrieval and fuse the channels.

    ``mode`` is one of ``lexical | semantic | hybrid``. ``hybrid`` degrades gracefully: if
    one channel is unavailable the result says so in ``degraded`` rather than silently
    returning the other one.
    """
    degraded: list[str] = []

    lexical: list[LexicalHit] = []
    semantic: list[SemanticHit] = []

    if mode in {"lexical", "hybrid"} and lexical_index is not None:
        lexical = lexical_index.search(query, top_k=top_k * 3, allowed_records=allowed_records)
        if mode == "lexical" and not lexical and mode == "hybrid":
            degraded.append("lexical")
    if mode in {"semantic", "hybrid"} and semantic_index is not None and semantic_index.size:
        qvec = semantic_index.embed_query(query, embeddings_model)
        semantic = semantic_index.search(qvec, top_k=top_k * 3, allowed_records=allowed_records)

    if mode == "lexical":
        degraded = ["semantic"]
    elif mode == "semantic":
        degraded = ["lexical"]
    else:
        if not semantic and semantic_index is None or semantic_index.size == 0:
            degraded.append("semantic")
        if not lexical and (lexical_index is None or lexical_index.size == 0):
            degraded.append("lexical")

    if mode == "lexical" or not semantic:
        fused = [(h.chunk_id, h.score, h.rank, None) for h in lexical[:top_k]]
    elif mode == "semantic" or not lexical:
        fused = [(h.chunk_id, h.score, None, h.rank) for h in semantic[:top_k]]
    elif fusion == "weighted":
        fused = weighted_fuse(lexical, semantic, top_k=top_k)
    else:
        fused = rrf_fuse(lexical, semantic, k=rrf_k, top_k=top_k)

    lex_by_id = {h.chunk_id: h for h in lexical}
    sem_by_id = {h.chunk_id: h for h in semantic}

    hits: list[FusedHit] = []
    for rank, (chunk_id, score, lex_r, sem_r) in enumerate(fused, start=1):
        info = chunk_lookup.get(chunk_id)
        if info is None:
            continue
        record_id, ordinal, text, start_char, end_char = info
        agreement = (
            "both" if lex_r and sem_r else "lexical_only" if lex_r else "semantic_only"
        )
        hits.append(
            FusedHit(
                record_id=record_id,
                chunk_id=chunk_id,
                text=text,
                start_char=start_char,
                end_char=end_char,
                ordinal=ordinal,
                score=score,
                rank=rank,
                lexical_rank=lex_r,
                semantic_rank=sem_r,
                lexical_score=lex_by_id[chunk_id].score if chunk_id in lex_by_id else None,
                semantic_score=sem_by_id[chunk_id].score if chunk_id in sem_by_id else None,
                agreement=agreement,
            )
        )

    explanation = _explain(mode, fusion, hits, degraded)
    return HybridResult(
        query=query,
        hits=hits,
        mode=mode,
        channels={"lexical": len(lexical), "semantic": len(semantic)},
        degraded=[d for d in dict.fromkeys(degraded) if d],
        explanation=explanation,
    )


def _explain(mode: str, fusion: str, hits: list[FusedHit], degraded: list[str]) -> str:
    both = sum(1 for h in hits if h.agreement == "both")
    lex_only = sum(1 for h in hits if h.agreement == "lexical_only")
    sem_only = sum(1 for h in hits if h.agreement == "semantic_only")
    parts = [
        f"Ranking used the {mode} channel"
        + (f" with {fusion} fusion." if mode == "hybrid" else ".")
    ]
    if mode == "hybrid":
        parts.append(
            f"{both} result(s) were found by both BM25 and embeddings, "
            f"{lex_only} by keyword match only and {sem_only} by meaning only."
        )
    if degraded:
        names = ", ".join(degraded)
        parts.append(
            f"The {names} channel was unavailable, so those results are missing. "
            "Ranking quality is reduced accordingly."
        )
    return " ".join(parts)


def group_by_record(hits: Iterable[FusedHit]) -> list[dict[str, Any]]:
    """Collapse chunk hits into one entry per record, keeping the best chunk."""
    best: dict[int, FusedHit] = {}
    for hit in hits:
        current = best.get(hit.record_id)
        if current is None or hit.score > current.score:
            best[hit.record_id] = hit
    out = []
    for hit in sorted(best.values(), key=lambda h: -h.score):
        out.append(hit.as_dict())
    return out

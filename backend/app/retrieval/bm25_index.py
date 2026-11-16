"""BM25 lexical retrieval baseline.

Deliberately simple: Okapi BM25 over record chunks, via ``rank_bm25`` for the reference
implementation. Rebuilds from the database on demand and caches per-index-version, so a
deleted record can never leave a stale hit behind.

Scores are returned unnormalised alongside the ranks; the fusion layer in
:mod:`app.retrieval.hybrid` handles score comparability.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)

K1 = 1.5
B = 0.75


@dataclass
class LexicalHit:
    chunk_id: int
    record_id: int
    ordinal: int
    text: str
    start_char: int
    end_char: int
    score: float
    rank: int

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "record_id": self.record_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "score": round(self.score, 6),
            "rank": self.rank,
            "retriever": "bm25",
        }


class Bm25Index:
    """Thread-safe BM25 index over chunks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25 = None
        self._chunks: list[tuple[int, int, int, str, int, int]] = []
        self._tokenizer = _tokenize
        self._signature: tuple | None = None

    # -- build -----------------------------------------------------------
    def build(self, rows: Iterable[tuple[int, int, int, str, int, int]], signature: tuple | None = None) -> None:
        """``rows`` are ``(chunk_id, record_id, ordinal, text, start_char, end_char)``."""
        from rank_bm25 import BM25Okapi

        chunk_rows = list(rows)
        with self._lock:
            self._chunks = chunk_rows
            self._signature = signature
            if not chunk_rows:
                self._bm25 = None
                return
            corpus = [self._tokenizer(text) for (_cid, _rid, _o, text, _s, _e) in chunk_rows]
            self._bm25 = BM25Okapi(corpus, k1=K1, b=B)
        logger.info("built bm25 index chunks=%d", len(chunk_rows))

    def invalidate(self) -> None:
        with self._lock:
            self._bm25 = None
            self._chunks = []
            self._signature = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    # -- query -----------------------------------------------------------
    def search(self, query: str, top_k: int = 20, allowed_records: set[int] | None = None) -> list[LexicalHit]:
        with self._lock:
            bm25 = self._bm25
            chunks = list(self._chunks)
        if bm25 is None or not chunks:
            return []
        tokens = self._tokenizer(query)
        if not tokens:
            return []
        scores = np.asarray(bm25.get_scores(tokens), dtype=float)

        order = np.argsort(-scores)
        hits: list[LexicalHit] = []
        rank = 0
        for idx in order:
            chunk_id, record_id, ordinal, text, start_char, end_char = chunks[idx]
            if allowed_records is not None and record_id not in allowed_records:
                continue
            if scores[idx] <= 0.0:
                continue
            rank += 1
            hits.append(
                LexicalHit(
                    chunk_id=chunk_id,
                    record_id=record_id,
                    ordinal=ordinal,
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping intra-word hyphens and dots for refs like 20-45-71."""
    import re

    return re.findall(r"[a-z0-9][a-z0-9._'-]*", text.lower())


_default_index: Bm25Index | None = None
_default_lock = threading.Lock()


def get_default_index() -> Bm25Index:
    global _default_index
    if _default_index is None:
        with _default_lock:
            if _default_index is None:
                _default_index = Bm25Index()
    return _default_index


def reset_default_index() -> None:
    global _default_index
    with _default_lock:
        _default_index = None

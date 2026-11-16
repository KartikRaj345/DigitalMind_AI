"""Semantic embeddings with sentence-transformers + FAISS.

Optional: falls back to a deterministic hashing embedder when
``requirements-ml.txt`` is not installed. The fallback keeps the *pipeline* testable
(semantic channel exists, ranks are computed, scores fuse) without pretending to be a
good model - it is labelled ``fallback`` everywhere it surfaces, and the retrieval
evaluation reports the two channels separately so the degradation is visible.

The FAISS index is kept in memory and rebuilt from the ``embeddings`` table. Deleting a
record removes its rows, so the next rebuild cannot resurrect them.
"""

from __future__ import annotations

import struct
import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)

FALLBACK_DIM = 256

# Cosine floor for the semantic channel.
#
# Without it, a normalized embedding index will happily return the k nearest chunks for
# *any* query, including one about a topic that appears nowhere in the corpus. Those hits
# then suppress the "no matching discussion was found" wording, which is exactly the
# behaviour this project refuses: a search that always finds something cannot report an
# absence honestly.
#
# Measured on all-MiniLM-L6-v2, unrelated English sentence pairs score below ~0.15 and
# genuine topical matches sit above ~0.30. 0.20 sits in the gap.
MIN_SEMANTIC_SIMILARITY = 0.20


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


def sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class SemanticHit:
    chunk_id: int
    record_id: int
    score: float
    rank: int

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "record_id": self.record_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "retriever": "semantic",
        }


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


class HashingEmbedder:
    """Deterministic bag-of-ngrams hashing embedder.

    Not a semantic model: it captures lexical overlap in a dense form. Used so the hybrid
    fusion code path is exercised without torch. Always reported as ``fallback``.
    """

    name = "fallback:hashing-ngram-v1"
    is_fallback = True

    def encode(self, texts: list[str]) -> np.ndarray:
        import re

        out = np.zeros((len(texts), FALLBACK_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for g in grams:
                h = hash_g(g)
                out[i, h % FALLBACK_DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def hash_g(token: str) -> int:
    """Stable across processes (unlike Python's salted ``hash``)."""
    h = 1469598103934665603
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 (or any sentence-transformers model), CPU, fully local."""

    is_fallback = False

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class TransformerEmbedder:
    """Mean-pooled transformer encoder - the same arithmetic as sentence-transformers.

    Needed because ``sentence_transformers`` imports ``datasets``, which imports
    ``pyarrow``; on locked-down Windows machines an application-control policy can block
    that DLL, and the import fails even though the model itself is perfectly usable. Going
    straight to ``transformers.AutoModel`` plus mean pooling with attention masking
    reproduces the sentence-embedding recipe without the dependency chain.

    Used only as a second choice: if ``sentence_transformers`` imports cleanly, that path
    wins, because it is the tested implementation.
    """

    is_fallback = False

    def __init__(self, model_name: str, max_length: int = 256) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self.name = model_name
        self.max_length = max_length
        self.dim = int(self._model.config.hidden_size)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)


_embedder = None
_embedder_lock = threading.Lock()


def transformers_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def get_embedder(model_name: str):
    """Return the best available embedder; never raises.

    Preference order: ``sentence_transformers`` (tested implementation) ->
    ``transformers`` with mean pooling (same maths, no ``datasets`` import) -> a hashing
    fallback that is always reported as a fallback.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        if sentence_transformers_available():
            try:
                _embedder = SentenceTransformerEmbedder(model_name)
                logger.info("loaded sentence-transformers model=%s", model_name)
                return _embedder
            except Exception as exc:
                logger.warning(
                    "sentence-transformers unavailable (%s: %s); trying transformers directly",
                    type(exc).__name__,
                    str(exc)[:120],
                )
        if transformers_available():
            try:
                _embedder = TransformerEmbedder(model_name)
                logger.info(
                    "loaded transformers embedder model=%s dim=%d", model_name, _embedder.dim
                )
                return _embedder
            except Exception as exc:
                logger.warning(
                    "transformers embedder unavailable (%s); using hashing fallback",
                    type(exc).__name__,
                )
        _embedder = HashingEmbedder()
        logger.warning(
            "using hashing embedder fallback dim=%d - this is NOT a semantic model", FALLBACK_DIM
        )
        return _embedder


def reset_embedder() -> None:
    global _embedder
    with _embedder_lock:
        _embedder = None


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class SemanticIndex:
    """In-memory vector index over chunks.

    Uses FAISS ``IndexFlatIP`` on normalized vectors when available, and an exact numpy
    dot-product scan otherwise. Both are exact searches; FAISS just does it faster, so
    recall is identical and the fallback is not a quality regression.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vectors: np.ndarray | None = None
        self._rows: list[tuple[int, int]] = []  # (chunk_id, record_id)
        self._faiss_index = None
        self._model_name = ""

    def build(self, rows: Iterable[tuple[int, int, bytes]], model_name: str) -> int:
        """``rows`` are ``(chunk_id, record_id, packed_float32_vector)``."""
        materialized = list(rows)
        with self._lock:
            self._rows = []
            if not materialized:
                self._vectors = None
                self._faiss_index = None
                return 0
            dim = len(materialized[0][2]) // 4
            vecs = np.vstack(
                [np.frombuffer(v, dtype=np.float32, count=dim) for (_c, _r, v) in materialized]
            ).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
            self._vectors = vecs
            self._rows = [(c, r) for (c, r, _v) in materialized]
            self._model_name = model_name
            self._faiss_index = None
            if faiss_available():
                import faiss

                index = faiss.IndexFlatIP(dim)
                index.add(vecs)
                self._faiss_index = index
        logger.info(
            "built semantic index chunks=%d dim=%d backend=%s",
            len(materialized),
            dim,
            "faiss" if faiss_available() else "numpy",
        )
        return len(materialized)

    def invalidate(self) -> None:
        with self._lock:
            self._vectors = None
            self._rows = []
            self._faiss_index = None

    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_query(self, query: str, model_name: str) -> np.ndarray:
        embedder = get_embedder(model_name)
        return np.asarray(embedder.encode([query])[0], dtype=np.float32)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 20,
        allowed_records: set[int] | None = None,
        min_similarity: float = MIN_SEMANTIC_SIMILARITY,
    ) -> list[SemanticHit]:
        with self._lock:
            vectors = self._vectors
            rows = list(self._rows)
            faiss_index = self._faiss_index
        if vectors is None or not rows:
            return []
        if query_vec.shape[0] != vectors.shape[1]:
            logger.warning(
                "query dim %d != index dim %d; skipping semantic channel",
                query_vec.shape[0],
                vectors.shape[1],
            )
            return []
        q = query_vec / max(1e-9, float(np.linalg.norm(query_vec)))

        limit = min(top_k * 4, len(rows)) if allowed_records is not None else top_k
        if faiss_index is not None:
            scores, idxs = faiss_index.search(q.reshape(1, -1), limit)
            scores = scores[0]
            idxs = idxs[0]
        else:
            sims = vectors @ q
            order = np.argsort(-sims)[:limit]
            scores = sims[order]
            idxs = order

        hits: list[SemanticHit] = []
        rank = 0
        for score, idx in zip(scores, idxs):
            idx = int(idx)
            if idx < 0 or idx >= len(rows):
                continue
            chunk_id, record_id = rows[idx]
            if allowed_records is not None and record_id not in allowed_records:
                continue
            if float(score) < min_similarity:
                continue
            rank += 1
            hits.append(SemanticHit(chunk_id=chunk_id, record_id=record_id,
                                    score=float(score), rank=rank))
            if len(hits) >= top_k:
                break
        return hits


_default_index: SemanticIndex | None = None
_default_lock = threading.Lock()


def get_default_index() -> SemanticIndex:
    global _default_index
    if _default_index is None:
        with _default_lock:
            if _default_index is None:
                _default_index = SemanticIndex()
    return _default_index


def reset_default_index() -> None:
    global _default_index
    with _default_lock:
        _default_index = None


def pack_vector(vec: np.ndarray) -> bytes:
    return struct.pack(f"<{vec.shape[0]}f", *vec.astype(np.float32).tolist())


def unpack_vector(blob: bytes) -> np.ndarray:
    count = len(blob) // 4
    return np.asarray(struct.unpack(f"<{count}f", blob[: count * 4]), dtype=np.float32)

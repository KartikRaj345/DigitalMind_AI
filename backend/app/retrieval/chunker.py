"""Record chunking.

Chunks carry exact character offsets into the record body. That is the mechanism behind
every highlight in the UI and every citation in a QA answer: we never quote a chunk and
have to guess which part of the record it came from.

Strategy: split on paragraph boundaries first (emails and notes have natural ones), then
fall back to sentence packing when a paragraph is too long. Overlap is deliberate - one
sentence of overlap prevents a fact that straddles a boundary from being unreachable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHARS = 600
MAX_CHARS = 900
OVERLAP_SENTENCES = 1

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    ordinal: int
    text: str
    start_char: int
    end_char: int

    def as_tuple(self) -> tuple[int, str, int, int]:
        return (self.ordinal, self.text, self.start_char, self.end_char)


def chunk_text(text: str, target: int = TARGET_CHARS, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Split ``text`` into overlapping chunks with exact offsets."""
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    for para_start, para_end in _paragraph_spans(text):
        chunks.extend(_chunk_span(text, para_start, para_end, target, max_chars))

    # Merge degenerate single-character chunks into their neighbour to avoid noise.
    cleaned: list[Chunk] = []
    for chunk in chunks:
        if cleaned and chunk.text.strip() and len(chunk.text.strip()) < 2:
            cleaned[-1] = Chunk(
                cleaned[-1].ordinal,
                cleaned[-1].text,
                cleaned[-1].start_char,
                chunk.end_char,
            )
            continue
        cleaned.append(chunk)
    return cleaned


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for part in _PARAGRAPH_SPLIT.split(text):
        idx = text.find(part, pos)
        if idx < 0:  # pragma: no cover - defensive
            idx = pos
        stripped = part.strip()
        if stripped:
            start = idx + part.index(stripped)
            spans.append((start, start + len(stripped)))
        pos = idx + len(part)
    if not spans and text.strip():
        stripped = text.strip()
        start = text.index(stripped)
        spans.append((start, start + len(stripped)))
    return spans


def _chunk_span(
    text: str, start: int, end: int, target: int, max_chars: int
) -> list[Chunk]:
    segment = text[start:end]
    if len(segment) <= max_chars:
        return [Chunk(0, segment, start, end)]

    # Sentence-level packing with offsets preserved.
    sentences: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_SPLIT.finditer(segment):
        if match.start() > cursor:
            sentences.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(segment):
        sentences.append((cursor, len(segment)))

    out: list[tuple[int, int]] = []
    buf_start: int | None = None
    buf_end = 0
    for s_start, s_end in sentences:
        if buf_start is None:
            buf_start, buf_end = s_start, s_end
            continue
        if buf_end - buf_start + (s_end - s_start) > max_chars:
            out.append((buf_start, buf_end))
            # Overlap: rewind one sentence so a boundary-straddling fact stays reachable.
            idx = sentences.index((s_start, s_end))
            back = max(0, idx - OVERLAP_SENTENCES)
            buf_start, buf_end = sentences[back][0], s_end
        else:
            buf_end = s_end
    if buf_start is not None:
        out.append((buf_start, buf_end))

    return [
        Chunk(0, segment[a:b], start + a, start + b) for a, b in out if segment[a:b].strip()
    ]


def renumber(chunks: list[Chunk]) -> list[Chunk]:
    return [Chunk(i, c.text, c.start_char, c.end_char) for i, c in enumerate(chunks)]


def chunk_for_span(chunks: list[Chunk], start: int, end: int) -> Chunk | None:
    """The chunk that best covers a span, so a citation can be expanded to context."""
    best: Chunk | None = None
    best_overlap = 0
    for chunk in chunks:
        overlap = min(chunk.end_char, end) - max(chunk.start_char, start)
        if overlap > best_overlap:
            best, best_overlap = chunk, overlap
    return best

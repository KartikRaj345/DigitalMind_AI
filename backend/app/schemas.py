"""Pydantic response/request schemas for the API."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class SpanOut(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    extractor: str
    normalized: str | None = None
    entity_id: int | None = None
    entity_name: str | None = None


class EventOut(BaseModel):
    id: int | None = None
    title: str
    description: str = ""
    kind: str
    certainty: str
    start_at: dt.datetime | None = None
    end_at: dt.datetime | None = None
    precision: str
    raw_expression: str = ""
    start_char: int | None = None
    end_char: int | None = None
    confidence: float
    extractor: str
    note: str = ""


class EvidenceOut(BaseModel):
    start_char: int | None = None
    end_char: int | None = None
    text: str | None = None
    note: str | None = None
    polarity: str | None = None
    weight: float | None = None


class IndicatorOut(BaseModel):
    label: str
    display: str
    present: bool
    score: float
    threshold: float
    methods: dict[str, float]
    explanation: str
    evidence: list[EvidenceOut]
    calibrated: bool


class RecordSummary(BaseModel):
    id: int
    source_type: str
    title: str
    occurred_at: dt.datetime | None
    imported_at: dt.datetime
    origin: str
    participants: list[str]
    char_count: int
    risk_labels: list[str] = []
    severity: str = "none"
    snippet: str = ""


class RecordDetail(BaseModel):
    id: int
    source_type: str
    title: str
    body: str
    occurred_at: dt.datetime | None
    imported_at: dt.datetime
    origin: str
    external_id: str | None
    participants: list[str]
    extra: dict[str, Any]
    spans: list[SpanOut]
    events: list[EventOut]
    indicators: list[IndicatorOut]
    overall_estimate: dict[str, Any]
    entities: list[dict[str, Any]]
    backreferences: list[dict[str, Any]] = []
    chunks: list[dict[str, int]] = []


class RecordListOut(BaseModel):
    total: int
    items: list[RecordSummary]


# ---------------------------------------------------------------------------
# Search / QA
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    record_id: int
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    score: float
    rank: int
    lexical_rank: int | None
    semantic_rank: int | None
    agreement: str
    title: str | None = None
    source_type: str | None = None
    occurred_at: dt.datetime | None = None


class SearchOut(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]
    channels: dict[str, int]
    degraded: list[str]
    explanation: str
    total_records: int


class CitationOut(BaseModel):
    record_id: int
    source_type: str
    title: str
    occurred_at: str | None
    start_char: int
    end_char: int
    quote: str
    retriever: str
    score: float | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    mode: Literal["hybrid", "lexical", "semantic"] = "hybrid"
    top_k: int = Field(default=8, ge=1, le=50)


class AskOut(BaseModel):
    question: str
    intent: str
    answer: str
    citations: list[CitationOut]
    facts: list[dict[str, Any]]
    interpretation: list[str]
    coverage: dict[str, Any]
    method: dict[str, Any]
    caveat: str
    no_match: bool
    confidence: float
    confidence_basis: str


# ---------------------------------------------------------------------------
# Graph / timeline / insights
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    id: str
    kind: str
    label: str
    entity_type: str | None = None
    mention_count: int | None = None
    source_type: str | None = None
    occurred_at: str | None = None
    record_id: int | None = None
    resolution_confidence: float | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: float
    record_id: int | None
    start_char: int | None
    end_char: int | None
    method: str
    weight: float


class GraphOut(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    centrality: dict[str, dict[str, float]]
    explanation: str


class TimelineItemOut(BaseModel):
    record_id: int
    title: str
    source_type: str
    lane: str
    kind: str
    start_at: dt.datetime | None
    end_at: dt.datetime | None
    sort_key: str
    certainty: str
    precision: str
    raw_expression: str
    confidence: float
    explanation: str
    start_char: int | None
    end_char: int | None
    has_risk: bool
    risk_labels: list[str]


class TimelineOut(BaseModel):
    items: list[TimelineItemOut]
    summary: dict[str, Any]


class InsightOut(BaseModel):
    id: int
    kind: str
    severity: str
    title: str
    statement: str
    facts: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    confidence: float
    confidence_basis: str
    status: str
    user_note: str | None
    record_ids: list[int]
    entity_ids: list[int]
    created_at: dt.datetime


class InsightUpdate(BaseModel):
    status: Literal["active", "dismissed", "corrected"] | None = None
    user_note: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Settings / import
# ---------------------------------------------------------------------------


class SettingsOut(BaseModel):
    retrieval_mode: str
    min_evidence_confidence: float
    risk_models: dict[str, Any]
    spacy: dict[str, Any]
    embeddings: dict[str, Any]
    privacy: dict[str, Any]
    data_sources: list[dict[str, Any]]


class SettingsUpdate(BaseModel):
    retrieval_mode: Literal["lexical", "semantic", "hybrid"] | None = None
    min_evidence_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    allow_external_llm: bool | None = None


class ImportResult(BaseModel):
    inserted: int
    skipped: int
    record_ids: list[int]
    files: list[dict[str, Any]] = []
    errors: list[str] = []


class HealthOut(BaseModel):
    status: str
    version: str
    records: int
    spacy_available: bool
    transformers_available: bool
    sentence_transformers_available: bool
    faiss_available: bool
    llm_enabled: bool
    external_transmission: bool

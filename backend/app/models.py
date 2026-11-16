"""SQLAlchemy ORM models.

Design notes
------------
* Every derived row (mention, relation, event, risk assessment, index entry) carries a
  foreign key back to its ``Record`` with ``ON DELETE CASCADE``. The privacy guarantee
  "deleting a record deletes everything derived from it" is therefore enforced by the
  database, and re-checked explicitly in :mod:`app.privacy`.
* Analysis output always stores the *character span* it came from, so the UI can point
  back at the exact words rather than the whole document.
* ``confidence`` is stored as a float in [0, 1] together with ``extractor``/``method`` so
  the UI can always say *which* model produced a claim, not just how sure it was.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class Record(Base):
    """One imported piece of digital history (email, message, doc, note, event, task)."""

    __tablename__ = "records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    body: Mapped[str] = mapped_column(Text, default="")

    # The timestamp the source system asserts ("recorded fact").
    occurred_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # When DigitalMind ingested it. Never used as a substitute for occurred_at.
    imported_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Identifies the record in the integration it came from ("demo", "upload:notes.md").
    origin: Mapped[str] = mapped_column(String(128), default="demo", index=True)
    external_id: Mapped[str | None] = mapped_column(String(256))

    participants: Mapped[list[str]] = mapped_column(JSON, default=list)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    content_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    mentions: Mapped[list["EntityMention"]] = relationship(
        back_populates="record", cascade="all, delete-orphan", passive_deletes=True
    )
    events: Mapped[list["Event"]] = relationship(
        back_populates="record", cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="record", cascade="all, delete-orphan", passive_deletes=True
    )
    assessments: Mapped[list["RiskAssessment"]] = relationship(
        back_populates="record", cascade="all, delete-orphan", passive_deletes=True
    )
    __table_args__ = (UniqueConstraint("origin", "external_id", name="uq_record_origin_ext"),)

    @property
    def display_title(self) -> str:
        return self.title or self.body.strip().splitlines()[0][:80] if self.body.strip() else "Untitled"

    def span_text(self, start: int, end: int) -> str:
        """Return the exact source substring a span points at."""
        if start is None or end is None or start < 0 or end > len(self.body):
            return ""
        return self.body[start:end]


# ---------------------------------------------------------------------------
# Entities and the knowledge graph
# ---------------------------------------------------------------------------


class Entity(Base):
    """A canonical entity (person, org, product, ...) after entity resolution."""

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), index=True)
    normalized_name: Mapped[str] = mapped_column(String(256), index=True)
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Aggregate confidence that this cluster is a single real-world entity.
    resolution_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    mentions: Mapped[list["EntityMention"]] = relationship(back_populates="entity")


class EntityMention(Base):
    """A single extracted mention of an entity, tied to a source character span."""

    __tablename__ = "entity_mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    record_id: Mapped[int] = mapped_column(
        ForeignKey("records.id", ondelete="CASCADE"), index=True
    )
    entity_id: Mapped[int | None] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(String(512))
    label: Mapped[str] = mapped_column(String(32))
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    extractor: Mapped[str] = mapped_column(String(32), default="rules")

    record: Mapped[Record] = relationship(back_populates="mentions")
    entity: Mapped[Entity | None] = relationship(back_populates="mentions")


class EntityAlias(Base):
    """Surface form that was merged into an entity, kept for explainability."""

    __tablename__ = "entity_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    alias: Mapped[str] = mapped_column(String(512))
    normalized_alias: Mapped[str] = mapped_column(String(512), index=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(32), default="merge")


class Relation(Base):
    """An edge in the knowledge graph, always evidenced by a record + span."""

    __tablename__ = "relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    target_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    # 'mentioned_in', 'sent', 'participant', 'about', ...
    relation_type: Mapped[str] = mapped_column(String(48), index=True)
    record_id: Mapped[int | None] = mapped_column(
        ForeignKey("records.id", ondelete="CASCADE"), index=True
    )
    start_char: Mapped[int | None] = mapped_column(Integer)
    end_char: Mapped[int | None] = mapped_column(Integer)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    method: Mapped[str] = mapped_column(String(32), default="co_mention")

    __table_args__ = (
        UniqueConstraint(
            "source_entity_id",
            "target_entity_id",
            "relation_type",
            "record_id",
            name="uq_relation",
        ),
    )


# ---------------------------------------------------------------------------
# Events and timeline
# ---------------------------------------------------------------------------


class Event(Base):
    """An event extracted from a record.

    ``certainty`` is the important field:

    * ``recorded``  - the source system gave us a timestamp (``occurred_at`` / ICS DTSTART)
    * ``explicit``  - the text names an absolute date ("on 4 March 2026")
    * ``inferred``  - the text only gives relative ordering ("last week", "yesterday")

    The timeline renders the first two as facts and the third as an estimate with a range.
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    record_id: Mapped[int] = mapped_column(
        ForeignKey("records.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[str] = mapped_column(Text, default="")
    # How the time was arrived at, shown verbatim next to inferred items.
    note: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(32), default="event")
    certainty: Mapped[str] = mapped_column(String(16), default="explicit", index=True)

    # Normalised interval. For 'inferred' events both ends may be set (a range).
    start_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    end_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    # Finest precision we can defend: 'minute' | 'day' | 'week' | 'month' | 'unknown'
    precision: Mapped[str] = mapped_column(String(16), default="unknown")

    # The literal source text ("as we discussed last week") and where it was.
    raw_expression: Mapped[str] = mapped_column(String(256), default="")
    start_char: Mapped[int | None] = mapped_column(Integer)
    end_char: Mapped[int | None] = mapped_column(Integer)

    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    extractor: Mapped[str] = mapped_column(String(32), default="rules")
    participants: Mapped[list[str]] = mapped_column(JSON, default=list)

    record: Mapped[Record] = relationship(back_populates="events")


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class RiskAssessment(Base):
    """Per-record, per-indicator risk evidence. Never a bare score.

    ``label`` is one of ``urgency | threat_language | emotional_pressure |
    payment_request``. ``present`` is the flagged/not decision; ``score`` is the
    calibrated probability behind it; ``explanation`` is shown verbatim in the UI.
    """

    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    record_id: Mapped[int] = mapped_column(
        ForeignKey("records.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(String(32), index=True)
    present: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.5)
    method: Mapped[str] = mapped_column(String(48), default="rules")
    explanation: Mapped[str] = mapped_column(Text, default="")
    # JSON list of {start_char, end_char, text, matched_pattern}
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    calibrated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    record: Mapped[Record] = relationship(back_populates="assessments")
    __table_args__ = (
        UniqueConstraint("record_id", "label", "method", name="uq_assessment"),
    )


class Insight(Base):
    """A user-facing conclusion. Always inspectable, dismissible and correctable.

    ``evidence`` is a JSON list of {record_id, start_char, end_char, quote} so the UI can
    render "why do you think this?" as a list of clickable citations.
    """

    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="info")  # info | notice | warning
    title: Mapped[str] = mapped_column(String(512))
    # Model interpretation, kept distinct from extracted facts.
    statement: Mapped[str] = mapped_column(Text, default="")
    # Facts the statement rests on (verbatim quotes + spans).
    facts: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    confidence_basis: Mapped[str] = mapped_column(String(64), default="")
    # active | dismissed | corrected
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    user_note: Mapped[str | None] = mapped_column(Text)
    record_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    entity_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# ---------------------------------------------------------------------------
# Retrieval index
# ---------------------------------------------------------------------------


class Chunk(Base):
    """A retrievable unit of text with a character offset into its record.

    Offsets are what let search results and QA answers highlight the exact words.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    record_id: Mapped[int] = mapped_column(
        ForeignKey("records.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    start_char: Mapped[int] = mapped_column(Integer, default=0)
    end_char: Mapped[int] = mapped_column(Integer, default=0)
    # 'lexical' (BM25) always; 'semantic' when the embedding index is built.
    indexed_lexical: Mapped[bool] = mapped_column(Boolean, default=False)
    indexed_semantic: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding_row: Mapped[int | None] = mapped_column(Integer)

    record: Mapped[Record] = relationship(back_populates="chunks")
    __table_args__ = (
        UniqueConstraint("record_id", "ordinal", name="uq_chunk_ordinal"),
        Index("ix_chunk_lexical", "indexed_lexical"),
    )


class Embedding(Base):
    """Vector store for semantic retrieval (FAISS-backed when available)."""

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), index=True, unique=True
    )
    record_id: Mapped[int] = mapped_column(
        ForeignKey("records.id", ondelete="CASCADE"), index=True
    )
    model: Mapped[str] = mapped_column(String(128), default="")
    dim: Mapped[int] = mapped_column(Integer, default=0)
    # float32 little-endian blob; FAISS copies this into its own index.
    vector: Mapped[bytes] = mapped_column(LargeBinary)


# ---------------------------------------------------------------------------
# Settings / provenance
# ---------------------------------------------------------------------------


class AppSetting(Base):
    """Persisted user preferences and analysis configuration."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class DataSource(Base):
    """An import origin. Real integrations are declared but not configured."""

    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32), default="local")  # local | integration
    configured: Mapped[bool] = mapped_column(Boolean, default=False)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

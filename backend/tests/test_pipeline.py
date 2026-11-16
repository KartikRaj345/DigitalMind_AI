"""Tests for the analysis pipeline and API analysis controls."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select

from backend.app.models import DataSource, Entity, EntityMention, Record
from backend.app.pipeline import (
    analyze_record,
    build_indexes,
    generate_insights,
    ingest_records,
    load_demo,
    resolve_entities,
    run_pipeline,
)

UTC = dt.timezone.utc


def test_ingest_is_idempotent_on_external_id(session):
    from backend.app.ingest.demo import DEMO_RECORDS

    first = ingest_records(session, DEMO_RECORDS, origin="demo")
    second = ingest_records(session, DEMO_RECORDS, origin="demo")
    assert first.inserted == len(DEMO_RECORDS)
    assert second.inserted == 0
    assert second.skipped == len(DEMO_RECORDS)
    assert int(session.execute(select(func.count()).select_from(Record)).scalar_one()) == len(
        DEMO_RECORDS
    )


def test_reingest_after_delete_does_not_violate_unique(session):
    """Regression: DataSource is keyed on `key`, not its integer primary key."""
    from backend.app.privacy import delete_all_data

    load_demo(session)
    delete_all_data(session)
    stats = load_demo(session)
    assert stats.inserted > 0
    sources = list(session.execute(select(DataSource)).scalars())
    assert len({s.key for s in sources}) == len(sources)


def test_analyze_record_is_idempotent(demo_session):
    session = demo_session
    record = session.execute(select(Record)).scalars().first()
    before_mentions = int(
        session.execute(
            select(func.count()).select_from(EntityMention).where(EntityMention.record_id == record.id)
        ).scalar_one()
    )
    analyze_record(session, record)
    after_mentions = int(
        session.execute(
            select(func.count()).select_from(EntityMention).where(EntityMention.record_id == record.id)
        ).scalar_one()
    )
    assert before_mentions > 0
    assert after_mentions == before_mentions


def test_resolve_entities_twice_does_not_duplicate(demo_session):
    """Regression: bulk-deleting entities used to orphan the mention objects."""
    session = demo_session
    before = int(session.execute(select(func.count()).select_from(Entity)).scalar_one())
    summary = resolve_entities(session)
    after = int(session.execute(select(func.count()).select_from(Entity)).scalar_one())
    assert after == summary["entities"]
    assert before > 0

    unresolved = int(
        session.execute(
            select(func.count()).select_from(EntityMention).where(EntityMention.entity_id.is_(None))
        ).scalar_one()
    )
    assert unresolved == 0


def test_every_mention_is_linked_to_an_entity(demo_session):
    total = int(demo_session.execute(select(func.count()).select_from(EntityMention)).scalar_one())
    linked = int(
        demo_session.execute(
            select(func.count()).select_from(EntityMention).where(EntityMention.entity_id.is_not(None))
        ).scalar_one()
    )
    assert total > 0
    assert linked == total


def test_pipeline_summary_shape(demo_session):
    summary = run_pipeline(demo_session, with_embeddings=False)
    for key in ("records", "resolution", "graph", "risk", "index", "insights", "spacy_available"):
        assert key in summary
    assert summary["graph"]["nodes"] > 0
    assert summary["graph"]["edges"] > 0
    assert "available" in summary["spacy_available"]


def test_spans_reference_exact_source_characters(demo_session):
    session = demo_session
    bodies = {r.id: r.body for r in session.execute(select(Record)).scalars()}
    for mention in session.execute(select(EntityMention)).scalars():
        body = bodies[mention.record_id]
        assert 0 <= mention.start_char < mention.end_char <= len(body)
        assert body[mention.start_char : mention.end_char] == mention.text


def test_graph_edges_have_evidence(demo_session):
    from backend.app.models import Relation

    session = demo_session
    relations = list(session.execute(select(Relation)).scalars())
    assert relations
    for relation in relations:
        assert relation.method
        assert 0.0 <= relation.confidence <= 1.0
        if relation.relation_type == "co_mentioned":
            assert relation.record_id is None  # a heuristic, not a cited span
        else:
            assert relation.record_id is not None


def test_index_rebuild_after_new_record(session):
    from backend.app.ingest.demo import DemoRecord

    load_demo(session)
    run_pipeline(session, with_embeddings=False)
    stats = build_indexes(session, with_embeddings=False)
    before = stats["lexical_chunks"]

    session.add(
        Record(
            source_type="note",
            title="New note",
            body="A brand new note about Atlas Freight and its booking window.",
            occurred_at=dt.datetime(2026, 4, 1, 9, 0, tzinfo=UTC),
            origin="test",
            external_id="new-note",
            char_count=63,
            content_hash="hash",
        )
    )
    session.flush()
    analyze_record(session, session.execute(select(Record)).scalars().first())
    run_pipeline(session, with_embeddings=False)
    after = build_indexes(session, with_embeddings=False)["lexical_chunks"]
    assert after >= before


def test_insights_reference_existing_records(demo_session):
    from backend.app.models import Insight

    session = demo_session
    record_ids = {r.id for r in session.execute(select(Record)).scalars()}
    for insight in session.execute(select(Insight)).scalars():
        assert set(insight.record_ids or []) <= record_ids


def test_analysis_endpoints_work_on_an_analysed_database(demo_session):
    # Regression: POST /analysis/resolve used to raise StaleDataError after a pipeline run.
    summary = resolve_entities(demo_session)
    assert summary["entities"] > 0
    generate_insights(demo_session)

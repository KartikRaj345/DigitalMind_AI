"""Tests for the synthetic dataset generator.

The dataset is a graded deliverable, so its *properties* are tested, not just the fact that
it runs. The leakage test in particular guards a bug this file was written to catch: two
templates drawing the same entity/topic/date combination produced byte-identical bodies,
which put duplicates on both sides of the train/test split.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pytest

from backend.app.datasets import load_split, split_records
from backend.app.ingest.generator import generate_corpus, split_records as split_by_scenario


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n=320, seed=20260310)


def test_meets_the_brief_size_requirement(corpus):
    assert len(corpus) >= 300


def test_generation_is_deterministic():
    a = generate_corpus(n=80, seed=20260310)
    b = generate_corpus(n=80, seed=20260310)
    assert [r.body for r in a] == [r.body for r in b]
    assert [r.risk_labels for r in a] == [r.risk_labels for r in b]


def test_a_different_seed_changes_the_corpus():
    a = generate_corpus(n=80, seed=20260310)
    b = generate_corpus(n=80, seed=999)
    assert [r.body for r in a] != [r.body for r in b]


def test_no_duplicate_bodies(corpus):
    """Regression: identical bodies put the same string in train and test."""
    counts = collections.Counter(r.body.strip() for r in corpus)
    duplicates = {body: n for body, n in counts.items() if n > 1}
    assert duplicates == {}, f"{len(duplicates)} duplicate body/ies in the corpus"


def test_no_train_test_leakage(corpus):
    splits = split_by_scenario(corpus, seed=20260310)
    train_bodies = {r.body.strip() for r in splits["train"]}
    test_bodies = {r.body.strip() for r in splits["test"]}
    dev_bodies = {r.body.strip() for r in splits["dev"]}
    assert train_bodies & test_bodies == set()
    assert train_bodies & dev_bodies == set()


def test_gold_span_offsets_are_exact(corpus):
    """Every gold span must be the substring its offsets claim."""
    for record in corpus:
        for span in record.entity_spans:
            assert 0 <= span.start < span.end <= len(record.body), record.id
            assert record.body[span.start : span.end] == span.text, (
                record.id,
                span.text,
            )


def test_gold_spans_do_not_overlap(corpus):
    for record in corpus:
        spans = sorted(record.entity_spans, key=lambda s: s.start)
        for a, b in zip(spans, spans[1:]):
            assert a.end <= b.start, f"{record.id}: {a.text!r} overlaps {b.text!r}"


def test_most_records_carry_at_least_one_gold_span(corpus):
    with_spans = sum(1 for r in corpus if r.entity_spans)
    # Records whose body genuinely contains no annotatable entity are allowed, but they
    # must be a small minority or the NER evaluation is measuring an empty gold set.
    assert with_spans / len(corpus) >= 0.80, f"only {with_spans}/{len(corpus)} have gold spans"


def test_most_records_carry_at_least_one_risk_label(corpus):
    labelled = sum(1 for r in corpus if r.risk_labels)
    assert 0.10 <= labelled / len(corpus) <= 0.35, (
        f"positive rate {labelled / len(corpus):.2f} is outside the intended band; "
        "an extreme rate makes precision or recall untestable"
    )


def test_benign_lookalikes_exist_and_are_unlabelled(corpus):
    """The false-positive control.

    Without records that carry the same surface features as the attack records but carry
    no label, a high F1 is meaningless: nothing would ever be a false positive.
    """
    lookalikes = [r for r in corpus if r.benign_lookalike]
    assert len(lookalikes) >= 30
    for record in lookalikes:
        assert record.risk_labels == []

    risky = [r for r in corpus if r.risk_labels]
    # The look-alikes must actually share the surface features that drive the risky
    # records. These are the four that matter: a currency amount, account vocabulary,
    # the word "closed", and a numeric time expression.
    shared_tokens = ["$", "account", "closed"]
    risky_text = " ".join(r.body for r in risky).lower()
    look_text = " ".join(r.body for r in lookalikes).lower()
    for token in shared_tokens:
        assert token in risky_text, f"risky records lack {token!r}"
        assert token in look_text, f"look-alikes lack the surface feature {token!r}"

    # Both sets must contain a numeric time expression, so neither can be separated by
    # "has a number in it".
    time_pattern = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,3}\s+minutes?\b")
    assert time_pattern.search(risky_text)
    assert time_pattern.search(look_text)


def test_risky_records_are_multi_label(corpus):
    risky = [r for r in corpus if r.risk_labels]
    multi = [r for r in risky if len(r.risk_labels) >= 2]
    assert len(multi) / len(risky) >= 0.5, "multi-label structure is not being exercised"


def test_backreference_records_exist(corpus):
    backrefs = [r for r in corpus if r.backreference]
    assert len(backrefs) >= 10
    for record in backrefs:
        assert "as we discussed" in record.body.lower()


def test_splits_are_stratified_by_scenario(corpus):
    splits = split_by_scenario(corpus, seed=20260310)
    train_scenarios = {r.scenario for r in splits["train"]}
    test_scenarios = {r.scenario for r in splits["test"]}
    assert train_scenarios == test_scenarios
    # Each split must retain the risky/benign mix, or dev is useless for threshold tuning.
    for name, group in splits.items():
        risky = sum(1 for r in group if r.risk_labels)
        assert 0 < risky < len(group), f"{name} has no risky records"


def test_split_sizes_are_roughly_seventy_fifteen_fifteen(corpus):
    splits = split_by_scenario(corpus, seed=20260310)
    total = len(corpus)
    assert abs(len(splits["train"]) / total - 0.70) < 0.06
    assert abs(len(splits["dev"]) / total - 0.15) < 0.06
    assert abs(len(splits["test"]) / total - 0.15) < 0.06
    assert len(splits["train"]) + len(splits["dev"]) + len(splits["test"]) == total


def test_all_six_source_types_are_represented(corpus):
    types = {r.source_type for r in corpus}
    assert types == {"email", "message", "document", "note", "calendar", "task"}


def test_records_have_timestamps_in_a_sane_window(corpus):
    times = sorted(r.occurred_at for r in corpus)
    assert times[0] < times[-1]
    span_days = (times[-1] - times[0]).days
    assert 60 <= span_days <= 180, f"date span of {span_days} days looks wrong"


def test_all_records_have_content():
    corpus = generate_corpus(n=320, seed=20260310)
    for record in corpus:
        assert record.body.strip(), record.id
        assert record.title.strip(), record.id


def test_unique_ids(corpus):
    ids = [r.id for r in corpus]
    assert len(ids) == len(set(ids))


def test_written_files_round_trip(tmp_path: Path, corpus):
    from backend.app.ingest.generator import write_dataset

    write_dataset(tmp_path, n=320, seed=20260310)
    split = load_split("test", directory=tmp_path)
    assert len(split) == 48
    assert (tmp_path / "dataset_meta.json").exists()
    meta = json.loads((tmp_path / "dataset_meta.json").read_text(encoding="utf-8"))
    assert meta["total"] == 320
    assert sum(meta["counts"].values()) == 320

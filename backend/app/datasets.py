"""Shared dataset loading for training and evaluation.

The synthetic corpus is generated from a seed; the real public corpus is downloaded once
and cached. Everything returns the same shape so a script can train on synthetic and
evaluate on real without branching.

Splits are stratified by scenario (see :func:`split_records`) so dev is useful for
threshold selection.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .ingest.generator import GoldRecord, generate_corpus, split_records, write_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_DIR = DATA_DIR / "datasets"
SYNTHETIC_DIR = DATASET_DIR / "synthetic"
PUBLIC_DIR = DATASET_DIR / "public"

RISK_LABELS = ("urgency", "threat_language", "emotional_pressure", "payment_request")


@dataclass
class Split:
    ids: list[str]
    texts: list[str]
    labels: np.ndarray  # (n, 4) float32 in {0,1}
    records: list[GoldRecord] = field(default_factory=list)
    name: str = ""

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def positives(self) -> dict[str, int]:
        return {label: int(self.labels[:, i].sum()) for i, label in enumerate(RISK_LABELS)}

    @property
    def positive_rate(self) -> float:
        if self.labels.size == 0:
            return 0.0
        return float((self.labels.sum(axis=1) > 0).mean())


def ensure_synthetic(seed: int = 20260310, n: int = 320, force: bool = False) -> Path:
    """Generate the synthetic corpus if it is missing."""
    if not force and (SYNTHETIC_DIR / "all.jsonl").exists():
        return SYNTHETIC_DIR
    write_dataset(SYNTHETIC_DIR, n=n, seed=seed)
    return SYNTHETIC_DIR


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _to_record(row: dict[str, Any]) -> GoldRecord:
    from .ingest.generator import GoldSpan

    spans = [
        GoldSpan(s["text"], s["label"], int(s["start"]), int(s["end"]))
        for s in row.get("entity_spans", [])
    ]
    return GoldRecord(
        id=row["id"],
        source_type=row.get("source_type", "message"),
        title=row.get("title", ""),
        body=row.get("body", ""),
        occurred_at=_parse_dt(row.get("occurred_at")),
        participants=row.get("participants", []),
        risk_labels=row.get("risk_labels", []),
        entity_spans=spans,
        backreference=row.get("backreference", False),
        scenario=row.get("scenario", ""),
        benign_lookalike=row.get("benign_lookalike", False),
    )


def _parse_dt(value: str | None):
    if not value:
        return None
    from dateutil import parser as dateparser

    try:
        return dateparser.parse(value)
    except (ValueError, TypeError, OverflowError):
        return None


def load_split(name: str, directory: Path = SYNTHETIC_DIR) -> Split:
    rows = load_jsonl(directory / f"{name}.jsonl")
    records = [_to_record(r) for r in rows]
    return _build_split(name, records)


def _build_split(name: str, records: Sequence[GoldRecord]) -> Split:
    ids = [r.id for r in records]
    texts = [r.body for r in records]
    labels = np.zeros((len(records), len(RISK_LABELS)), dtype=np.float32)
    index = {label: i for i, label in enumerate(RISK_LABELS)}
    for i, rec in enumerate(records):
        for label in rec.risk_labels:
            if label in index:
                labels[i, index[label]] = 1.0
    return Split(ids=ids, texts=texts, labels=labels, records=list(records), name=name)


def load_all_synthetic() -> dict[str, Split]:
    ensure_synthetic()
    return {name: load_split(name) for name in ("train", "dev", "test")}


# ---------------------------------------------------------------------------
# Real public dataset
# ---------------------------------------------------------------------------

# SMS Spam Collection, UCI: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
# Licence: CC BY 4.0 (attribution required; we cite it in reports/).
#
# Why this one: it is a real, human-labelled, well-known spam corpus, available as a
# single CSV, small enough to train on CPU, and its licence permits redistribution with
# attribution. It gives us *real* wording for the pressure tactics our synthetic
# templates imitate, which is exactly the check the synthetic data cannot provide.
SMS_SPAM = {
    "key": "sms_spam_collection",
    "name": "SMS Spam Collection v1",
    "source_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection",
    "licence": "CC BY 4.0",
    "citation": (
        "Almeida, A., Gomez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the study "
        "of SMS spam filtering. Proceedings of the 2011 European Conference on Machine "
        "Learning and Principles of Data Mining (ECML PKDD 2011), LNCS 6913, pp. 307-314."
    ),
}

HAM_LABEL = 0
SPAM_LABEL = 1


def _spans_for_labels(body: str) -> list[dict[str, Any]]:
    """Map spam words onto our four indicators so the real corpus is multi-label.

    This mapping is deliberately crude and is documented as such: we are not claiming the
    corpus has our labels, we are projecting ours onto real text so the classifier can be
    tested on wording it did not generate. The mapping table lives here so the report can
    print it and a reader can audit it.
    """
    from .risk.lexicon import RISK_LABELS, scan

    hits = scan(body)
    spans: list[dict[str, Any]] = []
    for label in RISK_LABELS:
        for h in hits.get(label, []):
            if h.polarity != "signal":
                continue
            spans.append(
                {
                    "text": h.text,
                    "label": label,
                    "start_char": h.start,
                    "end_char": h.end,
                    "extractor": "lexicon_projection",
                }
            )
    return spans


def sms_to_indicators(body: str) -> list[str]:
    return sorted({s["label"] for s in _spans_for_labels(body)})


def load_public_sms(
    path: Path | None = None, limit: int = 5000, seed: int = 20260310
) -> Split | None:
    """Load the cached SMS Spam Collection and project our indicators onto it.

    Returns ``None`` if the file has not been downloaded; callers must handle that and
    the report must say the public-data evaluation was skipped.
    """
    csv_path = path or (PUBLIC_DIR / SMS_SPAM["key"] / "smsspamcollection.csv")
    if not csv_path.exists():
        return None
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    # The canonical UCI file is comma-separated with a header; several community mirrors
    # redistribute it as tab-separated with no header. Accept both.
    sample = csv_path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        dialect = csv.excel_tab
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh, dialect=dialect)
        first = next(reader, None)
        if first:
            label0 = (first[0] or "").strip().lower()
            if label0 in {"ham", "spam"}:
                rows.append({"label": label0, "body": (first[1] if len(first) > 1 else "").strip()})
        for row in reader:
            if len(row) < 2:
                continue
            label, body = row[0].strip().lower(), row[1].strip()
            if not body or label not in {"ham", "spam"}:
                continue
            rows.append({"label": label, "body": body})
    if len(rows) > limit:
        idx = rng.choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(idx)]

    records: list[GoldRecord] = []
    for i, row in enumerate(rows):
        body = row["body"]
        is_spam = row["label"].lower() == "spam"
        spans = _spans_for_labels(body)
        indicators = sorted({s["label"] for s in spans})
        rec = GoldRecord(
            id=f"sms-{i:05d}",
            source_type="message",
            title=body[:60],
            body=body,
            occurred_at=None,
            participants=[],
            # The binary spam label is projected onto "this message shows at least one
            # pressure indicator". Spam in this corpus is overwhelmingly urgency+payment,
            # so this is a reasonable, and auditable, approximation - not ground truth.
            risk_labels=indicators if is_spam else [],
            entity_spans=[],
            scenario="public_sms_spam" if is_spam else "public_sms_ham",
        )
        records.append(rec)
    return _build_split("public_sms", records)


def public_dataset_path() -> Path:
    return PUBLIC_DIR / SMS_SPAM["key"] / "smsspamcollection.csv"


def public_dataset_meta() -> dict[str, Any]:
    return dict(SMS_SPAM)


def readme_for_public(dataset: dict[str, Any]) -> str:
    return f"""# {dataset['name']}

Source: {dataset['source_url']}
Licence: {dataset['licence']}

Citation
--------
{dataset['citation']}

How DigitalMind uses it
-----------------------
The multi-label risk indicators (urgency, threat language, emotional pressure, payment
request) are projected onto this real text with DigitalMind's own rule lexicon
(`app/datasets.py:_spans_for_labels`). The projection is auditable and printed in
`reports/evaluation_report.md`; it is NOT ground truth for those indicators. What the
dataset provides that our synthetic corpus cannot is *real human wording* for pressure
tactics, so it tests whether the classifier generalises beyond its own templates.

This directory is not committed. Run:

    python backend/scripts/fetch_public_data.py
"""

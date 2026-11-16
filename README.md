# DigitalMind

A private, explainable AI workspace that helps you understand connections across your own
digital records — emails, messages, documents, notes, calendar events and tasks.

It extracts entities and normalizes dates, resolves names into canonical entities, builds a
knowledge graph where **every edge is evidenced by a record and a character span**, places
records on a timeline that separates recorded facts from inferred ordering, reports risk as
**four independent indicators with cited evidence**, and answers questions with citations.

Everything runs on your machine. No record content leaves this machine. There is no account,
no telemetry, and no third-party integration.

---

## Run it

### Windows

```powershell
.\run.ps1
```

### macOS / Linux

```bash
./run.sh
```

That one command creates a virtualenv, installs dependencies, downloads the spaCy model if
needed, loads the demo corpus, runs the full analysis pipeline, and serves the app at
**http://127.0.0.1:8000**.

> **New here? Read [`docs/RUNNING.md`](docs/RUNNING.md)** — a step-by-step walkthrough with
> the exact output to expect at every stage, what to click in each of the ten views, and
> troubleshooting for the eight things that can go wrong.

Already have a Python 3.11 environment?

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m backend.run
```

Useful flags:

| Flag | Effect |
|---|---|
| `--reset` | Delete the local database and start fresh |
| `--no-demo` | Start with an empty workspace |
| `--no-reindex` | Skip the embedding index — much faster startup |
| `--skip-download` | Never touch the network (works offline) |

`run.ps1` wraps these plus three of its own, so you never need to touch `python`:

| Flag | Effect |
|---|---|
| `-Reset` | Same as `--reset` |
| `-NoDemo` / `-NoReindex` / `-SkipDownload` | Same as the long forms |
| `-Setup` | Force venv creation and dependency install |
| `-Test` | Run the test suite and exit |
| `-Evaluate` | Run the full evaluation suite and exit |
| `-Port 8080` / `-BindAddress 0.0.0.0` | Override the bind address |

> Use `-BindAddress`, **not** `-Host`. `$Host` is a read-only PowerShell automatic
> variable and declaring a parameter with that name makes the script fail on line 1.

To rebuild the frontend after changing it:

```bash
cd frontend && npm install && npm run build
```

The backend then serves `frontend/dist` at `/`. For frontend development use
`npm run dev` (port 5173) with the backend on port 8000.

---

## What you should see in the first two minutes

**Open Dashboard.** Counts by source type, four separate risk-indicator cards, and a
strip stating that analysis is local.

**Open Risk.** One record at the top: *"URGENT FINAL NOTICE - account closure in 10
minutes"*. It is flagged for all four indicators. Each one shows the exact words that
triggered it — `$4,250.00`, `within 10 minutes`, `will be closed permanently` — as clickable
citations into the record.

Below the indicators there is a single number. It is labelled **estimate**, and the card
shows the formula that produced it, the per-indicator contributions, and a line saying it is
not a probability of fraud. It exists for sorting, not for judging.

**Open the same record.** The body is rendered with every entity span and every piece of
risk evidence highlighted. Clicking an evidence quote from the Risk view lands here with
that span selected.

**Now open the benign look-alike** — *"Reminder: 10 days left to verify your account
details"*. It is a genuine bank reminder. It contains the word "closed", a deadline, and
account vocabulary. **It is not flagged.** That is the false-positive test, and it is
asserted in `backend/tests/test_api.py` and in
`backend/scripts/evaluate_demo_scenarios.py`.

**Open Timeline.** Three lanes. *Recorded* is what the source system asserted. *Explicit
dates* are what the text named. *Inferred* is relative wording like "last week", shown as a
**range** with the raw phrase beside it, because "last week" does not name a day.

**Open Insights and find Scenario B.** A message says *"As we discussed last week"*. The
insight reads:

> This record refers back to an earlier discussion ("As we discussed"), but **No matching
> discussion was found in the records analyzed.** I searched 4 earlier record(s) using
> keyword and semantic search. That is not evidence that no such discussion took place — the
> original channel may simply not be connected or imported yet.

The insight also states what was searched, and cites the exact phrase and character span.

**Open Ask** and try:

- `What records mention Northwind Bank?` — six cited records with verbatim quotes
- `What happened before this payment request?` — the timeline walked backwards, with each
  item labelled by how its time was established
- `What records mention the Zephyrine acquisition?` — the cautious empty state above
- `Which records are risky?` — indicator-level breakdown, never a verdict

**Open Settings → Privacy.** Counts, database size, the log policy, what deleting a record
actually deletes, and the fact that no integration is configured. **Delete all data** is
behind a typed confirmation.

---

## The rules this project holds itself to

These are enforced in code and asserted in tests, not just documented.

**Every fact links to its source and its exact words.** Extraction, risk evidence, QA
citations and graph edges all carry `(record_id, start_char, end_char)`. Record bodies are
stored byte-for-byte. `enforce_citation_integrity` re-checks every citation against the
source text before the API returns it.

**Extracted facts are separated from model interpretations.** Spans are labelled *fact*.
Insights are labelled *interpretation*, carry a confidence, and carry the basis for that
confidence in words.

**Cautious wording.** The headline is always "Potential manipulation risk detected". A
deny-list of claims we never make — `is a scammer`, `is fraudulent`, `guaranteed scam`, and
seven others — is checked against every generated string.

**No unexplained aggregate.** `overall_estimate()` returns `is_estimate: true`, the formula,
the four contributions with their weights, and a caveat. Tests assert all four are present.

**A missing record is never proof that something did not happen.** One required sentence,
used everywhere, with the coverage of the search attached so you can judge how partial your
data is.

**Users can inspect, dismiss or correct any insight.** Dismiss, restore, and a correction
note that marks the insight `corrected`. Dismissals are respected by the default views.

---

## Architecture

```
backend/app/
  config.py            settings; everything local-first, overridable by DIGITALMIND_* env vars
  db.py, models.py     SQLAlchemy 2.0 + SQLite. Every derived row cascades from its record.
  pipeline.py          ingest -> extract -> resolve -> graph -> risk -> index -> insights
  api.py               all routes, under /api
  logging_config.py    log redaction; record content is never logged

  nlp/
    rules.py           regex/lexicon baseline: money, orgs, people, events, emails
    ner.py             spaCy wrapper with noise filtering and graceful degradation
    dates.py           absolute vs relative expressions; relative ones resolve to ranges
    extract.py         merges the two extractors, keeping provenance per span

  entity/
    resolve.py         blocking + rapidfuzz + union-find; merges are auditable
    graph.py           NetworkX graph; mentioned_in edges derived, never persisted stale

  risk/
    lexicon.py         four indicators, signals AND mitigators, each with a human-readable note
    tfidf_lr.py        one-vs-rest logistic regression; thresholds tuned on dev
    transformer_clf.py fine-tuned DistilBERT, multi-label, calibrated
    calibration.py     isotonic + Platt, ECE/MCE/Brier, all in numpy
    assess.py          indicator combination, cautious wording, the labelled estimate

  retrieval/
    chunker.py         overlapping chunks with exact character offsets
    bm25_index.py      Okapi BM25 baseline
    embeddings.py      sentence-transformers -> transformers mean-pooling -> hashing fallback
    hybrid.py          RRF and weighted fusion, both evaluated
    qa.py              intents, citations, the no-match wording
    answering.py       assembles answers from retrieved spans

  timeline/builder.py  three lanes, records-first ordering
  privacy/__init__.py  cascade delete + verification, LLM policy gate
  ingest/              demo corpus, seeded generator, file parsers
```

```
backend/scripts/       generate_dataset, fetch_public_data, train_risk,
                       evaluate_{demo_scenarios,extraction,entity_resolution,risk,retrieval},
                       error_analysis, make_report
backend/tests/         108 tests
frontend/src/          React + Vite + TypeScript + Tailwind, ten views
reports/               evaluation_report.md and per-component JSON
docs/DECISIONS.md      16 decisions with rejected alternatives
```

---

## NLP and ML

**Baselines first.** Rules/regex extraction and risk detection, TF-IDF + logistic regression
for risk, BM25 for retrieval. Every baseline is evaluated against the neural components so
the report can say what each one actually contributes.

**Main models.** spaCy NER · a fine-tuned DistilBERT multi-label risk classifier with
per-label dev-tuned thresholds and probability calibration · sentence-transformer
embeddings with a FAISS index · hybrid retrieval fusion.

**Evaluation.** Strict and boundary-relaxed F1 for NER · per-indicator P/R/F1 plus macro and
micro for risk, with the false-positive rate on benign records reported next to every F1 ·
pairwise and B-cubed F1 plus a **distractor false-merge rate** for entity resolution ·
Recall@k / MRR / nDCG@k for retrieval · ECE / MCE / Brier for calibration · threshold sweeps
and ablations for each component · grouped error analysis.

**Data.** 320 seeded synthetic records with gold entity spans, gold multi-label risk, gold
back-references, and a 70/15/15 stratified split. Benign look-alikes carry the *same*
surface features as the attack records — deadlines, amounts, account vocabulary, urgency —
because precision is untestable without them. Plus one real public corpus (SMS Spam
Collection, UCI, **CC BY 4.0**) for wording we did not write.

Run the whole suite:

```powershell
.\run.ps1 -Evaluate
```

or step by step — see the reproduction section of `reports/evaluation_report.md`.

---

## Privacy

- **Local-first.** No account, no telemetry, no outbound request. Model downloads happen at
  install time, once, from public registries.
- **Logs contain no record content.** Only ids, counts, durations and model names. A
  redaction filter is installed as a backstop in case a dependency logs a payload.
- **Deleting a record deletes everything derived from it** — entities, aliases, graph edges,
  events, risk assessments, insights and both index entries. Enforced by schema cascade, an
  explicit child-first sweep, *and* a verification query that fails loudly. Tested in
  `backend/tests/test_privacy.py`.
- **The LLM layer is off by default.** Enabling it requires an explicit endpoint and is
  disclosed in the UI. When enabled it may only rephrase an already-cited answer, and
  citation integrity is re-verified afterwards. The app is fully functional without it.
- **No integrations are configured**, and the import screen says so for each of Gmail,
  Outlook, Slack, Notion and Calendar, with the export-and-upload alternative. It never
  implies a connection exists.

---

## Importing your own records

Drag and drop `.txt`, `.md`, `.eml`, `.csv`, `.json` or `.ics` files on the Import screen.
Everything is parsed locally with the standard library.

- **`.eml`** — headers and body; the `Date:` header becomes the source timestamp
- **`.ics`** — VEVENT blocks; `DTSTART` becomes the source timestamp; attendee display names
  come from the `CN` parameters
- **`.csv`** — needs a `body`/`text`/`content` column; `date` and `participants` are optional
- **`.json`** — a list of objects, or an object with a `records` list
- **`.txt` / `.md`** — one record per `---` section, or per markdown heading

A record with no source timestamp is marked as such and kept out of the timeline's
*recorded* lane. It is never given the import time and presented as a fact.

---

## Known limitations

Read `reports/evaluation_report.md` §8 for the full list. The three that matter most:

1. **Synthetic-data bias.** The classifier scores high partly because the benign look-alikes
   reuse the attack records' sentence frames. That gap measures template discrimination, not
   generalisation. The public-corpus numbers are the ones that speak to real wording.
2. **False positives on legitimate urgent messages.** Real institutions use urgent language.
   The false-positive rate on benign records is reported next to every F1 for that reason.
3. **English only.** spaCy is the English pipeline; the date patterns, the risk lexicon and
   the generator are all English. Nothing has been tested on another language.

Also: no coreference resolution, no base-rate adjustment, no causal inference from the
timeline, and no action is ever taken on a record.

---

## Testing

```bash
.\run.ps1 -Test          # or: python -m pytest -q
```

108 tests covering extraction, date normalization, entity resolution, risk scoring and
wording, the timeline lanes, the cascade delete, every parser, the full API surface, and the
product rules. Marked tests: `-m slow` for training, `-m ml` for the optional extras.

---

## Tech stack

Python 3.11 · FastAPI · SQLAlchemy 2.0 + SQLite · spaCy · scikit-learn · rank_bm25 ·
rapidfuzz · NetworkX · React 18 + Vite + TypeScript + Tailwind CSS · pytest

Optional: torch · transformers · sentence-transformers · FAISS

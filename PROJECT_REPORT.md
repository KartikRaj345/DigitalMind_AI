# DigitalMind - Project Report

**Date:** October 28, 2026  
**Version:** 0.1.0  
**Type:** Private, explainable AI workspace for analyzing personal digital records  
**Architecture:** Local-first, privacy-preserving

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Project Goals & Philosophy](#project-goals--philosophy)
- [Core Principles](#core-principles)
- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Data Pipeline](#data-pipeline)
- [NLP & Machine Learning](#nlp--machine-learning)
- [Entity Resolution & Knowledge Graph](#entity-resolution--knowledge-graph)
- [Risk Detection](#risk-detection)
- [Retrieval & Question Answering](#retrieval--question-answering)
- [Timeline & Insights](#timeline--insights)
- [Privacy & Security](#privacy--security)
- [Evaluation](#evaluation)
- [Technology Stack](#technology-stack)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Future Considerations](#future-considerations)

---

## Executive Summary

DigitalMind is a fully local, explainable AI workspace that helps users understand connections across their personal digital records (emails, messages, documents, notes, calendar events, and tasks). The system extracts entities, normalizes dates, builds auditable knowledge graphs, detects potential manipulation patterns with cited evidence, and answers questions with verifiable citations—all without sending data to external services.

The project emphasizes **transparency over automation**: every fact links to its source with exact character spans, interpretations are clearly labeled as such, missing information is reported honestly rather than inferred, and users maintain full control over their data.

---

## Project Goals & Philosophy

### Primary Goals
1. **Local-first analysis** - All processing happens on the user's machine with no telemetry or external data transmission
2. **Explainability** - Every conclusion must be traceable to specific text with character-level citations
3. **Honest uncertainty** - Distinguish between facts, explicit statements, and inferred interpretations
4. **Cautious risk assessment** - Flag patterns without making definitive accusations
5. **User agency** - Allow inspection, correction, and dismissal of all insights

### Design Philosophy
- **Facts vs interpretations**: Extracted entities are facts about text; risk assessments and insights are labeled interpretations with confidence
- **Absence is not proof**: Missing records are reported as "not found" with search coverage, never as "didn't happen"
- **Evidence-based**: Every edge in the knowledge graph requires a record and character span as evidence
- **Graceful degradation**: Core functionality works without heavy ML dependencies
- **Defensive engineering**: Multiple layers of verification for privacy, citations, and data integrity

---

## Core Principles

These principles are enforced in code and asserted in tests:

| Principle | Implementation |
|-----------|---------------|
| **Every fact links to source** | All spans, citations, and graph edges store `(record_id, start_char, end_char)`. Bodies preserved byte-for-byte. |
| **Separate facts from interpretations** | Spans labeled as facts; insights labeled as interpretations with confidence and basis. |
| **Cautious wording** | Deny-list prevents accusatory claims like "is a scammer", "is fraudulent". Headline is always "Potential manipulation risk detected". |
| **No unexplained aggregates** | Risk estimates labeled `is_estimate: true` with formula, per-indicator weights, and explicit caveat they're not fraud probabilities. |
| **Absence != non-existence** | Required wording for no matches, with coverage reporting and omission risk caveats. |
| **User control** | All insights can be inspected, dismissed, restored, or corrected with notes. |

---

## Architecture Overview

### System Structure

```
DigitalMind/
├── backend/           # FastAPI backend
│   ├── app/          # Core application logic
│   │   ├── api.py    # REST API endpoints
│   │   ├── pipeline.py # Ingest-extract-resolve-graph-risk-index-insights
│   │   ├── config.py # Settings (local-first, env-overridable)
│   │   ├── db.py     # SQLAlchemy 2.0 + SQLite
│   │   └── models.py # Database schema
│   ├── scripts/      # Utility scripts (eval, training, data generation)
│   └── tests/        # 108 tests
├── frontend/         # React + Vite + TypeScript + Tailwind
│   └── src/          # 10 views
├── data/             # Local data (gitignored)
├── artifacts/        # Trained models (gitignored)
├── reports/          # Evaluation reports (committed)
└── docs/             # Documentation
```

### Processing Pipeline

1. **Ingest** - Load demo corpus or user files (.txt, .md, .eml, .csv, .json, .ics)
2. **Extract** - NLP extraction of entities, dates, and back-references
3. **Resolve** - Entity resolution into canonical entities with aliases
4. **Graph** - Build knowledge graph with evidenced edges
5. **Risk** - Assess manipulation risk across 4 independent indicators
6. **Index** - Build BM25 and embedding indexes for retrieval
7. **Insights** - Generate observations, threads, and back-reference insights

All derived data cascades from records with verification on deletion.

---

## Key Features

### 10 Frontend Views
1. **Dashboard** - Overview with record counts, source types, and 4 risk indicator cards
2. **Records** - Browse all records with full details
3. **Record Detail** - View with highlighted entities and risk evidence
4. **Risk** - Per-indicator breakdown with cited evidence, no single verdict
5. **Timeline** - Three lanes (Recorded, Explicit dates, Inferred) with range-based display
6. **Graph** - Knowledge graph visualization with evidenced edges
7. **Entities** - Canonical entities with aliases and mentions
8. **Insights** - Observations with dismiss/correct/restore actions
9. **Ask** - Question answering with citations and coverage
10. **Settings/Privacy/Import** - Data management and import

### Supported File Formats
- `.txt`, `.md` - Plain text/markdown (split by `---` or headings)
- `.eml` - Email with headers, `Date:` becomes source timestamp
- `.ics` - Calendar events, `DTSTART` becomes timestamp, attendees from `CN`
- `.csv` - Requires `body`/`text`/`content`, optional `date`/`participants`
- `.json` - List or object with `records` array

---

## Data Pipeline

### Synthetic Dataset (Primary Evaluation Corpus)
- **Size**: 320 seeded records (deterministic, seed `20260310`)
- **Split**: 70% train (224) / 15% dev (48) / 15% test (48), stratified by scenario
- **Gold annotations**: Entity spans (812, 100% offset-valid), multi-label risk (4 indicators), back-references
- **Scenarios**: 4 types - plain, risk, back-reference, benign look-alike
- **Quality controls**: Zero duplicate bodies, zero train/test leakage (fixed via resampling), all spans valid
- **Benign look-alikes**: 96 records with same surface features as attacks (amounts, deadlines, account vocabulary) but unlabeled - critical for false positive testing

### Public Dataset
- **SMS Spam Collection** (UCI, CC BY 4.0) - 5,000 messages for real-world validation
- Tests generalization beyond synthetic templates

### Data Integrity
- Record bodies preserved byte-for-byte
- Character offsets validated: `body[start:end] == text` for all spans
- Citation integrity enforced before API responses
- No whitespace normalization in parsers

---

## NLP & Machine Learning

### Extraction Layer (`backend/app/nlp/`)

| Module | Purpose | Approach |
|--------|---------|----------|
| **rules.py** | Deterministic baseline (reference implementation) | Regex for money, emails, orgs, people, events. Always available. |
| **ner.py** | spaCy NER wrapper | `en_core_web_sm` with aggressive noise filtering, graceful degradation if unavailable |
| **dates.py** | Date/time normalization | Distinguishes absolute vs relative. Relative resolve to ranges with uncertainty. Handles back-references. |
| **extract.py** | Orchestrator | Merges rules + spaCy (agreement boosts confidence), separates spans/events/backrefs |

**Key design**: Rules are conservative and explainable. spaCy provides recall boost but is heavily filtered (rejects times, postcodes, spans crossing lines, etc.). Agreement between extractors increases confidence.

### Risk Classification (`backend/app/risk/`)

Four independent indicators, scored separately:

| Indicator | Purpose | Approach |
|-----------|---------|----------|
| **urgency** | Time pressure | Lexicon + patterns |
| **threat_language** | Coercive threats | Lexicon with context |
| **emotional_pressure** | Manipulation tactics | Lexicon patterns |
| **payment_request** | Financial demands | Lexicon with demand framing required (bare account numbers insufficient) |

**Scoring mechanism:**
- Lexicon has positive signals and **mitigators** (negative weights)
- Formula: `1 - Π(1 - w_signal)` reduced by `Π(1 - |w_mitigator|)`
- Mitigators catch benign phrases like "we will never ask you to move money", "visit a branch"
- Thresholds tuned on dev set
- Calibrated with isotonic regression or Platt scaling (in-tree implementation)

**Models:**
- **Rules baseline** - Fast, explainable
- **TF-IDF + Logistic Regression** - One-vs-rest, per-label thresholds
- **DistilBERT** (optional) - Fine-tuned multi-label, calibrated, requires `requirements-ml.txt`

### Retrieval (`backend/app/retrieval/`)

| Component | Purpose |
|-----------|---------|
| **chunker.py** | Overlapping chunks with exact character offsets |
| **bm25_index.py** | Okapi BM25 baseline (lexical) |
| **embeddings.py** | sentence-transformers or fallback: in-tree mean-pooling transformer embedder or deterministic hashing |
| **hybrid.py** | RRF (Reciprocal Rank Fusion, k=60) and weighted fusion, both evaluated |
| **qa.py** | Intent classification, citation assembly, honest no-match handling |
| **answering.py** | Builds answers from retrieved spans only |

**Embedding strategy**: Tries sentence-transformers first, falls back to in-tree transformer mean-pooling if blocked (handles Windows DLL issues), finally falls back to hashing embedder. FAISS for speed (optional), exact numpy for fallback.

---

## Entity Resolution & Knowledge Graph

### Entity Resolution (`backend/app/entity/resolve.py`)

**Algorithm:** Blocking + RapidFuzz + Union-Find
- Normalization and blocking to reduce comparisons
- String similarity with careful safeguards
- Union-find for transitive merging
- Full audit trail of merges

**Critical safeguards (added after bug discovery):**
1. **Leading-token veto** - Hard veto if normalized forms have different leading tokens (prevents `L. Ashworth` merging with `N. Ashworth`)
2. **No partial matching for people** - Uses whole-string ratio only, never `token_set_ratio` or `partial_ratio`
3. **Token containment restrictions** - Only when extra tokens carry no identity (role words, single initials)
4. **Email handle rule** - `priya.raman@studio.example` → `priya raman` (high confidence)

**Benchmark:** Synthetic surface-form benchmark with distractor pairs sharing surnames. Found and fixed false-merge bug where rate was 1.00.

### Knowledge Graph (`backend/app/entity/graph.py`)

- Built with NetworkX
- **Every edge requires evidence**: `(record_id, start_char, end_char)`
- `mentioned_in` edges derived, never persisted stale
- Orphaned entities (no mentions) removed on record deletion
- Edges cascade with records

---

## Timeline & Insights

### Timeline Builder (`backend/app/timeline/builder.py`)

Three distinct lanes:

| Lane | Source | Display | Notes |
|------|--------|---------|-------|
| **Recorded** | Source system timestamp (email Date:, ICS DTSTART, task due) | Points/timeline | Facts from metadata |
| **Explicit** | Dates written in text | Points/timeline | Extracted absolute dates |
| **Inferred** | Relative expressions ("last week", "in 10 minutes") | **Ranges (bars)** | Resolved against record timestamp, marked inferred with explanation |

**Rules:**
- Inferred items sorted by range end (never jump ahead)
- Records with no source timestamp don't use import time as fact
- Each inferred item shows raw phrase and resolution explanation
- Distinction enforced in UI and tests

### Insights

Generated types:
- `risk_observation` - Risk pattern observations
- `record_thread` - Connected records
- `discussion_reference` - Back-reference found with matches
- `unresolved_discussion_reference` - Back-reference with no matches found

**For unresolved back-references (Scenario B):**
- States "No matching discussion was found in the records analyzed" (never claims it didn't happen)
- Reports search coverage (how many records searched, date range, sources)
- Cites exact phrase and character span
- Includes omission risk caveat
- Can be dismissed/restored/corrected

**Safety:** Back-reference from earliest record → `unverifiable_discussion_reference` (nothing to search). All insights pass through deny-list check.

---

## Privacy & Security

### Local-First Guarantees
- **No network transmission** of record content
- **No telemetry, no accounts**
- **Model downloads only at install time** from public registries
- **Optional LLM off by default**, requires explicit endpoint + UI disclosure
- **Integrations not configured** by default, shown with export-upload alternative

### Data Protection
- **No content in logs** - Redacting filter drops content-shaped keys, logs only counts/IDs
- **Cascade deletion** - Three layers: `ON DELETE CASCADE`, child-first sweep, verification
- **Orphan cleanup** - Entities with no mentions removed
- **Index invalidation** - Both BM25 and embedding indexes cleared
- **Verification** - `verify_record_purged()` checks all derived tables, fails loudly on residue

### LLM Safety (if enabled)
- Only allowed to **rephrase** already-cited answers
- Citation integrity re-verified after rephrasing
- Must have citations to use LLM
- Fully functional without LLM
- Explicit disclosure in UI

### Deletion
- User can delete all data via Settings → Privacy (typed confirmation)
- Or manually delete `data/digitalmind.db*`
- Verified to remove all derived tables

---

## Evaluation

### Results Summary (from `reports/evaluation_report.md`)

**Dataset:** 320 synthetic records, 70/15/15 stratified split

#### Extraction (NER)
| System | Strict F1 | Precision | Recall | Boundary F1 |
|--------|-----------|-----------|--------|-------------|
| Rules | 0.8780 | 0.8372 | 0.9231 | 0.9512 |
| spaCy | 0.5919 | 0.6226 | 0.5641 | 0.7713 |
| Hybrid (rules+spaCy) | 0.8638 | 0.7929 | 0.9487 | 0.9105 |

Dates: absolute precision 0.875, relative in-range 1.0

#### Entity Resolution
- Pairwise F1: **0.8539** (P=1.0000, R=0.7450)
- Distractor false-merge rate: **0.0000** (0/4)
- Bug fixed: initial run had false-merge rate 1.00 (L. Ashworth merged with N. Ashworth)

#### Risk Classification
| Model | Macro F1 | Benign FPR | Lookalike FPR |
|-------|----------|------------|---------------|
| Rules | 0.7568 | 0.2051 | 0.5333 |
| TF-IDF + LR | 0.7950 | 0.0000 | 0.0000 |
| DistilBERT | *trained separately* | — | — |

*Note:* TF-IDF performs well on synthetic data partly due to template learning; public corpus provides real-world validation.

#### Retrieval
| Method | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| BM25 | 0.5069 | 0.7917 | 0.8750 | 0.8581 |
| Embeddings | 0.3611 | 0.5069 | 0.6528 | 0.5076 |
| Hybrid RRF | 0.4375 | 0.7222 | 0.8333 | 0.7998 |

*Note:* BM25 beats hybrid on entity-name queries (lexical match), fusion helps on paraphrase queries.

#### Demo Scenarios
- **19/19 passed, 0 failed**
- Scenario A: Payment demand flagged with all 4 indicators, benign look-alike NOT flagged
- Scenario B: Unresolved back-reference with correct cautious wording, no false claims

#### Public Corpus (SMS Spam Collection)
- Evaluated on 5000 messages (13.2% spam)
- Provides validation on real-world wording

### Evaluation Metrics Reported
- **Extraction**: Strict F1, boundary-relaxed F1, per-label precision/recall
- **Risk**: Per-indicator P/R/F1, macro/micro, FPR on benign and lookalike, calibration (ECE/MCE/Brier)
- **Entity Resolution**: Pairwise F1, B-cubed F1, distractor false-merge rate, threshold sweeps, ablations
- **Retrieval**: Recall@k, MRR, nDCG@k across query types
- **Calibration**: Reliability diagrams, isotonic/Platt comparison

---

## Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Core language (pinned) |
| **FastAPI** | REST API |
| **SQLAlchemy 2.0** | ORM with SQLite |
| **spaCy** | NER (optional) |
| **scikit-learn** | TF-IDF, logistic regression |
| **rank-bm25** | BM25 indexing |
| **rapidfuzz** | Fuzzy string matching |
| **NetworkX** | Knowledge graph |
| **python-dateutil** | Date parsing |

### ML/Optional
| Technology | Purpose |
|------------|---------|
| **torch** | Deep learning backend |
| **transformers** | DistilBERT, in-tree embeddings fallback |
| **sentence-transformers** | Embeddings (optional) |
| **FAISS** | Vector search (optional) |
| **datasets** | Training data loading |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **TypeScript** | Type safety |
| **Vite** | Build tool/dev server |
| **Tailwind CSS** | Styling |

### Testing & Dev
| Technology | Purpose |
|------------|---------|
| **pytest** | Testing (108 tests) |
| **Uvicorn** | ASGI server |

---

## Testing

### Test Suite: 108 tests
Run with `.\run.ps1 -Test` or `python -m pytest -q`

**Coverage:**
| Test File | Focus |
|-----------|-------|
| `test_nlp_rules.py` | Extraction spans, overlaps, money/org/person rules |
| `test_dates.py` | Date normalization, relative resolution, ranges |
| `test_risk_calibration.py` | Isotonic monotonicity, Platt scaling, ECE |
| `test_privacy.py` | Cascade delete verification across all tables |
| `test_product_rules.py` | Wording rules, no-match wording, timeline lanes, deny-list |
| `test_parsers.py` | All file formats, upload-to-pipeline |
| `test_api.py` | Full HTTP surface, both demo scenarios |
| `test_pipeline.py` | Idempotency, span offsets, graph evidence |
| `test_dataset.py` | Synthetic dataset properties (added recently) |

**Markers:** `-m slow` for training, `-m ml` for optional ML extras

**Runtime:** ~4 minutes (mostly model loading)

---

## Known Limitations

### Data & Generalization
1. **Synthetic bias** - Classifier scores high partly due to template learning. Benign look-alikes reuse attack sentence frames. Public corpus numbers matter more.
2. **English only** - spaCy English pipeline, English patterns/lexicon/generator. Untested in other languages.
3. **Template-based data** - 320 synthetic records generated from templates; may not capture full real-world diversity.

### Model Limitations
4. **No coreference resolution** - Doesn't track "he/she/it" back to entities
5. **No base-rate adjustment** - Risk scores not calibrated to population base rates
6. **No causal inference** - Timeline shows ordering, not causation
7. **spaCy confidence** - Fixed proxy values (0.8-0.85), not true probabilities
8. **DistilBERT optional** - Not trained by default

### Design Constraints
9. **False positives on legitimate urgent messages** - Real institutions use urgency. FPR reported explicitly.
10. **Small corpus** - Designed for personal archives, not enterprise scale
11. **No automated actions** - System never acts on records, only analyzes
12. **Limited file formats** - Specific formats supported, others require conversion

### Technical
13. **Windows DLL issues** - Some environments need in-tree fallbacks (handled)
14. **Python 3.11 only** - Pinned due to dependency wheel availability
15. **SQLite only** - Single-user, local database

Full details in `reports/evaluation_report.md` §8.

---

## Future Considerations

### Potential Enhancements
- **Coreference resolution** - Link pronouns to entities
- **Multi-language support** - Extend beyond English
- **More file formats** - Support additional email/archive formats
- **Improved entity resolution** - Handle more complex name variations
- **Configurable risk thresholds** - User-adjustable sensitivity
- **Export functionality** - Export analysis results
- **Enhanced timeline** - Better handling of complex temporal expressions

### Architectural Notes
- LLM layer designed but off by default - can be enabled with careful gating
- Plugin architecture not implemented but design allows extension
- Could support other vector stores beyond FAISS
- Database could be swapped but SQLite fits local-first use case

---

## Quick Start

```powershell
# Windows
.\run.ps1

# macOS/Linux
./run.sh

# Run tests
.\run.ps1 -Test

# Run evaluation
.\run.ps1 -Evaluate
```

**Access:** http://127.0.0.1:8000  
**Default:** Loads 11 demo records with full analysis  
**Privacy:** All local, no external transmission

---

## Project Statistics

| Metric | Value |
|--------|-------|
| **Backend Python files** | ~60 modules |
| **Frontend files** | 18 files (10 views) |
| **Total tests** | 108 |
| **Demo records** | 11 |
| **Synthetic dataset** | 320 records with gold labels |
| **Architecture decisions documented** | 16 |
| **Evaluation reports** | 7 JSON files + 1 markdown report |

---

*This project demonstrates a privacy-first approach to personal knowledge management with strong emphasis on explainability, evidence, and honest uncertainty. Every design decision prioritizes user agency and verifiability over automation.*
# RUNNING.md — step-by-step

A walkthrough with the exact output you should see at each step.

---

## Contents

- [Step 0 — Prerequisites](#step-0--prerequisites)
- [Step 1 — Start the app](#step-1--start-the-app)
- [Step 2 — What to click, and what you should see](#step-2--what-to-click-and-what-you-should-see)
- [Step 3 — Run the tests](#step-3--run-the-tests)
- [Step 4 — Run the full evaluation](#step-4--run-the-full-evaluation)
- [Step 5 — Optional: train DistilBERT](#step-5--optional-train-distilbert)
- [Flags](#flags)
- [Starting without the wrapper](#starting-without-the-wrapper)
- [Troubleshooting](#troubleshooting)

---

## Step 0 — Prerequisites

**Python 3.11 is the only hard requirement.**

DigitalMind is pinned to 3.11 because spaCy, torch and FAISS only publish wheels for
specific minor versions. If your default `python` is newer, `run.ps1` finds `py -3.11` and
uses that to create `.venv` — you do not need to change your system interpreter.

Check what you have:

```powershell
py -0p          # lists installed Python versions
python --version # your default
```

**Nothing else is required.** These are optional and each has a defined fallback:

| Optional | Without it |
|---|---|
| `en_core_web_sm` | Extraction uses the regex baseline only; `/api/health` reports `spacy_available: false` |
| `requirements-ml.txt` | Risk uses rules + TF-IDF; no DistilBERT, no neural embeddings |
| `frontend/dist` | API still runs; use the Vite dev server for the UI |

`run.ps1` handles all of it. It will:

1. create `.venv` with Python 3.11 if missing
2. `pip install -r requirements.txt`
3. `npm install` in `frontend/` if `node_modules` is missing
4. download `en_core_web_sm` if missing (skip with `-SkipDownload`)
5. load the demo corpus and run the full pipeline if the database is empty
6. serve

**Already done on this machine:** venv, dependencies, spaCy model, frontend build,
TF-IDF model, synthetic dataset, public dataset. Go straight to Step 1.

---

## Step 1 — Start the app

```powershell
.\run.ps1
```

macOS / Linux:

```bash
./run.sh
```

### Expected output

Uvicorn's stderr usually interleaves first; the banner then flushes. Both appear.

```
Installing Python dependencies ...

DigitalMind
============================================================
  python        3.11.0
  dependencies  all present
  database      C:\Users\karti\OneDrive\Documents\DigitalMind\data\digitalmind.db
  spaCy model en_core_web_sm is installed
  demo data     loading and analysing (this takes ~10-30 seconds) ...
  demo data     11 records, 43 entities, 5 insights
  ml extras     available (torch + transformers)
  frontend      serving frontend\dist at /

  privacy       all analysis runs locally; no record content leaves this machine.
                the optional external LLM layer is off (the default).

  open          http://127.0.0.1:8000

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Reading the banner

| Line | Healthy | If not |
|---|---|---|
| `python 3.11.0` | 3.11.x | The script refuses to start on other versions |
| `dependencies all present` | yes | Stops with the exact install command |
| `database …/data/digitalmind.db` | created on first run | Check folder permissions |
| `spaCy model … is installed` | installed | Says "missing" → rules baseline only, app still works |
| `demo data 11 records, 43 entities, 5 insights` | first run | Second run shows `records 11 already imported` — expected |
| `ml extras available` | torch + transformers present | "not installed" → rules + TF-IDF only |
| `frontend serving frontend/dist at /` | built | "not built" → run `cd frontend; npm install; npm run build` |

### Confirm it responds

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | ConvertTo-Json
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "records": 11,
  "spacy_available": true,
  "transformers_available": true,
  "sentence_transformers_available": true,
  "faiss_available": true,
  "llm_enabled": false,
  "external_transmission": false
}
```

`llm_enabled` and `external_transmission` must both be `false`. That is the privacy
posture, and no configuration changes them.

Stop with **Ctrl+C**.

### Faster startup

```powershell
.\run.ps1 -NoReindex     # ~10s instead of ~30s
.\run.ps1 -Reset         # wipe the database and reload the demo corpus
.\run.ps1 -NoDemo        # empty workspace, import your own files
```

---

## Step 2 — What to click, and what you should see

Open **http://127.0.0.1:8000**. Six checks, in this order. Each one is asserted in the test
suite, so if reality differs from this, something is wrong.

### 2a. Dashboard

- 11 records, split across 6 source types (email, message, document, note, calendar, task)
- **Four separate** risk cards: Urgency, Threat language, Emotional pressure, Payment
  request — each showing `1`
- A disclaimer saying the indicators are not verdicts
- **No single combined risk number.** That is deliberate.

### 2b. Risk

One record at the top, severity **warning**:

> **Potential manipulation risk detected:** emotional pressure, payment request, threat
> language and urgency are present in this record. This is a pattern match, not a
> judgement about the sender.

Each indicator expands to the exact words that triggered it:

| Indicator | Evidence shown |
|---|---|
| Urgency | `within 10 minutes`, `must be resolved within 10 minutes` |
| Threat language | `will be closed`, `we will have no choice but` |
| Emotional pressure | `extremely worried`, `terrified`, `you will lose everything` |
| Payment request | `$4,250.00`, `transfer $4,250.00 to the verification account` |

Every evidence quote is clickable and jumps to the record with that character span
highlighted.

Below the indicators is a single number, labelled **estimate**, with its formula
(`payment_request 0.35, threat_language 0.30, urgency 0.20, emotional_pressure 0.15`),
the per-indicator contributions, and *"it is not a probability of fraud"* — all in the same
card. It exists for sorting, not for judging.

### 2c. The benign look-alike — the one that matters most

Search for **"Reminder: 10 days left"** and open it.

This is a *genuine* bank reminder. It contains the word "closed", a deadline, and account
vocabulary. The attack message contains the same surface features.

**Expect: zero risk chips, severity `none`.**

The rule baseline flagged this message in an earlier build. The mitigator weights in
`app/risk/lexicon.py` ("we will never ask you to move money", "no charge for verification",
"visit a branch") were added specifically to stop it. If it is flagged now, something
regressed — `test_scenario_a_leaves_the_benign_reminder_alone` will have failed too.

### 2d. Timeline

Three lanes, visually distinct:

| Lane | Colour | What it holds | Demo count |
|---|---|---|---|
| **Recorded** | teal | Timestamps the source system asserted (email `Date:`, ICS `DTSTART`, task due date) | 11 |
| **Explicit dates** | blue | Dates the text named, e.g. "4 March 2026" | 13 |
| **Inferred** | slate, soft-edged bars | Relative wording only — "last week" — shown as a **range** | 3 |

Inferred items display the raw phrase beside them (`last week`) plus an explanation of how
they were resolved. They never appear as a point in time, because "last week" does not name
a day.

A record with no source timestamp does **not** get its import time used as a substitute —
it goes to the explicit lane with a caveat.

### 2e. Insights — Scenario B

Four kinds are present: `risk_observation`, `record_thread`, `discussion_reference`,
`unresolved_discussion_reference`.

Open the `unresolved_discussion_reference`. This is Scenario B: a message saying
*"As we discussed last week"* where no matching record exists.

> This record refers back to an earlier discussion ("As we discussed"), but **No matching
> discussion was found in the records analyzed.** I searched 4 earlier record(s) using
> keyword and semantic search. That is not evidence that no such discussion took place —
> the original channel may simply not be connected or imported yet.

Also on that card:

- the coverage panel — how many records were searched, over what date range
- a caveat naming the omission risk
- the cited phrase, clickable, pointing at the exact characters
- **Dismiss** / **Restore** / **Correct** buttons

The `discussion_reference` insight is the control case: the *other* back-reference in the
corpus **does** have a match, proving the search works in both directions.

### 2f. Ask

| Question | Expected result |
|---|---|
| `What records mention Northwind Bank?` | ~6 records, each with a verbatim quote, timestamp and clickable citation |
| `What happened before this payment request?` | ~11 citations; the timeline walked backwards, each item labelled *recorded timestamp* or *date written in the text* |
| `What records mention the Zephyrine acquisition?` | The cautious empty state above, with a "what was searched" panel. Never any claim that nothing happened. |
| `Which records are risky?` | Indicator-level breakdown. Never a verdict about a sender. |
| `Who was involved in the account review?` | People and organizations only — no dates or amounts |

Every answer carries a caveat and a confidence with its basis stated in words.

### 2g. Settings → Privacy

- Row counts per table, database size on disk
- The log policy: *"Record content is never written to the application log."*
- What deleting a record actually deletes
- Per-component model availability, honestly reported
- **Delete all data**, behind a typed confirmation

And Settings → Import shows all five integrations (Gmail, Outlook, Slack, Notion, Calendar)
as **not configured**, each with the export-and-upload alternative. Nothing implies a
connection exists.

---

## Step 3 — Run the tests

```powershell
.\run.ps1 -Test
```

**Expect:**

```
........................................................................ [ 66%]
....................................                                     [100%]
108 passed in 245.31s
```

~4 minutes — most of it is spaCy and the transformer loading.

By file:

| File | Covers |
|---|---|
| `test_nlp_rules.py` | Extraction spans are exact substrings, no overlaps, money/org/person rules |
| `test_risk_calibration.py` | Isotonic monotonicity, Platt recovery, ECE on a known case |
| `test_privacy.py` | **Cascade delete** — every derived table empty after deleting a record |
| `test_product_rules.py` | Cautious wording, no unexplained aggregate, the no-match wording, timeline lanes |
| `test_parsers.py` | `.txt` `.md` `.eml` `.csv` `.json` `.ics`, plus upload → pipeline |
| `test_api.py` | The full HTTP surface including both scenarios |
| `test_pipeline.py` | Idempotency, span offsets, graph evidence |

---

## Step 4 — Run the full evaluation

```powershell
.\run.ps1 -Evaluate
```

Nine scripts, seven files written to `reports/`. **~10 minutes.**

### Expected output (abridged)

```
train  n= 224 positives={'urgency': 36, 'threat_language': 38, 'emotional_pressure': 40, 'payment_request': 40}
dev    n=  48 ...
test   n=  48 ...
tfidf-lr test macro-F1: 0.7437

Scenario A - payment demand in context of a real account issue
  [PASS] A. payment message is flagged for all four indicators
  [PASS] A. every cited span points at the words it quotes - 19 spans cited, 0 mismatched
  [PASS] A. benign look-alike reminder is NOT flagged - flagged=[]
  [PASS] A. the earlier account records are connected to the payment message
  [PASS] A. timeline places the earlier records before the payment request
  [PASS] A. recorded and inferred items are in separate lanes

Scenario B - reference to a discussion that may not be in the data
  [PASS] B. an unresolved-discussion insight is produced
  [PASS] B. the required cautious wording appears verbatim
  [PASS] B. the wording does not claim the discussion did not happen
  [PASS] B. the insight cites the exact phrase in the source record
  [PASS] B. a back-reference WITH a matching record is not reported as unmatched

19 passed, 0 failed

  rules   strict-F1=0.8780 P=0.8372 R=0.9231 boundary-F1=0.9512
  spacy   strict-F1=0.5919 P=0.6226 R=0.5641 boundary-F1=0.7713
  hybrid  strict-F1=0.8638 P=0.7929 R=0.9487 boundary-F1=0.9105
  dates:  absolute-precision=0.875 relative-in-range=1.0

pairwise F1 = 0.8539 (P=1.0000, R=0.7450)
distractor false-merge rate = 0.0000 (0/4)

  bm25             R@1=0.5069 R@5=0.7917 R@10=0.8750 MRR=0.8581
  embeddings       R@1=0.3611 R@5=0.5069 R@10=0.6528 MRR=0.5076
  hybrid_rrf       R@1=0.4375 R@5=0.7222 R@10=0.8333 MRR=0.7998

  rules       macro-F1=0.7568 benign-FPR=0.2051 lookalike-FPR=0.5333
  tfidf-lr    macro-F1=0.7950 benign-FPR=0.0000 lookalike-FPR=0.0000

evaluating on the real public corpus (5000 messages)
  spam fraction: 0.132
```

### Read §8 before quoting any number

`reports/evaluation_report.md` §8 (Limitations) is the section that matters. Two results
look wrong until you read it:

**"TF-IDF beats the rules baseline"** — partly because it learned our own templates. The
benign look-alikes reuse the attack records' sentence frames, which is the one distinction a
bag-of-ngrams model is good at and a human is not fooled by. The public-corpus figures are
the ones that speak to wording we did not write.

**"BM25 beats hybrid fusion"** — because most benchmark queries are entity names lifted from
the record: the best case for exact lexical matching, the worst case for embeddings. Fusion
is kept for the paraphrase queries, where BM25 has no lexical overlap at least. The per-kind
breakdown in §6 shows both.

### Output files

| File | Contents |
|---|---|
| `reports/evaluation_report.md` | **The written report.** 9 sections, ~400 lines |
| `reports/demo_scenarios.json` | 19 assertions with pass/fail and evidence |
| `reports/extraction_eval.json` | Per-label NER metrics + grouped errors |
| `reports/entity_resolution_eval.json` | Threshold sweep, ablations, 7 qualitative cases |
| `reports/risk_eval.json` | Per-indicator metrics, calibration, reliability bins |
| `reports/retrieval_eval.json` | 4 configurations × 3 query kinds |
| `reports/error_analysis.json` | Failures grouped by cause, with mitigations |

---

## Step 5 — Optional: train DistilBERT

The app is complete without it. `/api/health` and the dashboard report it as
`"status": "not available"` rather than hiding it.

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.train_risk --model distilbert
```

**Expect:**

```
INFO backend.app.risk.transformer_clf: epoch=1 steps=42 mean_loss=0.3412
INFO backend.app.risk.transformer_clf: epoch=2 steps=84 mean_loss=0.1874
INFO backend.app.risk.transformer_clf: epoch=3 steps=126 mean_loss=0.1129
INFO backend.app.risk.transformer_clf: saved classifier to artifacts\risk\distilbert-risk
distilbert test macro-F1: 0.xxxx
```

CPU-only. A few minutes. Then re-run `.\run.ps1 -Reset` to pick it up, and re-run
`.\run.ps1 -Evaluate` to include it in the report.

Artifacts land in `artifacts/risk/distilbert-risk/`. Calibration params and per-label
thresholds go to `risk_calibration.json` alongside the weights.

---

## Flags

### `run.ps1`

| Flag | Effect |
|---|---|
| `-Reset` | Delete the database and reload the demo corpus |
| `-NoDemo` | Start with an empty workspace |
| `-NoReindex` | Skip the embedding index — much faster startup |
| `-SkipDownload` | Never touch the network (works offline) |
| `-Setup` | Force venv creation and dependency install |
| `-Test` | Run the test suite and exit |
| `-Evaluate` | Run the full evaluation suite and exit |
| `-Port 8080` | Override the port |
| `-BindAddress 0.0.0.0` | Override the bind address |

> Use `-BindAddress`, **not** `-Host`. `$Host` is a read-only PowerShell automatic variable,
> so a parameter with that name makes the script fail on line 1 with
> `Cannot overwrite variable Host`. This was a real bug, fixed in `9f93429`.

### `run.sh`

Same flags with `--` prefixes: `--setup --reset --no-demo --no-reindex --skip-download
--test --evaluate --host --port`.

### `python -m backend.run`

The script `run.ps1` wraps. Takes the same flags. Useful for skipping the npm check.

---

## Starting without the wrapper

Two terminals, for frontend development:

```powershell
# terminal 1
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000 --reload
```

```bash
# terminal 2
cd frontend && npm run dev
```

The Vite dev server runs on **http://localhost:5173** and proxies `/api` to port 8000.

Once `frontend/dist` exists (after any `npm run build`), the backend serves the whole app
at `/` and you only need one terminal.

---

## Troubleshooting

### `Cannot overwrite variable Host`

You used `-Host`. Use `-BindAddress`.

### `run.ps1 cannot be loaded because running scripts is disabled`

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Or, once per machine: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### `Python 3.11 needed for DigitalMind`

Install from python.org, then re-run. The script finds it via `py -3.11`.

### `frontend not built`

```bash
cd frontend
npm install
npm run build
```

### The app starts but the semantic channel is missing

Check `sentence_transformers_available` in `/api/health`. If `false`, the app has silently
fallen back to a hashing embedder, which is **not** a semantic model. The Settings panel
shows this. It is also possible on locked-down Windows machines where an application-control
policy blocks the `pyarrow` DLL that `sentence-transformers` imports — in that case the
in-tree mean-pooling embedder takes over instead, and FAISS is used with dim 384.

### `distilbert: not available`

Expected until Step 5. Nothing else breaks.

### Port 8000 already in use

```powershell
.\run.ps1 -Port 8080
```

### Stale analysis after editing code

```powershell
.\run.ps1 -Reset
```

or, without restarting: `POST /api/analysis/rerun`.

### Start completely clean

```powershell
.\run.ps1 -Setup -Reset -NoReindex
```

### Delete everything

Settings → Privacy → **Delete all data**, or:

```powershell
Remove-Item data\digitalmind.db* -Force
```

---

## Where things are

```
data/digitalmind.db          your workspace (gitignored — never contains committed records)
data/datasets/synthetic/     seeded 320-record corpus with gold labels
data/datasets/public/        SMS Spam Collection (CC BY 4.0, gitignored)
artifacts/risk/              trained models (gitignored)
reports/                     evaluation output (committed)
docs/DECISIONS.md            16 decisions with rejected alternatives
```

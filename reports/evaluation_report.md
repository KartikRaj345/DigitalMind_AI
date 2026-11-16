# DigitalMind — evaluation report

Generated 2026-10-16 06:59 UTC. Every number in this document comes from a script in `backend/scripts/` writing a JSON file in `reports/`. Sections that were not evaluated say so explicitly and name the command that would evaluate them.

---

## 1. What was built and what it does

DigitalMind is a local-first workspace that reads your own records and connects them. It extracts entities and normalized dates, resolves names into canonical entities, builds a knowledge graph where every edge is evidenced, places records on a timeline that separates recorded facts from inferred ordering, reports risk indicators separately with cited spans, and answers questions with citations.

### Scenario verification

`19 passed, 0 failed` — every check is an assertion about the running application, not about a stored number.

| Check | Result |
|---|---|
| A. payment message is flagged for all four indicators | pass |
| A. every cited span points at the words it quotes | pass |
| A. benign look-alike reminder is NOT flagged | pass |
| A. the earlier account records are connected to the payment message | pass |
| A. timeline places the earlier records before the payment request | pass |
| A. recorded and inferred items are in separate lanes | pass |
| B. an unresolved-discussion insight is produced | pass |
| B. the required cautious wording appears verbatim | pass |
| B. the wording does not claim the discussion did not happen | pass |
| B. the insight cites the exact phrase in the source record | pass |
| B. a back-reference WITH a matching record is not reported as unmatched | pass |
| Ask: 'what happened before' cites the earlier records | pass |
| Ask: every citation matches the source span | pass |
| Ask: an unmatched query uses the required wording | pass |
| Ask: 'what records mention' returns cited records with spans | pass |
| Ask: a topic absent from the data uses the cautious wording | pass |
| No forbidden claim wording anywhere in the output | pass |
| Wording guard passes | pass |
| Any overall figure is labelled as an estimate and shows its formula | pass |

---

## 2. Data

**Synthetic corpus** — 320 records generated from seed `20260310`, split {'train': 224, 'dev': 48, 'test': 48} into train/dev/test. Composition is stratified by scenario so each split keeps the same risky/benign mix:

| Split | Records | Positive rate |
|---|---|---|
| train | 224 | 0.201 |
| dev | 48 | 0.208 |
| test | 48 | 0.188 |

Scenarios present in every split: `generated_backref`, `generated_benign_lookalike`, `generated_plain`, `generated_risk`

**Real public corpus**

SMS Spam Collection v1 (UCI), **5000 messages**, spam fraction 0.132. Licence: **CC BY 4.0**. It is downloaded once by `python -m backend.scripts.fetch_public_data` and is not committed to the repository.

Our four indicators are *projected* onto this text with DigitalMind's own rule lexicon (`app/datasets.py:_spans_for_labels`). The projection is auditable, but it is **not** ground truth for those indicators - see the error analysis for what it revealed.

---

## 3. Entity extraction and date normalization

Test split, 48 records. Strict scoring requires exact character offsets *and* the correct label; boundary-relaxed scoring counts any overlap, which separates 'found the name' from 'found it with the right edges'.

| Configuration | Precision | Recall | Strict F1 | Boundary F1 |
|---|---|---|---|---|
| `rules` | 0.837 | 0.923 | 0.878 | 0.951 |
| `spacy` | 0.623 | 0.564 | 0.592 | 0.771 |
| `hybrid` | 0.793 | 0.949 | 0.864 | 0.910 |

Per-label (hybrid configuration):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| MONEY | 1.000 | 1.000 | 1.000 | 24 |
| ORG | 0.673 | 0.917 | 0.776 | 36 |
| PERSON | 0.806 | 0.947 | 0.871 | 57 |

**Date normalization**

| Check | Value | n |
|---|---|---|
| Absolute dates resolving to the stated day | 0.875 | 16 |
| Relative expressions given a range, not a point | 1.000 | 18 |
| Relative expressions resolving before the record | 0.278 | 18 |

Mismatches against an independent `dateutil` parse:

- `12/03/2026` → resolved `2026-03-12T00:00:00`, expected `2026-12-03T00:00:00`
- `March 2026` → resolved `2026-03-01T00:00:00`, expected `2026-03-16T00:00:00`

**Finding.** On this corpus the rules baseline has higher strict F1 than the spaCy pipeline, and the hybrid adds recall while costing precision. That is the honest result: spaCy is a general-domain model being scored against templates written in our own style, so it over-generates. The hybrid is kept because recall matters more than precision for an exploratory graph, and because every span carries its extractor and confidence so a user can judge it. The ablation table in `extraction_eval.json` holds the error breakdown.

---

## 4. Risk indicators

Four indicators are detected and reported independently: **urgency**, **threat language**, **emotional pressure**, **payment request**. There is no single headline score anywhere in the product; the optional aggregate is returned with its formula and labelled an estimate.

### dev split (48 records)

| Method | Macro F1 | Precision (any) | Recall (any) | False-positive rate on benign records (95% CI) |
|---|---|---|---|---|
| `rules` | 0.809 | 0.588 | 1.000 | 0.184 [0.092, 0.334] |
| `tfidf-lr` | 0.900 | 1.000 | 1.000 | 0.000 [0.000, 0.092] |
| `distilbert` | — | — | — | not available |

### test split (48 records)

| Method | Macro F1 | Precision (any) | Recall (any) | False-positive rate on benign records (95% CI) |
|---|---|---|---|---|
| `rules` | 0.773 | 0.529 | 1.000 | 0.205 [0.108, 0.355] |
| `tfidf-lr` | 0.744 | 1.000 | 1.000 | 0.000 [0.000, 0.090] |
| `distilbert` | — | — | — | not available |

### Per-indicator detail on the test split

**`rules`**

| Indicator | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| urgency | 0.625 | 0.714 | 0.667 | 7 |
| threat_language | 0.875 | 0.875 | 0.875 | 8 |
| emotional_pressure | 0.889 | 1.000 | 0.941 | 8 |
| payment_request | 0.438 | 1.000 | 0.609 | 7 |
| any_indicator | 0.529 | 1.000 | 0.692 | 9 |

**`tfidf-lr`**

| Indicator | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| urgency | 0.750 | 0.857 | 0.800 | 7 |
| threat_language | 0.750 | 0.375 | 0.500 | 8 |
| emotional_pressure | 0.875 | 0.875 | 0.875 | 8 |
| payment_request | 0.750 | 0.857 | 0.800 | 7 |
| any_indicator | 1.000 | 1.000 | 1.000 | 9 |

### Calibration

| Method | Indicator | ECE | MCE | Brier |
|---|---|---|---|---|
| `rules` | urgency | 0.100 | 0.144 | 0.075 |
| `rules` | threat_language | 0.115 | 0.243 | 0.059 |
| `rules` | emotional_pressure | 0.012 | 0.066 | 0.019 |
| `rules` | payment_request | 0.135 | 0.428 | 0.071 |
| `tfidf-lr` | urgency | 0.073 | 0.110 | 0.039 |
| `tfidf-lr` | threat_language | 0.075 | 0.179 | 0.026 |
| `tfidf-lr` | emotional_pressure | 0.062 | 0.093 | 0.024 |
| `tfidf-lr` | payment_request | 0.090 | 0.045 | 0.043 |

ECE is the expected calibration error over 10 equal-width bins; MCE is the worst bin. Both are computed in-tree (`app/risk/calibration.py`) and unit-tested without torch installed.

> **The DistilBERT classifier was not evaluated in this run.** install requirements-ml.txt and run backend/scripts/train_risk.py --model distilbert. Every other number above stands without it; the app ships with the rules + TF-IDF path and reports the classifier as unavailable rather than pretending otherwise.

---

## 5. Entity resolution

Scored on a surface-form benchmark of 36 entities (288 variant forms) built from the corpus vocabulary: `canonical`, `upper case`, `lower case`, `title prefix`, `middle initial`, `email handle`, `surname typo`, `trailing punctuation`, `leading article`, `role suffix`, `lookalike domain`, `misspelled org token`.

| Metric | Value |
|---|---|
| Pairwise precision | 1.000 |
| Pairwise recall | 0.739 |
| Pairwise F1 | 0.850 |
| B-cubed F1 | 0.871 |
| False merges between must-not-merge pairs | 0.000 |

### Threshold sweep

| Strong / weak | Pairwise F1 | FP pairs | FN pairs | B-cubed F1 | False-merge rate |
|---|---|---|---|---|---|
| 1.01 / 1.01 | 0.430 | 0 | 732 | 0.534 | 0.000 |
| 0.99 / 0.95 | 0.699 | 0 | 466 | 0.747 | 0.000 |
| 0.95 / 0.90 | 0.842 | 0 | 275 | 0.865 | 0.000 |
| 0.92 / 0.86 | 0.850 | 0 | 263 | 0.871 | 0.000 |
| 0.88 / 0.80 | 0.857 | 0 | 253 | 0.877 | 0.000 |
| 0.85 / 0.75 | 0.857 | 0 | 253 | 0.877 | 0.000 |

### Ablations

| Configuration | Pairwise F1 | Δ |
|---|---|---|
| Full | 0.850 | — |
| Without containment and email-handle rules | 0.793 | 0.057 |
| Without the email-handle rule | 0.850 | 0.000 |

**Finding.** The token-containment rule is worth 0.057 pairwise F1. The email-handle rule scores 0 on its own: token containment already covers `first.last@domain` because the local part contributes the same content tokens. It is kept because it fires at 0.99 rather than 0.89 and so survives noise in the local part, but the report should not claim it as an independent win.

**A bug this evaluation caught.** The first run had a false-merge rate of 1.00: `L. Ashworth` and `N. Ashworth` were merging and taking `Lena Ashworth` and `Nadia Ashworth` with them. Two causes, both fixed in `app/entity/resolve.py`: containment ignored that the extra token was a *different person*, and the fuzzy path used `partial_ratio`, which scores 92% on any two names that share a long surname. The resolver now vetoes pairs whose leading tokens disagree (initials allowed) and uses whole-string `ratio` for people. Recorded in `docs/DECISIONS.md`.

---

## 6. Retrieval

99 chunks over 48 records, 72 queries. Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (fallback: False); FAISS available: True.

| Configuration | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| `bm25` | 0.507 | 0.681 | 0.792 | 0.875 | 0.858 | 0.879 |
| `embeddings` | 0.361 | 0.403 | 0.507 | 0.653 | 0.508 | 0.556 |
| `hybrid_rrf` | 0.438 | 0.611 | 0.722 | 0.833 | 0.800 | 0.786 |
| `hybrid_weighted` | 0.451 | 0.611 | 0.722 | 0.806 | 0.802 | 0.780 |

**Hybrid RRF broken down by query kind**

| Query kind | Queries | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| entity | 39 | 0.641 | 0.872 | 1.000 | 0.733 |
| paraphrase | 20 | 0.100 | 0.450 | 0.600 | 1.000 |
| topic | 13 | 0.346 | 0.692 | 0.692 | 0.692 |

### Limitations of this measurement

- Relevance is binary: a record is relevant or it is not. This suits 'which records mention X' and understates topical queries.
- Queries are derived from the same generator that wrote the records, so the corpus is easier than real user data would be.
- nDCG uses the same binary gains as Recall, so it mostly re-orders the Recall numbers rather than adding an independent view.

---

## 7. Error analysis

### Extraction errors, grouped by cause

| Cause | Count | Examples |
|---|---|---|
| project/period code tagged as a name | 10 | `[PERSON] 'Project Atlas'`; `[PERSON] 'Project Meridian'`; `[ORG] 'Project Meridian'` |
| organization suffix heuristic over-firing | 7 | `[ORG] 'Ingrid Abadi'`; `[ORG] 'SLA'`; `[ORG] 'Harbourline'` |
| entity missed (boundary or type disagreement) | 5 | `[PERSON] 'Ingrid Abadi'`; `[ORG] 'Account Services'`; `[ORG] 'Account Services'` |
| capitalisation heuristic over-firing | 3 | `[PERSON] 'Account Services'`; `[PERSON] 'Account Services'`; `[PERSON] 'Account Services'` |

Several of these are not model errors. PROJECT/period codes and numeric tokens are extracted with a low confidence and are not annotated as entities in the gold set, so they appear as false positives here.

### Risk

**Observation.** TF-IDF + logistic regression scores higher than the rule baseline.

The classifier memorised our own templates. The benign look-alikes reuse the same sentence frames as the attack records with a different middle clause, which is exactly the discrimination a bag-of-ngrams model is good at and a human is not fooled by. Treat the TF-IDF numbers as an upper bound that depends on the generator, not as evidence of generalisation.

*Mitigation:* The real public corpus is scored separately in risk_eval.json; that number is the one to trust about wording we did not write.

**Observation.** The rule baseline produces false positives on genuine urgent reminders.

'your account will be closed' and 'immediately' both appear in legitimate bank reminders. The mitigator list catches some of these and not all of them: a reminder that omits the 'no charge' sentence loses its mitigation.

*Mitigation:* The API reports each indicator separately with its evidence, so a user can see that the flag came from a policy sentence rather than a demand. Severity only reaches 'warning' when a payment demand is paired with a threat or a deadline.

### Retrieval

**Observation.** Recall@5: BM25 0.792 vs hybrid RRF 0.722.

Most queries in this benchmark are entity names lifted from the record, which is the strongest possible case for exact lexical matching and the weakest possible case for embeddings. Fusing a weaker channel with equal weight therefore costs recall on this corpus.

*Mitigation:* The per-kind breakdown shows where fusion does help. Keep the two channels separately selectable in Settings, and let the paraphrase queries - where BM25 has no lexical overlap at all - be the reason hybrid is the default.

**Observation.** Paraphrase queries: Recall@1 0.100 but MRR 1.000.

The right record is usually found, but several records are equally relevant so the first position is nearly a coin flip. Binary relevance makes nDCG look poor for the same reason.

*Mitigation:* Treat MRR as the meaningful number for topical queries and Recall@k for 'which records mention X'. Both are reported.

### Entity Resolution

**Observation.** Recall is lower than precision: P=1.000 vs R=0.739.

The resolver is deliberately conservative. A missed merge leaves two entities the user can inspect; a wrong merge silently asserts that two different people are the same, which is the more damaging error in a graph the user is meant to trust.

*Mitigation:* Clusters below the cohesion threshold are reported to the user as possibly-merged rather than being merged silently.

### What the real public corpus revealed

5000 messages (660 spam / 4340 ham). Indicators projected onto spam text: urgency = 78, threat_language = 1, emotional_pressure = 0, payment_request = 0.

- The projection onto our indicators is deliberately conservative: only 'urgency' survives on real spam text. Real SMS spam rarely makes an explicit threat or an explicit emotional appeal - it relies on scarcity and a prize instead.
- That is a genuine finding, not a bug. It means our four indicators cover the long-form business-email manipulation in Scenario A far better than they cover consumer SMS fraud, and the report should not claim otherwise.


---

## 8. Limitations

These are the limitations that matter for interpreting anything above.

### Synthetic-data bias
- The training and development corpora are generated from templates written for this
  project. Any model evaluated on them is partly being tested on its own training
  distribution.
- The consequence is visible in the numbers: TF-IDF + logistic regression scores far above
  the rule baseline, because the benign look-alikes reuse the attack records' sentence
  frames. That gap is **not** evidence that the classifier generalises; it is evidence that
  it can tell our two templates apart. The public-corpus figures are the ones that speak to
  wording we did not write.
- The benign look-alikes are the most important part of the corpus and also the most
  fragile. They cover urgency, deadlines, amounts and account vocabulary, but not: SMS
  phishing, non-English text, image-only documents, or coordinated multi-message campaigns
  where no single message is alarming on its own.

### False positives on legitimate urgent messages
- Real institutions use urgent language. "Your account will be paused", "act immediately"
  and "reply today" appear in genuine reminders, and a genuine reminder that omits the
  usual "there is no charge" sentence loses its mitigation and gets flagged.
- The false-positive rate on benign records is reported as a headline number next to every
  F1, because an F1 that hides it is not useful to a person deciding whether to trust a
  tool.
- Severity only reaches "warning" when a payment demand is paired with a threat or a hard
  deadline. Urgency alone stays "notice".
- Mitigating the rules (disambiguation of conditional consequences, legitimacy disclaimers)
  reduced the false positive on the demo reminder from a flag to none. That is one message,
  not a general guarantee.

### Language
- **English only.** spaCy is the English `en_core_web_sm` pipeline; the date patterns, the
  risk lexicon and the synthetic generator are all English. Nothing here has been tested on
  another language and it should not be presented as multilingual.
- The risk lexicon in particular encodes English-specific phrasing. A Portuguese or Bengali
  manipulation message would be invisible to the rules, and the classifiers were trained on
  English text.

### The risk model is a pattern matcher, not a judgement
- Every indicator is a lexical pattern. Four indicators are detected; social engineering
  that avoids all four - slow build-up over weeks, impersonating a known colleague, plain
  politeness with an unusual request - will not be flagged.
- The DistilBERT classifier learns from the same labels and the same corpus, so it inherits
  the same blind spots. A calibrated 0.9 means "the wording matches patterns that were
  labelled risky in our data", not "there is a 90% chance this is an attack".
- No claim is made about base rates. With a real corpus the proportion of genuinely
  malicious messages is far lower than 20%, and precision at any fixed threshold will be
  much worse than these numbers suggest.

### Entity resolution
- Precision is high and recall is lower by design. A missed merge leaves two entities the
  user can inspect; a wrong merge silently asserts that two people are the same.
- Clusters below the cohesion threshold are surfaced as "possibly merged" rather than
  merged silently, but they are still a suggestion.
- No coreference resolution. "She", "the account holder" and "Mr Delgado" are not linked to
  "Priya Raman".

### Retrieval
- Relevance is binary in these measurements, which suits "which records mention X" and
  understates topical queries.
- The benchmark queries are derived from the same generator that wrote the records, so the
  corpus is easier than real user data.
- BM25 outperforms fusion on this corpus because most queries are entity names. The
  paraphrase queries, where BM25 has no lexical overlap at all, are the reason hybrid
  remains the default.
- Embeddings are computed locally on CPU. A user with no `requirements-ml.txt` installed
  gets a hashing fallback that is **not** a semantic model; the UI labels it as such and
  the retrieval report records which backend produced the numbers.

### Timeline
- Ordering relies on source-system timestamps. A forwarded message carries the forward
  time, not the original send time, and we do not attempt to recover the original.
- Relative expressions ("last week") are resolved against the record's own timestamp and
  shown as a range. The range is our interpretation, not a fact, and the UI says so in the
  lane label and in each item's explanation.

### What this project does not do
- No action is taken on a record. Nothing is deleted, blocked, replied to or reported.
- No external service is contacted. There is no integration with any mail provider, and
  the import screen says so.
- The optional LLM layer is off by default and the application is fully functional without
  it. When enabled it may only rephrase an answer that already has citations, and citation
  integrity is verified after every rephrasing.

---

## 9. Reproducing this report

```
# one-time setup
pip install -r requirements.txt
pip install -r requirements-ml.txt        # optional heavy extras
python -m spacy download en_core_web_sm

# data and models
python -m backend.scripts.generate_dataset
python -m backend.scripts.fetch_public_data
python -m backend.scripts.train_risk --model tfidf
python -m backend.scripts.train_risk --model distilbert   # optional

# evaluation (each writes to reports/)
python -m backend.scripts.evaluate_demo_scenarios
python -m backend.scripts.evaluate_extraction
python -m backend.scripts.evaluate_entity_resolution
python -m backend.scripts.evaluate_risk
python -m backend.scripts.evaluate_retrieval
python -m backend.scripts.error_analysis
python -m backend.scripts.make_report
```

| Component | Version |
|---|---|
| python | 3.11.0 |
| platform | Windows |
| spacy | 3.8.4 |
| sklearn | 1.6.1 |
| torch | 2.5.1+cpu |
| transformers | 4.47.1 |
| faiss | 1.9.0 |
| sentence_transformers | 3.3.1 |
| spacy_model | 3.8.0 |

Every stage is deterministic given the seeds in the files above. `make_report.py` reads whatever exists in `reports/` and prints `skipped` with the command that would fill the gap, so an absent section is never mistaken for a good result.

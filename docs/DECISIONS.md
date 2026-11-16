# Architecture decisions

Each entry: what was decided, why, what was rejected, and what would change the decision.
Ordered roughly by when the decision was forced.

---

## 1. Python 3.11, pinned, not "3.11 or newer"

**Decided.** `requires-python = ">=3.11,<3.12"`.

**Why.** spaCy, torch, FAISS and transformers all ship platform-specific wheels whose
availability moves between Python versions. The first check of this project was whether the
whole stack was installable at all; on this machine Python 3.14 (the system default) had
no torch or FAISS wheels, while 3.11 had all of them. Pinning the minor version makes the
install reproducible rather than a weekly gamble.

**Rejected.** `>=3.11` — it looks friendlier and produces a broken install on any machine
whose default interpreter is newer. `3.10` — spaCy 3.8 supports it, but `X | None`
syntax throughout the codebase would need rewriting.

**Would change if.** spaCy and torch publish wheels for the current default interpreter
and the install is verified on a clean machine.

---

## 2. The app must work fully without the optional ML extras

**Decided.** `requirements.txt` contains only what the app needs to run. Everything heavy
(torch, transformers, sentence-transformers, FAISS, datasets) is in `requirements-ml.txt`
and is strictly optional. Every consumer of those packages checks availability and degrades:

| Missing | What happens |
|---|---|
| `en_core_web_sm` | Extraction falls back to the rules baseline; `/api/health` and the dashboard say `spacy_available: false`. |
| `sentence_transformers` / torch | A deterministic hashing embedder takes over. It is labelled `is_fallback: true` everywhere it surfaces, including the retrieval report. |
| FAISS | Exact numpy dot-product search over the same normalized vectors. Recall is identical — FAISS is a speed optimisation, not a quality one. |
| `transformers` | No DistilBERT classifier. Risk uses rules + TF-IDF, and the API reports `distilbert: {"status": "not available", "reason": ...}`. |

**Why.** "Local-first" is a promise about the user's machine, and the most likely machine is
one without a GPU and without patience for a 2 GB download. An app that requires the heavy
path to start cannot make that promise honestly. It also means the tests run in seconds.

**Cost.** Two code paths for retrieval. Both are exercised: the fallback by default in the
test suite, the real path by `evaluate_retrieval.py`.

---

## 3. `TransformerEmbedder` — mean pooling by hand, to avoid a blocked DLL

**Decided.** If `sentence_transformers` fails to import, fall back to
`transformers.AutoModel` plus attention-masked mean pooling and L2 normalisation, rather
than to the hashing embedder.

**Why.** `sentence_transformers` imports `datasets`, which imports `pyarrow`. On Windows
machines with an application-control policy, that DLL load fails with
`ImportError: ... blocked by an Application Control policy` — while the model itself is
completely usable. Mean pooling over the last hidden state with the attention mask is the
arithmetic sentence-transformers performs, so this is the same model, not a different one.

**Rejected.** Dropping the `datasets` dependency (it is only needed for training-set
loading, which this project does not use). Both fixes are worth doing; the in-tree
embedder also removes a heavyweight import from the app's startup path.

---

## 4. Probability calibration is implemented in-tree

**Decided.** `app/risk/calibration.py`: pool-adjacent-violators isotonic regression and
Platt scaling, both in numpy, plus ECE / MCE / Brier.

**Why.** Two reasons. The `calibration` PyPI package has no maintained wheel for Windows +
Python 3.11, and it would have failed the same way as everything else in that category.
More importantly, this project reports a number to a human who will act on it, so the
calibration step has to be auditable: `expected_calibration_error` is twenty lines and
unit-tested without torch installed. A dependency would have been a black box next to the
most trust-sensitive number in the product.

**Rejected.** `scikit-learn`'s `CalibratedClassifierCV` — it fits on probabilities from a
fitted estimator, which does not compose with a fine-tuned multi-label model whose
calibration set is the dev split.

---

## 5. Risk is reported as four independent indicators, never as one headline

**Decided.** `urgency`, `threat_language`, `emotional_pressure` and `payment_request` are
scored, thresholded and displayed separately. `overall_estimate()` exists but returns
`is_estimate: true` together with its formula, its per-indicator contributions, and a
caveat stating it is not a probability of fraud.

**Why.** A single score hides the thing the user actually needs: *why* this was flagged.
Two records can score identically and mean completely different things — one has an urgent
payment demand, the other has a threat with no money involved. Combining them destroys the
information the user needs to judge.

**The weighting is not hidden.** 0.35 / 0.30 / 0.20 / 0.15 is a judgement, stated in the
response payload and rendered next to the number. `test_product_rules.py` asserts the
formula and the weights are present, so the display cannot regress to a bare number.

**Severity is separate from the score.** A record reaches `warning` only when a payment
demand is paired with a threat or a hard deadline. Urgency alone stays `notice`, because
genuine reminders are urgent — the false-positive case the whole project is designed around.

---

## 6. The rules lexicon uses signal *and* mitigator weights

**Decided.** Each indicator has positive entries and negative ones. The score is
`1 - Π(1 - w_signal)` reduced by `Π(1 - |w_mitigator|)`.

**Why.** The first version scored only positives and flagged the genuine bank reminder in
the demo corpus, because it contains "the account will be paused until you visit a branch".
Real institutions write policy sentences; attackers write demands. The mitigators encode
that distinction explicitly — "we will never ask you to move money", "no charge for
verification", "visit a branch" — and each one is shown to the user with the text it matched.

**Also.** Bare payment coordinates ("sort code", "IBAN") carry weight 0.3, below the 0.4
threshold, because a genuine account-summary document is full of them. Only a *demand*
framing ("transfer $X", "payment is required", "release the funds") can raise the indicator
on its own.

**Cost.** The lexicon is English-specific and hand-maintained. Recorded as a limitation.

---

## 7. Timeline: three lanes, and inferred items are ranges

**Decided.** `recorded` (source-system timestamp), `explicit` (the text names an absolute
date), `inferred` (only relative wording). Inferred items render as bars and carry the raw
wording plus an explanation of how they were resolved.

**Why.** A single sorted list is where a naive implementation quietly lies: it puts "last
week" and "2026-03-04" next to each other as if they were the same kind of claim. "Last
week" does not name a day. Resolving it against the record's timestamp produces an
*interpretation*, and the UI says so.

**Ordering rule.** Inferred items sort by the *end* of their range, so a "last week" event
never jumps ahead of a fact that came later. A record with no source timestamp does not
fall back to its import time and call it a fact — it goes to the explicit lane with a
caveat, and `test_product_rules.py` asserts this.

---

## 8. Absence is reported as a search result, never as a fact about the world

**Decided.** One required sentence, in `app/retrieval/qa.py`:
`NO_MATCH = "No matching discussion was found in the records analyzed."`
Every empty result also returns `coverage` (what was searched, over what date range, from
which source types), a `caveat` naming the omission risk, and `no_match: true`.

**Why.** "I found no record of that conversation" and "that conversation did not happen" are
different claims, and only the first is supportable from a partial import. The coverage
payload exists so the user can judge how partial their data is.

**Three guards, because this is easy to break accidentally:**
- The semantic channel has a cosine floor (`MIN_SEMANTIC_SIMILARITY = 0.20`). Without it a
  normalized-vector index returns the k nearest chunks for *any* query, so nothing ever
  comes back empty and the honest absence wording becomes unreachable.
- A back-reference from the *earliest* record in the corpus produces
  `unverifiable_discussion_reference`, not `unresolved_discussion_reference` — there is
  nothing to search, and saying "no match" would imply a search happened.
- `FORBIDDEN_CLAIMS` is a deny-list checked against every generated string before it
  reaches the API, and asserted in both `test_product_rules.py` and
  `evaluate_demo_scenarios.py`.

---

## 9. Entity resolution: a leading-token veto, and no partial matching for people

**Decided.** Two rules added after the evaluation caught a bug:
- `_first_token_compatible` returns `False` (a hard veto, before any scoring) when two
  normalized forms have different leading tokens. An initial matches its expansion
  (`l` / `lena` agree, `l` / `n` do not).
- For `PERSON`, similarity uses whole-string `fuzz.ratio`. `token_set_ratio` and
  `partial_ratio` are never used for names.
- Token containment fires only when the *extra* tokens carry no identity (role words or
  single-letter initials).

**Why — the bug the evaluation found.** The first benchmark run scored pairwise precision
0.76, recall 0.81, and a **false-merge rate of 1.00**: `L. Ashworth` and `N. Ashworth`
merged, and the union-find chain took `Lena Ashworth` and `Nadia Ashworth` with them. Two
causes: middle initials were stripped from content tokens, so containment saw identical
token sets; and `partial_ratio("l ashworth", "n ashworth")` is 94.6% purely because the
surname is long. No threshold separates that from `Rafaela Fontaine` / `Rafael Fontaine`, a
genuine match — which is why the veto is structural rather than numeric.

After the fix: pairwise precision 1.00, false-merge rate 0.00, and the containment ablation
shows a real contribution (+0.057 F1).

**Kept despite a zero-delta ablation:** the email-handle rule
(`priya.raman@studio.example` → `priya raman`). Token containment already covers that case,
so the ablation shows 0.00. It is kept because it fires at 0.99 rather than 0.89 and so
survives noise in the local part. The report says this rather than claiming a win.

---

## 10. The entity-resolution benchmark is synthetic, by necessity

**Decided.** The corpus mentions each entity once, so scoring resolution on it directly
would be vacuous. Instead `evaluate_entity_resolution.py` generates a surface-form benchmark
from the corpus vocabulary: twelve variant kinds per entity, plus **distractor pairs** that
share a surname and must stay apart.

**Why.** A resolver that scores 1.0 because nothing ever looks alike is not measuring
anything. The distractor check (`distractor_false_merge_rate`) is reported next to pairwise
F1 for exactly this reason, and it is the metric that found the bug in decision 9.

**Limitation, stated.** The variants are generated by the same author as the resolver, so
the benchmark can flatter it. Recall being lower than precision is deliberate: a missed
merge leaves two entities the user can inspect; a wrong merge silently asserts that two
people are the same.

---

## 11. Hybrid fusion defaults to RRF, and both fusions are evaluated

**Decided.** Production uses Reciprocal Rank Fusion (`k=60`). `weighted_fuse` (min-max
normalised weighted sum) is implemented so the evaluation can compare them.

**Why RRF.** BM25 scores are unbounded and corpus-dependent; cosine similarities are
bounded in [0,1]. Any linear blend needs per-corpus tuning and breaks when the user adds
one record. RRF uses only ranks, so it is corpus-independent.

**Honest result.** On the synthetic corpus BM25 alone beats RRF fusion (Recall@5 0.79 vs
0.73), because most benchmark queries are entity names lifted from the record — the best
case for exact lexical matching and the worst case for embeddings. Fusion is retained for
the paraphrase queries, where BM25 has no lexical overlap at all, and the per-kind
breakdown is in the report so the trade-off is visible rather than asserted.

---

## 12. Character offsets are the contract, and the UI honours them

**Decided.** Every extracted span, evidence quote, citation and event stores integer
offsets into the record body. Nothing is stored as "the snippet", and record bodies are
preserved byte-for-byte by the parsers (no whitespace normalisation).

**Why.** Every highlight and every citation depends on it. The frontend renders the body as
a single untouched text node and draws highlight boxes from `Range.getClientRects()`
rather than wrapping text in `<mark>` — wrapping moves the DOM out from under the offsets
the moment two highlights overlap, which they do constantly.

**Enforcement.** `test_pipeline.py` asserts `body[start:end] == text` for every stored
mention; `enforce_citation_integrity` re-checks every citation against the source body after
answering, and `evaluate_demo_scenarios.py` asserts it for the live app.

---

## 13. Deleting a record deletes everything derived from it — and we verify it

**Decided.** Three layers: `ON DELETE CASCADE` in the schema, an explicit child-first sweep
in `app/privacy/__init__.py`, and `verify_record_purged()` which re-queries every derived
table and returns a list of problems.

**Why the explicit sweep.** SQLite's cascade depends on `PRAGMA foreign_keys=ON`, which is
per-connection and easy to lose in a refactor. A privacy guarantee must not depend on a
pragma. The sweep is child-first so it works even with foreign keys off, and the verification
means a bug here shows up as a failed API response and a red test, not as residue on disk.

**Orphaned entities** (those left with no mentions) are removed too, along with their
aliases. Insights that cited the deleted record are updated or deleted. Both indices are
invalidated.

---

## 14. Logs contain no record content

**Decided.** `app/logging_config.py` installs a `RedactingFilter` that drops messages
containing content-shaped keys (`body`, `text`, `prompt`, `subject`, ...) when they are
long. Helpers log counts and identifiers only: `ingested inserted=10 skipped=0 origin=demo`.

**Why.** Debug logging is where private data leaks. The filter is a backstop — the code
does not log content in the first place — but a third-party library logging a payload would
otherwise go straight to stderr.

---

## 15. The LLM layer is off, gated, and structurally unable to invent facts

**Decided.** `LlmPolicy` refuses by default; enabling requires both an explicit endpoint and
a UI disclosure. When enabled it may only *rephrase* an answer that already has citations,
and `enforce_citation_integrity` verifies the result afterwards.

**Why.** A generative layer is the fastest way to break every guarantee this project makes.
Rather than trusting a prompt, the architecture makes the failure detectable: the answer is
assembled from retrieved spans first, and any rephrasing is checked against the source text
before it is returned.

**The app works fully without it.** Nothing in the request path requires an LLM.

---

## 16. `docs/DECISIONS.md` records what the evaluation changed

The benchmark-driven fixes (decisions 6, 9, 11) are the ones worth keeping in mind when
reading the code: each of them is a case where a measured number contradicted the
implementation's intent, and the code was changed rather than the caveat being written.

---

## Appendix: rejected approaches

| Approach | Why not |
|---|---|
| Finetune spaCy's NER on the synthetic data | 320 records with template-generated entities. The model would learn the generator, and the gain over the rules baseline would not transfer to real email. |
| Generate answers with an LLM, retrieve for grounding only | Directly contradicts "a missing record is never proof that something did not happen" — a generator will produce a fluent sentence for any question. Retrieval assembles answers from quotes instead. |
| A single "risk score" in the dashboard | Decision 5. |
| Softmax over indicators to give an "overall risk class" | Same objection, dressed up. |
| Read `.mbox` / `.pst` directly | Beyond scope, and the brief lists the supported formats. |
| Connect to Gmail/Outlook via OAuth | The brief asks for a clear "integrations not configured" state. The import screen lists each integration, its status, and the export-then-upload alternative. |
| Trigram or dense retrieval without BM25 | BM25 is the baseline the brief requires, and on a small personal corpus exact matching is genuinely strong. |
| Store embeddings in FAISS only | An on-disk FAISS index can drift from the database after a delete. Vectors live in a table with `ON DELETE CASCADE`; FAISS is a derived, rebuildable cache. |

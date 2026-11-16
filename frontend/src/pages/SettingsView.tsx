import { useCallback, useEffect, useState } from "react";

import {
  deleteAllData,
  getSettings,
  reindex,
  rerunAnalysis,
  rerunResolution,
  updateSettings,
  type RetrievalMode,
  type SettingsOut,
} from "../lib/api";
import { formatBytes } from "../lib/format";
import {
  AvailabilityRow,
  Callout,
  Card,
  Disclosure,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  cx,
} from "../components/ui";

const RETRIEVAL_MODES: { value: RetrievalMode; label: string; note: string }[] = [
  { value: "lexical", label: "Lexical (BM25)", note: "Keyword matching only." },
  { value: "semantic", label: "Semantic (embeddings)", note: "Meaning-based similarity." },
  {
    value: "hybrid",
    label: "Hybrid",
    note: "Both channels fused, with per-channel counts shown in every result.",
  },
];

const CONFIRM_PHRASE = "delete all data";

export function SettingsView() {
  const [settings, setSettings] = useState<SettingsOut | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("hybrid");
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [confirmation, setConfirmation] = useState("");
  const [deleteOpen, setDeleteOpen] = useState(false);

  const load = useCallback(() => {
    setError(null);
    getSettings()
      .then((value) => {
        setSettings(value);
        setRetrievalMode(value.retrieval_mode);
        setMinConfidence(value.min_evidence_confidence);
      })
      .catch(setError);
  }, []);

  useEffect(load, [load]);

  const run = async (label: string, fn: () => Promise<unknown>, message: string) => {
    setBusy(label);
    setError(null);
    setNotice(null);
    try {
      await fn();
      setNotice(message);
      load();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  };

  if (error && !settings) {
    return (
      <>
        <PageHeader title="Settings" />
        <ErrorBlock error={error} onRetry={load} />
      </>
    );
  }

  if (!settings) {
    return (
      <>
        <PageHeader title="Settings" />
        <LoadingBlock label="Loading settings" />
      </>
    );
  }

  const { risk_models: models, spacy, embeddings, privacy, data_sources } = settings;

  return (
    <>
      <PageHeader
        eyebrow="Configuration"
        title="Settings"
        description="What is configured, what is available, and what leaving this machine would mean. Nothing here enables a connection on its own."
        actions={
          <button type="button" className="dm-btn" onClick={load}>
            Refresh
          </button>
        }
      />

      {error ? (
        <div className="mb-4">
          <ErrorBlock error={error} />
        </div>
      ) : null}
      {notice ? (
        <p
          role="status"
          className="mb-4 rounded-md border border-teal-300 bg-teal-50 px-4 py-3 text-sm text-teal-950"
        >
          {notice}
        </p>
      ) : null}

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Data sources" subtitle="Local imports only">
          {data_sources.length === 0 ? (
            <EmptyState title="No data source recorded">
              Nothing has been imported yet, so no source counts exist.
            </EmptyState>
          ) : (
            <ul className="space-y-2">
              {data_sources.map((source) => (
                <li
                  key={source.key}
                  className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-100 pb-2 last:border-0 last:pb-0"
                >
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-ink-900">{source.label}</p>
                    <p className="text-xs text-ink-600">
                      {source.kind}
                      {source.configured ? " · configured" : " · not configured"}
                    </p>
                  </div>
                  <span className="dm-chip border-ink-300 bg-white text-ink-800">
                    {source.record_count} record(s)
                  </span>
                </li>
              ))}
            </ul>
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              className="dm-btn"
              disabled={busy !== null}
              onClick={() =>
                void run("reindex", async () => {
                  await reindex();
                }, "Indexes rebuilt from the current records.")
              }
            >
              {busy === "reindex" ? "Rebuilding…" : "Rebuild search indexes"}
            </button>
            <button
              type="button"
              className="dm-btn"
              disabled={busy !== null}
              onClick={() =>
                void run("rerun", async () => {
                  await rerunAnalysis();
                }, "Analysis re-run over every record.")
              }
            >
              {busy === "rerun" ? "Re-running…" : "Re-run analysis"}
            </button>
            <button
              type="button"
              className="dm-btn"
              disabled={busy !== null}
              onClick={() =>
                void run("resolve", async () => {
                  await rerunResolution();
                }, "Entity resolution re-run.")
              }
            >
              {busy === "resolve" ? "Resolving…" : "Re-resolve entities"}
            </button>
          </div>
        </Card>

        <Card title="Analysis components" subtitle="Availability is reported as-is, including when a model is missing.">
          <ul>
            <AvailabilityRow
              name="Rules / lexicon baseline"
              available={models.rules.available}
              detail={models.rules.description}
            />
            <AvailabilityRow
              name="TF-IDF + logistic regression"
              available={models.tfidf_lr.available}
              detail={
                models.tfidf_lr.available
                  ? models.tfidf_lr.description
                  : `${models.tfidf_lr.description} — not trained, so it contributes nothing right now.`
              }
            />
            <AvailabilityRow
              name="DistilBERT classifier"
              available={models.distilbert.available}
              detail={
                models.distilbert.available
                  ? models.distilbert.description
                  : `${models.distilbert.description} — not installed, so risk detection relies on the rules baseline.`
              }
            />
            <AvailabilityRow
              name="spaCy named-entity recognition"
              available={spacy.available}
              detail={`${spacy.note} Model: ${spacy.model}.`}
            />
            <AvailabilityRow
              name="BM25 keyword index"
              available={true}
              detail="Always available; it is the lexical channel used by search and by answers."
            />
            <AvailabilityRow
              name="Sentence embeddings"
              available={embeddings.available}
              detail={
                embeddings.fallback
                  ? `Not installed; a hashing fallback is used instead, which is weaker. ${embeddings.indexed} vector(s) indexed.`
                  : `${embeddings.indexed} vector(s) indexed with ${embeddings.model ?? "an unspecified model"}.`
              }
            />
            <AvailabilityRow
              name="FAISS vector index"
              available={embeddings.faiss}
              detail={
                embeddings.faiss
                  ? "Installed; similarity search uses FAISS."
                  : "Not installed; similarity search falls back to a plain in-memory scan, which is slower but not less private."
              }
            />
          </ul>
          {models.notes.length > 0 ? (
            <div className="mt-3 space-y-1">
              {models.notes.map((note) => (
                <p key={note} className="text-xs text-ink-600">
                  {note}
                </p>
              ))}
            </div>
          ) : null}
          <Disclosure summary="Thresholds in use">
            <ul className="space-y-1 text-xs">
              {models.tfidf_lr.thresholds
                ? Object.entries(models.tfidf_lr.thresholds).map(([label, value]) => (
                    <li key={label}>
                      TF-IDF {label}: {value.toFixed(2)}
                    </li>
                  ))
                : (
                    <li>TF-IDF thresholds are unavailable because the model is not loaded.</li>
                  )}
              {models.distilbert.thresholds
                ? Object.entries(models.distilbert.thresholds).map(([label, value]) => (
                    <li key={label}>
                      DistilBERT {label}: {value.toFixed(2)}
                    </li>
                  ))
                : (
                    <li>DistilBERT thresholds are unavailable because the model is not loaded.</li>
                  )}
            </ul>
          </Disclosure>
        </Card>
      </div>

      <Card
        className="mt-5"
        title="Retrieval and evidence"
        subtitle="These two settings change what the search and answer views will show."
      >
        <div className="grid gap-5 lg:grid-cols-2">
          <fieldset>
            <legend className="dm-section-title">Retrieval mode</legend>
            <div className="mt-2 space-y-2">
              {RETRIEVAL_MODES.map((option) => (
                <label
                  key={option.value}
                  className={cx(
                    "flex cursor-pointer items-start gap-2 rounded-md border px-3 py-2",
                    retrievalMode === option.value
                      ? "border-teal-600 bg-teal-50"
                      : "border-ink-200 bg-white",
                  )}
                >
                  <input
                    type="radio"
                    name="retrieval-mode"
                    className="mt-1 h-4 w-4"
                    checked={retrievalMode === option.value}
                    onChange={() => setRetrievalMode(option.value)}
                  />
                  <span>
                    <span className="block text-sm font-medium text-ink-900">{option.label}</span>
                    <span className="block text-xs text-ink-600">{option.note}</span>
                  </span>
                </label>
              ))}
            </div>
            <button
              type="button"
              className="dm-btn dm-btn-primary mt-3"
              disabled={busy !== null || retrievalMode === settings.retrieval_mode}
              onClick={() =>
                void run(
                  "settings",
                  async () => {
                    await updateSettings({ retrieval_mode: retrievalMode });
                  },
                  `Retrieval mode set to ${retrievalMode}.`,
                )
              }
            >
              Save retrieval mode
            </button>
          </fieldset>

          <div>
            <label htmlFor="min-confidence" className="dm-section-title">
              Minimum evidence confidence
            </label>
            <p className="mt-1 text-xs text-ink-600">
              Low-confidence evidence stays visible but is de-emphasised. Raising this hides weak
              citations rather than inventing stronger ones.
            </p>
            <div className="mt-3 flex items-center gap-3">
              <input
                id="min-confidence"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={minConfidence}
                onChange={(event) => setMinConfidence(Number(event.target.value))}
                className="w-full"
                aria-valuetext={`${minConfidence.toFixed(2)}`}
              />
              <output
                htmlFor="min-confidence"
                className="w-14 shrink-0 text-right text-sm tabular-nums text-ink-900"
              >
                {minConfidence.toFixed(2)}
              </output>
            </div>
            <button
              type="button"
              className="dm-btn dm-btn-primary mt-3"
              disabled={busy !== null || minConfidence === settings.min_evidence_confidence}
              onClick={() =>
                void run(
                  "settings",
                  async () => {
                    await updateSettings({ min_evidence_confidence: minConfidence });
                  },
                  `Minimum evidence confidence set to ${minConfidence.toFixed(2)}.`,
                )
              }
            >
              Save minimum confidence
            </button>
          </div>
        </div>
      </Card>

      <Card className="mt-5" title="Privacy" subtitle="Counts only. No record content is read to build this panel.">
        <div className="grid gap-5 lg:grid-cols-2">
          <div>
            <p className="dm-section-title">Stored locally</p>
            <ul className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
              {Object.entries(privacy.counts).map(([name, count]) => (
                <li key={name} className="rounded-md border border-ink-200 bg-white px-3 py-2">
                  <p className="text-xs text-ink-600">{name.replace(/_/g, " ")}</p>
                  <p className="text-lg font-semibold tabular-nums text-ink-900">{count}</p>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-ink-600">
              Database size {formatBytes(privacy.database_bytes)} at{" "}
              <span className="font-mono">{privacy.database_path}</span>
            </p>
          </div>

          <div className="space-y-3">
            <Callout
              tone={privacy.external_transmission ? "risk" : "teal"}
              title={
                privacy.external_transmission
                  ? "External transmission is enabled"
                  : "No external transmission"
              }
            >
              <p>{privacy.llm.disclosure}</p>
            </Callout>
            <div>
              <p className="dm-section-title">Optional external LLM</p>
              <p className="mt-1 text-sm text-ink-800">
                {privacy.llm.enabled
                  ? `Enabled, pointing at ${privacy.llm.endpoint ?? "an unspecified endpoint"} with model ${privacy.llm.model ?? "unspecified"}.`
                  : "Disabled. This is the default and DigitalMind is fully functional without it."}
              </p>
              <p className="mt-1 text-xs text-ink-600">{privacy.llm.note}</p>
              {privacy.llm.enabled ? (
                <button
                  type="button"
                  className="dm-btn mt-2"
                  disabled={busy !== null}
                  onClick={() =>
                    void run(
                      "llm",
                      async () => {
                        await updateSettings({ allow_external_llm: false });
                      },
                      "External LLM layer disabled.",
                    )
                  }
                >
                  Disable the external LLM layer
                </button>
              ) : (
                <p className="mt-1 text-xs text-ink-600">
                  It can only be enabled on the server, because turning it on is a decision about
                  where your record text is allowed to go.
                </p>
              )}
            </div>
            <div>
              <p className="dm-section-title">Log policy</p>
              <p className="mt-1 text-sm text-ink-800">{privacy.log_policy}</p>
            </div>
            <div>
              <p className="dm-section-title">Delete semantics</p>
              <p className="mt-1 text-sm text-ink-800">{privacy.delete_semantics}</p>
            </div>
          </div>
        </div>
      </Card>

      <Card
        className="mt-5"
        title="Delete all data"
        subtitle="Removes every record and everything derived from it."
      >
        <p className="text-sm text-ink-700">
          This deletes records, entities, graph edges, events, risk assessments, insights and index
          entries from the local database. There is no undo and no backup taken for you.
        </p>
        {!deleteOpen ? (
          <button
            type="button"
            className="dm-btn dm-btn-risk mt-3"
            onClick={() => setDeleteOpen(true)}
            aria-expanded={deleteOpen}
          >
            Delete all data
          </button>
        ) : (
          <div
            role="alert"
            className="mt-3 rounded-md border-2 border-risk-400 bg-risk-50 px-4 py-4"
          >
            <p className="text-sm font-semibold text-risk-950">
              This cannot be undone. Type the phrase below to confirm.
            </p>
            <label htmlFor="delete-confirmation" className="mt-3 block text-sm text-risk-950">
              Type <span className="font-mono font-semibold">{CONFIRM_PHRASE}</span> to confirm
            </label>
            <input
              id="delete-confirmation"
              className="dm-input mt-1"
              value={confirmation}
              onChange={(event) => setConfirmation(event.target.value)}
              autoComplete="off"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                className="dm-btn dm-btn-risk"
                disabled={confirmation !== CONFIRM_PHRASE || busy !== null}
                onClick={() =>
                  void run("delete", async () => {
                    const outcome = await deleteAllData();
                    setNotice(outcome.message);
                    setConfirmation("");
                    setDeleteOpen(false);
                  }, "All local data deleted.")
                }
              >
                {busy === "delete" ? "Deleting…" : "Delete everything permanently"}
              </button>
              <button
                type="button"
                className="dm-btn"
                onClick={() => {
                  setDeleteOpen(false);
                  setConfirmation("");
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </Card>
    </>
  );
}

import { useCallback, useEffect, useState } from "react";

import { listInsights, updateInsight, type Insight, type InsightStatus } from "../lib/api";
import { formatDateTime, formatScore } from "../lib/format";
import { recordLink } from "../lib/router";
import {
  Card,
  Disclosure,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  cx,
} from "../components/ui";

const FILTERS: { key: "active" | "dismissed" | "corrected" | "all"; label: string }[] = [
  { key: "active", label: "Active" },
  { key: "dismissed", label: "Dismissed" },
  { key: "corrected", label: "Corrected by you" },
  { key: "all", label: "All" },
];

const SEVERITY_TONE: Record<string, string> = {
  warning: "border-risk-400 bg-risk-50 text-risk-950",
  notice: "border-risk-200 bg-white text-risk-950",
  info: "border-teal-300 bg-white text-teal-950",
};

export function InsightsView() {
  const [status, setStatus] = useState<"active" | "dismissed" | "corrected" | "all">("active");
  const [items, setItems] = useState<Insight[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    listInsights({ status })
      .then(setItems)
      .catch(setError);
  }, [status]);

  useEffect(load, [load]);

  const patch = async (id: number, payload: { status?: InsightStatus; user_note?: string }) => {
    setBusyId(id);
    setActionError(null);
    try {
      const updated = await updateInsight(id, payload);
      setItems((current) =>
        current
          ? current.map((item) => (item.id === updated.id ? updated : item))
          : [updated],
      );
      return updated;
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "The change could not be saved. Please try again.",
      );
      return null;
    } finally {
      setBusyId(null);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Cross-record conclusions"
        title="Insights"
        description="Each insight is a statement the analysis produced across records, with its cited facts and the basis for its confidence. Anything you disagree with can be dismissed or corrected."
        actions={
          <div role="group" aria-label="Filter insights by status" className="flex flex-wrap gap-1">
            {FILTERS.map((filter) => (
              <button
                key={filter.key}
                type="button"
                aria-pressed={status === filter.key}
                onClick={() => setStatus(filter.key)}
                className={cx(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  status === filter.key
                    ? "bg-teal-700 text-white"
                    : "bg-white text-ink-700 hover:bg-ink-100",
                )}
              >
                {filter.label}
              </button>
            ))}
          </div>
        }
      />

      {actionError ? (
        <div role="alert" className="mb-4 rounded-md border border-risk-300 bg-risk-50 px-4 py-3 text-sm text-risk-950">
          {actionError}
        </div>
      ) : null}

      {error ? <ErrorBlock error={error} onRetry={load} /> : null}
      {!error && !items ? <LoadingBlock label="Loading insights" /> : null}

      {items && items.length === 0 ? (
        <EmptyState title={`No ${status === "all" ? "" : status} insight`}>
          Insights are generated from the records currently imported. There is nothing in this state
          to show, which is not evidence that nothing is happening in your records.
        </EmptyState>
      ) : null}

      {items && items.length > 0 ? (
        <ul className="space-y-4">
          {items.map((insight) => (
            <li key={insight.id}>
              <InsightCard
                insight={insight}
                busy={busyId === insight.id}
                onDismiss={() => void patch(insight.id, { status: "dismissed" })}
                onRestore={() => void patch(insight.id, { status: "active" })}
                onCorrect={(note) => patch(insight.id, { status: "corrected", user_note: note })}
              />
            </li>
          ))}
        </ul>
      ) : null}
    </>
  );
}

function InsightCard({
  insight,
  busy,
  onDismiss,
  onRestore,
  onCorrect,
}: {
  insight: Insight;
  busy: boolean;
  onDismiss: () => void;
  onRestore: () => void;
  onCorrect: (note: string) => Promise<Insight | null>;
}) {
  const [correcting, setCorrecting] = useState(false);
  const [note, setNote] = useState(insight.user_note ?? "");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setNote(insight.user_note ?? "");
  }, [insight.user_note]);

  const hasSpanFacts = insight.facts.some(
    (fact) => typeof fact.start_char === "number" && typeof fact.end_char === "number",
  );

  return (
    <Card
      title={
        <span className="flex flex-wrap items-center gap-2">
          <span>{insight.title}</span>
          <span className={cx("dm-chip", SEVERITY_TONE[insight.severity] ?? "border-ink-300 bg-white text-ink-800")}>
            {insight.severity}
          </span>
          <span className="dm-chip border-ink-300 bg-white text-ink-800">{insight.kind.replace(/_/g, " ")}</span>
          <span
            className={cx(
              "dm-chip",
              insight.status === "active"
                ? "border-teal-300 bg-teal-50 text-teal-900"
                : insight.status === "dismissed"
                  ? "border-ink-300 bg-ink-100 text-ink-800"
                  : "border-blue-300 bg-blue-50 text-blue-900",
            )}
          >
            {insight.status}
          </span>
        </span>
      }
      subtitle={`Created ${formatDateTime(insight.created_at)} · ${insight.record_ids.length} record(s) referenced`}
    >
      <p className="text-sm leading-6 text-ink-900">{insight.statement}</p>

      <div className="mt-3 rounded-md border border-ink-200 bg-ink-50 px-3 py-2">
        <p className="dm-section-title">Confidence</p>
        <p className="mt-1 text-sm tabular-nums text-ink-900">
          {formatScore(insight.confidence)}
          <span className="ml-2 font-normal text-ink-700">{insight.confidence_basis}</span>
        </p>
      </div>

      {insight.facts.length > 0 ? (
        <div className="mt-3">
          <Disclosure
            summary={`Cited facts (${insight.facts.length})${
              hasSpanFacts ? " — click one to open the record at that passage" : ""
            }`}
          >
            <ul className="space-y-2">
              {insight.facts.map((fact, index) => {
                const linked =
                  typeof fact.start_char === "number" && typeof fact.end_char === "number";
                return (
                  <li key={`${fact.record_id}-${index}`}>
                    {linked ? (
                      <a
                        href={`#${recordLink(fact.record_id, fact.start_char, fact.end_char)}`}
                        className="block rounded-md border border-ink-200 bg-white px-3 py-2 text-sm hover:border-teal-600 hover:bg-teal-50"
                      >
                        <span className="italic text-ink-900">
                          “{fact.quote ?? `Record ${fact.record_id}`}”
                        </span>
                        <span className="mt-1 block text-xs text-ink-600">
                          record {fact.record_id} · characters {fact.start_char}–{fact.end_char}
                          {fact.note ? ` · ${fact.note}` : ""}
                        </span>
                      </a>
                    ) : (
                      <div className="rounded-md border border-ink-200 bg-white px-3 py-2 text-sm">
                        <a
                          href={`#${recordLink(fact.record_id)}`}
                          className="font-medium text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
                        >
                          record {fact.record_id}
                        </a>
                        {fact.note ? <span className="ml-2 text-xs text-ink-600">{fact.note}</span> : null}
                        <span className="mt-1 block text-xs text-ink-600">
                          This fact has no character range, so it opens the record without a
                          highlighted passage.
                        </span>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </Disclosure>
        </div>
      ) : (
        <p className="mt-3 text-xs text-ink-600">
          This insight carries no cited facts. Treat it as an unverified observation, not a finding.
        </p>
      )}

      {insight.user_note ? (
        <div className="mt-3 rounded-md border border-blue-300 bg-blue-50 px-3 py-2">
          <p className="dm-section-title">Your correction</p>
          <p className="mt-1 text-sm text-blue-950">{insight.user_note}</p>
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-ink-100 pt-3">
        {insight.status !== "dismissed" ? (
          <button type="button" className="dm-btn" onClick={onDismiss} disabled={busy}>
            Dismiss
          </button>
        ) : (
          <button type="button" className="dm-btn" onClick={onRestore} disabled={busy}>
            Restore
          </button>
        )}
        <button
          type="button"
          className="dm-btn"
          aria-expanded={correcting}
          onClick={() => setCorrecting((value) => !value)}
          disabled={busy}
        >
          {correcting ? "Close correction" : "Correct"}
        </button>
        {insight.status === "dismissed" && insight.user_note ? (
          <button type="button" className="dm-btn" onClick={onRestore} disabled={busy}>
            Restore to active
          </button>
        ) : null}
      </div>

      {correcting ? (
        <form
          className="mt-3 rounded-md border border-ink-300 bg-ink-50 px-3 py-3"
          onSubmit={async (event) => {
            event.preventDefault();
            const updated = await onCorrect(note.trim());
            if (updated) {
              setSaved(true);
              setCorrecting(false);
            }
          }}
        >
          <label htmlFor={`note-${insight.id}`} className="block text-sm font-medium text-ink-800">
            What is wrong with this insight?
          </label>
          <textarea
            id={`note-${insight.id}`}
            className="dm-input mt-1 min-h-[5rem]"
            maxLength={2000}
            value={note}
            onChange={(event) => {
              setNote(event.target.value);
              setSaved(false);
            }}
            placeholder="for example: these are two different suppliers, not one account"
          />
          <p className="mt-1 text-xs text-ink-600">
            Saving marks the insight as corrected and keeps your note next to it. The original
            statement is not erased.
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            <button type="submit" className="dm-btn dm-btn-primary" disabled={busy || note.trim() === ""}>
              {busy ? "Saving…" : "Save correction"}
            </button>
            <button type="button" className="dm-btn" onClick={() => setCorrecting(false)}>
              Cancel
            </button>
            {saved ? (
              <span role="status" className="self-center text-sm text-teal-800">
                Saved.
              </span>
            ) : null}
          </div>
        </form>
      ) : null}
    </Card>
  );
}

import { useCallback, useEffect, useState } from "react";

import type { Dashboard, Insight } from "../lib/api";
import { getDashboard, getRisk, listInsights } from "../lib/api";
import { riskLabelDisplay } from "../lib/format";
import { useNavigate } from "../lib/router";
import {
  AvailabilityRow,
  Callout,
  Card,
  Disclosure,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  RiskChip,
  Stat,
  cx,
} from "../components/ui";

const INDICATOR_ORDER = [
  "payment_request",
  "threat_language",
  "urgency",
  "emotional_pressure",
] as const;

const INDICATOR_NOTE: Record<string, string> = {
  urgency: "Deadlines and time pressure stated in the wording.",
  threat_language: "Consequences stated as certain, such as an account closing.",
  emotional_pressure: "Fear, pleading or personal appeals aimed at the reader.",
  payment_request: "A demand that money move.",
};

export function DashboardView() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState<unknown>(null);
  const navigate = useNavigate();
  const load = useCallback(() => {
    setError(null);
    getDashboard()
      .then(setData)
      .catch(setError);
  }, []);

  useEffect(load, [load]);

  if (error) {
    return (
      <>
        <PageHeader title="Dashboard" />
        <ErrorBlock error={error} onRetry={load} />
      </>
    );
  }

  if (!data) {
    return (
      <>
        <PageHeader title="Dashboard" />
        <LoadingBlock label="Loading dashboard" />
      </>
    );
  }

  const { records, risk, graph, timeline, insights, models, privacy } = data;

  return (
    <>
      <PageHeader
        eyebrow="Overview"
        title="Dashboard"
        description="Everything below is computed from the records currently imported on this machine. Nothing is sent anywhere."
        actions={
          <>
            <button type="button" className="dm-btn" onClick={load}>
              Refresh
            </button>
            <button type="button" className="dm-btn dm-btn-primary" onClick={() => navigate("/import")}>
              Import data
            </button>
          </>
        }
      />

      {risk.records_with_indicators > 0 ? (
        <div
          role="alert"
          className="mb-6 rounded-md border border-risk-300 bg-risk-50 px-4 py-4 text-sm text-risk-950"
        >
          <p className="text-sm font-semibold">Potential manipulation risk detected</p>
          <p className="mt-1">
            {risk.records_with_indicators} record(s) contain at least one wording indicator (
            {risk.warning_records} at warning level, {risk.notice_records} at notice level).
          </p>
          <p className="mt-1">{risk.disclaimer}</p>
          <button
            type="button"
            className="dm-btn dm-btn-risk mt-3"
            onClick={() => navigate("/risk")}
          >
            Review the indicators
          </button>
        </div>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Records" value={records.total} hint="Imported into the local database" />
        <Stat
          label="Records with indicators"
          value={risk.records_with_indicators}
          hint="At least one indicator present"
          tone={risk.records_with_indicators > 0 ? "risk" : "neutral"}
        />
        <Stat
          label="Entities"
          value={graph.entities}
          hint={`${graph.relations} graph edge(s)`}
          tone="teal"
        />
        <Stat label="Timeline events" value={timeline.events} hint="Recorded, explicit and inferred" />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Card
            title="Indicators, reported separately"
            subtitle="Four independent checks. There is no combined verdict, by design."
          >
            <div className="grid gap-4 sm:grid-cols-2">
              {INDICATOR_ORDER.map((label) => {
                const count = risk.indicator_counts[label] ?? 0;
                const flagged = count > 0;
                return (
                  <div
                    key={label}
                    className={cx(
                      "rounded-md border px-4 py-3",
                      flagged ? "border-risk-300 bg-risk-50" : "border-ink-200 bg-white",
                    )}
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <h3 className="text-sm font-semibold text-ink-900">
                        {riskLabelDisplay(label)}
                      </h3>
                      <span
                        className={cx(
                          "text-xl font-semibold tabular-nums",
                          flagged ? "text-risk-800" : "text-ink-500",
                        )}
                      >
                        {count}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-ink-600">
                      {flagged
                        ? `${count} record(s) contain this wording. ${INDICATOR_NOTE[label]}`
                        : `No record currently contains this wording. ${INDICATOR_NOTE[label]} This is an absence of matches in the analyzed data, not proof that nothing happened.`}
                    </p>
                  </div>
                );
              })}
            </div>
            <p className="mt-4 rounded-md border border-ink-200 bg-ink-50 px-3 py-2 text-xs text-ink-700">
              {risk.disclaimer}
            </p>
          </Card>
        </div>

        <Card title="Records by source type" subtitle="What has actually been imported">
          {Object.keys(records.by_source_type).length === 0 ? (
            <EmptyState title="No records imported yet">
              Load the demo corpus or upload files to populate the workspace. Nothing is fetched
              from an outside service.
            </EmptyState>
          ) : (
            <ul className="space-y-2">
              {Object.entries(records.by_source_type)
                .sort((a, b) => b[1] - a[1])
                .map(([sourceType, count]) => (
                  <li
                    key={sourceType}
                    className="flex items-center justify-between border-b border-ink-100 pb-2 last:border-0"
                  >
                    <span className="text-sm text-ink-800">{sourceType}</span>
                    <span className="text-sm font-semibold tabular-nums text-ink-900">{count}</span>
                  </li>
                ))}
            </ul>
          )}
          <button
            type="button"
            className="dm-btn mt-4 w-full"
            onClick={() => navigate("/records")}
          >
            Open the records inbox
          </button>
        </Card>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card
          title="Active insights"
          subtitle={`${insights.active} active · ${insights.dismissed} dismissed · ${insights.corrected} corrected`}
          actions={
            <button type="button" className="dm-btn" onClick={() => navigate("/insights")}>
              View all
            </button>
          }
        >
          <InsightsPreview />
        </Card>

        <Card
          title="Model availability"
          subtitle="Shown honestly: an unavailable component reduces what the analysis can claim."
        >
          <ul>
            <AvailabilityRow
              name="Rules baseline"
              available={models.risk.rules.available}
              detail={models.risk.rules.description}
            />
            <AvailabilityRow
              name="TF-IDF + logistic regression"
              available={models.risk.tfidf_lr.available}
              detail={models.risk.tfidf_lr.description}
            />
            <AvailabilityRow
              name="DistilBERT classifier"
              available={models.risk.distilbert.available}
              detail={models.risk.distilbert.description}
            />
            <AvailabilityRow
              name="spaCy named-entity recognition"
              available={models.ner.available}
              detail={models.ner.note}
            />
            <AvailabilityRow
              name="BM25 keyword index"
              available={models.retrieval.bm25.available}
              detail={`${models.retrieval.bm25.indexed_chunks} indexed chunk(s)`}
            />
            <AvailabilityRow
              name="Sentence embeddings"
              available={models.retrieval.embeddings.available}
              detail={
                models.retrieval.embeddings.fallback
                  ? `hashing fallback in use (${models.retrieval.embeddings.indexed} embedding(s))`
                  : `${models.retrieval.embeddings.indexed} embedding(s)`
              }
            />
            <AvailabilityRow
              name="FAISS vector index"
              available={models.retrieval.embeddings.faiss}
              detail="Faster similarity search when installed."
            />
          </ul>
          {models.risk.notes.length > 0 ? (
            <div className="mt-3 space-y-1">
              {models.risk.notes.map((note) => (
                <p key={note} className="text-xs text-ink-600">
                  {note}
                </p>
              ))}
            </div>
          ) : null}
        </Card>
      </div>

      <div className="mt-6">
        <Card title="Privacy" subtitle={privacy.summary}>
          <div className="grid gap-4 sm:grid-cols-3">
            <Callout tone="teal" title="Local analysis">
              <p>
                Analysis runs locally. No record content leaves this machine, and the optional
                external LLM layer is off by default.
              </p>
            </Callout>
            <Callout title="Third-party integrations">
              <p>
                None are configured. DigitalMind only reads the files you upload and the built-in
                demo corpus.
              </p>
            </Callout>
            <div className="rounded-md border border-ink-200 px-3 py-3">
              <p className="dm-section-title">Insight status</p>
              <ul className="mt-2 space-y-1 text-sm text-ink-800">
                <li>Active: {insights.active}</li>
                <li>Dismissed: {insights.dismissed}</li>
                <li>Corrected by you: {insights.corrected}</li>
              </ul>
            </div>
          </div>
          <Disclosure summary="What “no indicators” does and does not mean">
            <p>
              Every indicator count on this page is the number of records whose wording matched a
              check. A record with no indicators has not been shown to contain these patterns in
              the text that was analyzed. It is not a statement that the record is safe, and a
              missing match is not evidence that nothing happened.
            </p>
          </Disclosure>
          <button
            type="button"
            className="dm-btn mt-4"
            onClick={() => navigate("/settings")}
          >
            Model and data settings
          </button>
        </Card>
      </div>

      <div className="mt-6">
        <Card
          title="Records carrying indicators"
          subtitle="Sorted by level. Each row links to the exact wording that triggered it."
        >
          <FlaggedPreview />
        </Card>
      </div>
    </>
  );
}

function InsightsPreview() {
  const [rows, setRows] = useState<
    { id: number; title: string; statement: string; confidence: number; confidence_basis: string }[]
  | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    listInsights({ status: "active" })
      .then((items: Insight[]) => {
        if (cancelled) return;
        setRows(
          items.slice(0, 3).map((item) => ({
            id: item.id,
            title: item.title,
            statement: item.statement,
            confidence: item.confidence,
            confidence_basis: item.confidence_basis,
          })),
        );
      })
      .catch(() => {
        if (!cancelled) setRows([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!rows) return <LoadingBlock label="Loading insights" />;
  if (rows.length === 0) {
    return (
      <EmptyState title="No active insights">
        Insights are cross-record conclusions. They appear once records have been analyzed; they do
        not appear when there is nothing to connect.
      </EmptyState>
    );
  }
  return (
    <ul className="space-y-3">
      {rows.map((row) => (
        <li key={row.id} className="border-b border-ink-100 pb-3 last:border-0 last:pb-0">
          <p className="text-sm font-medium text-ink-900">{row.title}</p>
          <p className="mt-1 line-clamp-3 text-sm text-ink-700">{row.statement}</p>
          <p className="mt-1 text-xs text-ink-600">
            Confidence {row.confidence.toFixed(2)} · {row.confidence_basis}
          </p>
        </li>
      ))}
      <li>
        <button type="button" className="dm-btn" onClick={() => navigate("/insights")}>
          Review, dismiss or correct an insight
        </button>
      </li>
    </ul>
  );
}

interface FlaggedRow {
  id: number;
  title: string;
  source_type: string;
  severity: string;
  labels: string[];
}

function FlaggedPreview() {
  const [state, setState] = useState<
    { loading: true; error: null; rows: null } | { loading: false; error: unknown; rows: null } | { loading: false; error: null; rows: FlaggedRow[] }
  >({ loading: true, error: null, rows: null });
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    getRisk()
      .then((risk) => {
        if (cancelled) return;
        setState({
          loading: false,
          error: null,
          rows: risk.records.slice(0, 5).map((r) => ({
            id: r.record_id,
            title: r.title,
            source_type: r.source_type,
            severity: r.severity,
            labels: r.indicators.filter((i) => i.present).map((i) => i.label),
          })),
        });
      })
      .catch((error: unknown) => {
        if (!cancelled) setState({ loading: false, error, rows: null });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (state.loading) return <LoadingBlock label="Loading flagged records" />;
  if (state.error) return <ErrorBlock error={state.error} />;
  const rows = state.rows ?? [];
  if (rows.length === 0) {
    return (
      <EmptyState title="No record currently carries an indicator">
        No wording match was found in the records analyzed. That is not a clean bill of health for
        any sender.
      </EmptyState>
    );
  }
  return (
    <ul className="divide-y divide-ink-100">
      {rows.map((row) => (
        <li key={row.id} className="flex flex-wrap items-center justify-between gap-3 py-3">
          <div className="min-w-0">
            <a
              href={`#/record/${row.id}`}
              className="text-sm font-medium text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
            >
              {row.title}
            </a>
            <p className="mt-1 text-xs text-ink-600">{row.source_type}</p>
          </div>
          <RiskChip labels={row.labels} severity={row.severity as "none" | "notice" | "warning"} />
        </li>
      ))}
      <li className="pt-3">
        <button type="button" className="dm-btn" onClick={() => navigate("/risk")}>
          See the full risk analysis
        </button>
      </li>
    </ul>
  );
}

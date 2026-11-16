import { useCallback, useEffect, useState } from "react";

import { getRisk, type RiskOverview, type RiskRecord } from "../lib/api";
import { formatDateTime, formatScore } from "../lib/format";
import { recordLink } from "../lib/router";
import { EvidenceQuote } from "./RecordDetailView";
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
  SeverityBadge,
  SourceBadge,
  cx,
} from "../components/ui";

const INDICATOR_ORDER = [
  "payment_request",
  "threat_language",
  "urgency",
  "emotional_pressure",
] as const;

export function RiskView() {
  const [data, setData] = useState<RiskOverview | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [showAbsent, setShowAbsent] = useState(true);

  const load = useCallback(() => {
    setError(null);
    getRisk()
      .then(setData)
      .catch(setError);
  }, []);

  useEffect(load, [load]);

  if (error) {
    return (
      <>
        <PageHeader title="Risk analysis" />
        <ErrorBlock error={error} onRetry={load} />
      </>
    );
  }

  if (!data) {
    return (
      <>
        <PageHeader title="Risk analysis" />
        <LoadingBlock label="Loading risk assessments" />
      </>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow="Wording indicators"
        title="Risk analysis"
        description="Four independent checks, reported one at a time with the wording that triggered each. Nothing here is a verdict about a sender."
        actions={
          <label className="flex items-center gap-2 text-sm text-ink-800">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border-ink-400"
              checked={showAbsent}
              onChange={(event) => setShowAbsent(event.target.checked)}
            />
            Show indicators that are not present
          </label>
        }
      />

      <div className="mb-5 space-y-3">
        <Callout tone="neutral" title="Method note (from the analysis service)">
          <p>{data.method_note}</p>
        </Callout>
        <Callout tone="neutral" title="Benign reminder (from the analysis service)">
          <p>{data.benign_note}</p>
        </Callout>
      </div>

      {data.records.length === 0 ? (
        <EmptyState title="No record currently carries an indicator">
          <p>
            No wording match was found in the records analyzed. That is a statement about the text
            that was imported, not a clean bill of health for anyone.
          </p>
        </EmptyState>
      ) : (
        <div className="space-y-5">
          {data.records.map((record) => (
            <RiskRecordCard key={record.record_id} record={record} showAbsent={showAbsent} />
          ))}
        </div>
      )}

      <div className="mt-6">
        <Card
          title="Which models produced these indicators"
          subtitle="Availability is stated plainly so you know what the analysis could and could not see."
        >
          <ul>
            <AvailabilityRow
              name="Rules baseline"
              available={data.models.rules.available}
              detail={data.models.rules.description}
            />
            <AvailabilityRow
              name="TF-IDF + logistic regression"
              available={data.models.tfidf_lr.available}
              detail={data.models.tfidf_lr.description}
            />
            <AvailabilityRow
              name="DistilBERT classifier"
              available={data.models.distilbert.available}
              detail={data.models.distilbert.description}
            />
          </ul>
          {data.models.tfidf_lr.thresholds ? (
            <Disclosure summary="TF-IDF thresholds per indicator">
              <ul className="space-y-0.5">
                {Object.entries(data.models.tfidf_lr.thresholds).map(([label, value]) => (
                  <li key={label}>
                    {label}: {value.toFixed(2)}
                  </li>
                ))}
              </ul>
            </Disclosure>
          ) : null}
          {data.models.distilbert.thresholds ? (
            <Disclosure summary="DistilBERT thresholds per indicator">
              <ul className="space-y-0.5">
                {Object.entries(data.models.distilbert.thresholds).map(([label, value]) => (
                  <li key={label}>
                    {label}: {value.toFixed(2)}
                  </li>
                ))}
              </ul>
              {data.models.distilbert.calibration ? (
                <p className="mt-2">Calibration: {data.models.distilbert.calibration}</p>
              ) : null}
            </Disclosure>
          ) : null}
          {data.models.notes.length > 0 ? (
            <div className="mt-3 space-y-1">
              {data.models.notes.map((note) => (
                <p key={note} className="text-xs text-ink-600">
                  {note}
                </p>
              ))}
            </div>
          ) : null}
        </Card>
      </div>
    </>
  );
}

function RiskRecordCard({
  record,
  showAbsent,
}: {
  record: RiskRecord;
  showAbsent: boolean;
}) {
  const byLabel = new Map(record.indicators.map((indicator) => [indicator.label, indicator]));
  const flagged = record.indicators.filter((indicator) => indicator.present);
  const ordered = INDICATOR_ORDER.map((label) => byLabel.get(label)).filter(
    (indicator): indicator is NonNullable<typeof indicator> => Boolean(indicator),
  );
  const absent = ordered.filter((indicator) => !indicator.present);

  return (
    <Card
      title={
        <span className="flex flex-wrap items-center gap-2">
          <a
            href={`#${recordLink(record.record_id)}`}
            className="text-sm font-semibold text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
          >
            {record.title}
          </a>
          <SourceBadge sourceType={record.source_type} />
          <SeverityBadge severity={record.severity} />
        </span>
      }
      subtitle={`${formatDateTime(record.occurred_at)} · ${flagged.length} of 4 indicator(s) present`}
      actions={<RiskChip labels={flagged.map((i) => i.label)} severity={record.severity} />}
    >
      <div
        role="alert"
        className={cx(
          "mb-4 rounded-md border px-3 py-3 text-sm",
          record.severity === "warning"
            ? "border-risk-300 bg-risk-50 text-risk-950"
            : "border-risk-200 bg-risk-50/60 text-risk-950",
        )}
      >
        <p className="font-semibold">{record.headline}</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[40rem] border-collapse text-left">
          <caption className="sr-only">
            Indicator breakdown for {record.title}. Each row states presence, score, threshold,
            contributing methods and the explanation supplied by the analysis service.
          </caption>
          <thead>
            <tr className="border-b border-ink-200 text-xs uppercase tracking-wide text-ink-500">
              <th scope="col" className="py-2 pr-3">
                Indicator
              </th>
              <th scope="col" className="py-2 pr-3">
                State
              </th>
              <th scope="col" className="py-2 pr-3">
                Score / threshold
              </th>
              <th scope="col" className="py-2 pr-3">
                Methods that contributed
              </th>
              <th scope="col" className="py-2">
                Explanation
              </th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((indicator) => (
              <tr key={indicator.label} className="border-b border-ink-100 align-top last:border-0">
                <th scope="row" className="py-3 pr-3 text-sm font-medium text-ink-900">
                  {indicator.display}
                </th>
                <td className="py-3 pr-3">
                  <span
                    className={cx(
                      "dm-chip",
                      indicator.present
                        ? "border-risk-400 bg-risk-100 text-risk-900"
                        : "border-ink-300 bg-white text-ink-700",
                    )}
                  >
                    {indicator.present ? "present" : "not present"}
                  </span>
                </td>
                <td className="py-3 pr-3 text-xs tabular-nums text-ink-800">
                  {formatScore(indicator.score)} / {formatScore(indicator.threshold)}
                  {indicator.calibrated ? (
                    <span className="mt-0.5 block text-ink-600">calibrated model output</span>
                  ) : null}
                </td>
                <td className="py-3 pr-3 text-xs text-ink-800">
                  {Object.keys(indicator.methods).length === 0 ? (
                    <span className="text-ink-600">none</span>
                  ) : (
                    <ul className="space-y-0.5">
                      {Object.entries(indicator.methods).map(([name, value]) => (
                        <li key={name}>
                          {name}: {formatScore(value)}
                        </li>
                      ))}
                    </ul>
                  )}
                </td>
                <td className="py-3 text-xs text-ink-700">
                  {indicator.explanation}
                  {indicator.evidence.length > 0 ? (
                    <ul className="mt-2 space-y-1.5">
                      {indicator.evidence.map((item, index) => (
                        <li key={`${item.start_char}-${item.end_char}-${index}`}>
                          <EvidenceQuote
                            evidence={item}
                            recordId={record.record_id}
                            indicator={indicator.display}
                          />
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-1 text-ink-600">
                      No cited span for this indicator. There is no wording to inspect.
                    </p>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showAbsent && absent.length > 0 ? (
        <Disclosure summary={`Indicators checked and not present (${absent.length})`}>
          <ul className="space-y-1">
            {absent.map((indicator) => (
              <li key={indicator.label} className="text-xs">
                <strong>{indicator.display}</strong>: {indicator.explanation}
              </li>
            ))}
          </ul>
          <p className="mt-2 text-xs">
            “Not present” means no signal in the analyzed text crossed the threshold for that
            indicator. It is not a finding that the underlying situation is safe.
          </p>
        </Disclosure>
      ) : null}

      <div className="mt-4 rounded-md border border-ink-300 bg-ink-50 px-4 py-3">
        <p className="dm-section-title">Overall estimate</p>
        <p className="mt-1 flex flex-wrap items-baseline gap-2">
          <span className="text-2xl font-semibold tabular-nums text-ink-900">
            {record.overall_estimate.value.toFixed(2)}
          </span>
          <span className="dm-chip border-ink-400 bg-white text-ink-900">estimate</span>
          <span className="text-xs text-ink-700">
            {record.overall_estimate.flagged_count} of 4 indicator(s) present
          </span>
        </p>
        <p className="mt-2 text-xs text-ink-700">
          <strong>Formula:</strong> {record.overall_estimate.formula}
        </p>
        <ul className="mt-2 space-y-1 text-xs text-ink-800">
          {record.overall_estimate.contributions.map((item) => (
            <li key={item.label} className="flex flex-wrap gap-x-2">
              <span className="font-medium">{item.display}</span>
              <span className="tabular-nums">
                weight {item.weight.toFixed(2)} × score {item.score.toFixed(2)} ={" "}
                {item.contribution.toFixed(4)}
                {item.present ? "" : " (not present)"}
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-xs text-ink-700">
          <strong>Caveat:</strong> {record.overall_estimate.caveat}
        </p>
        {record.overall_estimate.methods.length > 0 ? (
          <p className="mt-1 text-xs text-ink-600">
            Methods: {record.overall_estimate.methods.join(", ")}
          </p>
        ) : null}
      </div>
    </Card>
  );
}

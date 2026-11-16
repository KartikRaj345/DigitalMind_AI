import { useCallback, useEffect, useMemo, useState } from "react";

import { deleteRecord, getRecord, type Evidence, type Indicator, type RecordDetail } from "../lib/api";
import {
  formatDateTime,
  formatScore,
  riskLabelDisplay,
  truncate,
} from "../lib/format";
import { recordLink, useNavigate } from "../lib/router";
import { HighlightedBody, type BodyMark } from "../components/HighlightedBody";
import {
  Callout,
  Card,
  Disclosure,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  NeutralChip,
  PageHeader,
  RiskChip,
  SourceBadge,
  cx,
} from "../components/ui";

/**
 * `GET /records/{id}` does not return a severity, but the UI must not invent one either.
 * This mirrors the backend's own rule (a money demand paired with a stated consequence or a
 * deadline escalates; urgency on its own does not) so the chip never overstates the level.
 */
function deriveSeverity(labels: string[]): "none" | "notice" | "warning" {
  if (labels.length === 0) return "none";
  const set = new Set(labels);
  if (set.has("payment_request") && (set.has("threat_language") || set.has("urgency"))) {
    return "warning";
  }
  return "notice";
}

function parseSpan(param: string | null): { start: number; end: number } | null {
  if (!param) return null;
  const match = /^(\d+)-(\d+)$/.exec(param);
  if (!match) return null;
  const start = Number.parseInt(match[1], 10);
  const end = Number.parseInt(match[2], 10);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  return { start, end };
}

export function RecordDetailView({
  recordId,
  spanParam,
}: {
  recordId: number;
  spanParam: string | null;
}) {
  const [record, setRecord] = useState<RecordDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [showEntities, setShowEntities] = useState(true);
  const [showRisk, setShowRisk] = useState(true);
  const [showEvents, setShowEvents] = useState(true);
  const [activeMarkId, setActiveMarkId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteMessage, setDeleteMessage] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const navigate = useNavigate();

  // Memoised so the citation-focus effect below does not re-run on every render.
  const focusSpan = useMemo(() => parseSpan(spanParam), [spanParam]);

  const load = useCallback(() => {
    setError(null);
    getRecord(recordId)
      .then(setRecord)
      .catch(setError);
  }, [recordId]);

  useEffect(load, [load]);

  // When arriving from a citation, focus the mark that covers exactly that character range.
  // The ids are built the same way as in `marks` below, so a citation always selects the box
  // the user clicked through to.
  useEffect(() => {
    if (!record || !focusSpan) return;
    const matches: string[] = [];

    record.indicators.forEach((indicator) => {
      if (!indicator.present) return;
      indicator.evidence.forEach((item, index) => {
        if (item.start_char === focusSpan.start && item.end_char === focusSpan.end) {
          matches.push(`risk-${index}-${item.start_char}-${item.end_char}`);
        }
      });
    });
    record.spans.forEach((span, index) => {
      if (span.start_char === focusSpan.start && span.end_char === focusSpan.end) {
        matches.push(`entity-${index}-${span.start_char}`);
      }
    });
    record.events.forEach((event, index) => {
      if (event.start_char === focusSpan.start && event.end_char === focusSpan.end) {
        matches.push(`event-${index}-${event.start_char}`);
      }
    });

    if (matches.length === 0) {
      // The characters exist in the body even when no extraction covers them. Say so rather
      // than pretending a highlight was found.
      setActiveMarkId(`manual-${focusSpan.start}`);
      return;
    }
    setShowRisk(true);
    setShowEntities(true);
    setShowEvents(true);
    setActiveMarkId(matches[0]);
  }, [record, focusSpan]);

  const marks = useMemo<BodyMark[]>(() => {
    if (!record) return [];
    const out: BodyMark[] = [];
    if (showEntities) {
      record.spans.forEach((span, index) => {
        out.push({
          id: `entity-${index}-${span.start_char}`,
          kind: "entity",
          start: span.start_char,
          end: span.end_char,
          label: `Entity: ${span.entity_name ?? span.text}`,
          description: `${span.label} · ${span.extractor} · confidence ${span.confidence.toFixed(2)}`,
          entityId: span.entity_id,
        });
      });
    }
    if (showRisk) {
      record.indicators.forEach((indicator) => {
        if (!indicator.present) return;
        indicator.evidence.forEach((item, index) => {
          if (item.start_char === null || item.end_char === null) return;
          out.push({
            id: `risk-${index}-${item.start_char}-${item.end_char}`,
            kind: "risk",
            start: item.start_char,
            end: item.end_char,
            label: `${indicator.display} evidence`,
            description: item.note ?? "Indicator evidence",
          });
        });
      });
    }
    if (showEvents) {
      record.events.forEach((event, index) => {
        if (event.start_char === null || event.end_char === null) return;
        out.push({
          id: `event-${index}-${event.start_char}`,
          kind: "event",
          start: event.start_char,
          end: event.end_char,
          label: `Event: ${event.title}`,
          description: `${event.certainty} · ${event.precision}`,
          eventId: event.id,
        });
      });
    }
    return out;
  }, [record, showEntities, showRisk, showEvents]);

  const flagged = record?.indicators.filter((indicator) => indicator.present) ?? [];

  const onDelete = async () => {
    if (!record) return;
    setDeleting(true);
    setError(null);
    try {
      const result = await deleteRecord(record.id);
      setDeleteMessage(result.message);
      setConfirmDelete(false);
      navigate("/records");
    } catch (deleteError) {
      setError(deleteError);
      setConfirmDelete(false);
    } finally {
      setDeleting(false);
    }
  };

  if (error && !record) {
    return (
      <>
        <PageHeader eyebrow="Record" title="Record unavailable" />
        <ErrorBlock error={error} onRetry={load} />
      </>
    );
  }

  if (!record) {
    return (
      <>
        <PageHeader eyebrow="Record" title="Loading record" />
        <LoadingBlock label={`Loading record ${recordId}`} />
      </>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow={`Record ${record.id} · ${record.source_type}`}
        title={record.title}
        description={
          <>
            {formatDateTime(record.occurred_at)} · imported {formatDateTime(record.imported_at)} ·{" "}
            {record.body.length} characters
            {record.participants.length > 0 ? ` · ${record.participants.join(", ")}` : ""}
          </>
        }
        actions={
          <>
            <a href="#/records" className="dm-btn">
              Back to records
            </a>
            <button
              type="button"
              className="dm-btn dm-btn-risk"
              onClick={() => setConfirmDelete((value) => !value)}
              aria-expanded={confirmDelete}
            >
              Delete this record
            </button>
          </>
        }
      />

      {confirmDelete ? (
        <div
          role="alert"
          className="mb-5 rounded-md border border-risk-300 bg-risk-50 px-4 py-4 text-sm text-risk-950"
        >
          <p className="font-semibold">
            Delete this record and everything derived from it?
          </p>
          <p className="mt-1">
            Its extracted entities, graph edges, events, risk assessments, insights and index
            entries are removed too. This cannot be undone.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" className="dm-btn dm-btn-risk" onClick={onDelete} disabled={deleting}>
              {deleting ? "Deleting…" : "Yes, delete it and everything derived from it"}
            </button>
            <button type="button" className="dm-btn" onClick={() => setConfirmDelete(false)}>
              Keep the record
            </button>
          </div>
        </div>
      ) : null}

      {deleteMessage ? (
        <p role="status" className="mb-5 rounded-md border border-teal-300 bg-teal-50 px-4 py-3 text-sm text-teal-950">
          {deleteMessage}
        </p>
      ) : null}

      {focusSpan ? (
        <p
          role="status"
          className="mb-5 rounded-md border border-teal-300 bg-teal-50 px-4 py-3 text-sm text-teal-950"
        >
          Opened from a citation. Characters {focusSpan.start}–{focusSpan.end} of this record are
          the cited text: “{truncate(record.body.slice(focusSpan.start, focusSpan.end), 240)}”
        </p>
      ) : null}

      <div className="grid gap-5 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="space-y-5">
          <Card
            title="Record body"
            subtitle="Highlights are positioned over the raw text, so every citation refers to the same characters you see."
            actions={
              <>
                <Toggle
                  checked={showEntities}
                  onChange={setShowEntities}
                  label="Show entities"
                  tone="entity"
                />
                <Toggle
                  checked={showRisk}
                  onChange={setShowRisk}
                  label="Show risk evidence"
                  tone="risk"
                />
                <Toggle checked={showEvents} onChange={setShowEvents} label="Show events" tone="event" />
              </>
            }
          >
            <HighlightedBody
              body={record.body}
              marks={marks}
              activeMarkId={activeMarkId}
              onSelect={(mark) => setActiveMarkId((current) => (current === mark.id ? null : mark.id))}
            />
            <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-ink-100 pt-3 text-xs text-ink-700">
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="inline-block h-3 w-4 rounded-sm bg-teal-300 ring-1 ring-teal-600" />
                Entity mention ({record.spans.length})
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="inline-block h-3 w-4 rounded-sm bg-risk-300 ring-1 ring-risk-700" />
                Risk evidence ({flagged.reduce((sum, i) => sum + i.evidence.length, 0)})
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="inline-block h-3 w-4 rounded-sm bg-blue-200 ring-1 ring-blue-700" />
                Event wording ({record.events.filter((e) => e.start_char !== null).length})
              </span>
            </div>
            {record.backreferences.length > 0 ? (
              <div className="mt-3 border-t border-ink-100 pt-3">
                <p className="dm-section-title">Phrases that refer back to earlier discussion</p>
                <ul className="mt-2 flex flex-wrap gap-2">
                  {record.backreferences.map((item) => (
                    <li key={`${item.start_char}-${item.end_char}`}>
                      <a
                        href={`#${recordLink(record.id, item.start_char, item.end_char)}`}
                        className="dm-chip border-ink-300 bg-white text-ink-800 hover:bg-ink-50"
                      >
                        “{item.text}” at {item.start_char}–{item.end_char}
                      </a>
                    </li>
                  ))}
                </ul>
                <p className="mt-2 text-xs text-ink-600">
                  A back-reference asserts that something was discussed before. If no matching
                  record is found, that is a gap in what has been imported, not a reason to conclude
                  nothing happened.
                </p>
              </div>
            ) : null}
          </Card>

          <Card
            title="Indicators for this record"
            subtitle="Each indicator is scored independently. None of them is a verdict about the sender."
            actions={
              <RiskChip
                labels={flagged.map((i) => i.label)}
                severity={deriveSeverity(flagged.map((i) => i.label))}
              />
            }
          >
            <div className="space-y-4">
              {record.indicators.map((indicator) => (
                <IndicatorRow
                  key={indicator.label}
                  indicator={indicator}
                  recordId={record.id}
                />
              ))}
            </div>
          </Card>

          <OverallEstimateCard record={record} />
        </div>

        <div className="space-y-5">
          <Card
            title="Entities"
            subtitle={`${record.entities.length} entity(ies) resolved from this record`}
          >
            {record.entities.length === 0 ? (
              <EmptyState title="No entity was extracted from this record">
                Nothing in the analyzed text matched a person, organisation or other entity pattern.
                That reflects the extractors that are available, not a statement about the record.
              </EmptyState>
            ) : (
              <ul className="space-y-4">
                {record.entities.map((entity) => (
                  <li key={entity.id} className="border-b border-ink-100 pb-3 last:border-0 last:pb-0">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <a
                        href={`#/graph?focus=entity:${entity.id}`}
                        className="text-sm font-semibold text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
                      >
                        {entity.name}
                      </a>
                      <NeutralChip>{entity.type}</NeutralChip>
                    </div>
                    <p className="mt-1 text-xs text-ink-600">
                      {entity.mention_count} mention(s) · resolution confidence{" "}
                      {entity.resolution_confidence.toFixed(2)}
                    </p>
                    {entity.surface_forms.length > 0 ? (
                      <div className="mt-2 rounded-md border border-ink-200 bg-ink-50 px-3 py-2">
                        <p className="dm-section-title">Merge provenance</p>
                        <p className="mt-1 text-xs text-ink-700">
                          These surface forms were resolved into this entity. Resolution merges similar
                          names and can be wrong.
                        </p>
                        <ul className="mt-2 flex flex-wrap gap-1.5">
                          {entity.surface_forms.map((form) => (
                            <li key={form}>
                              <span className="dm-chip border-ink-300 bg-white text-ink-800">{form}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : (
                      <p className="mt-2 text-xs text-ink-600">
                        Only one surface form was recorded for this entity, so no merge was needed.
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card
            title="Events"
            subtitle={`${record.events.length} event(s) extracted, each with its own certainty`}
          >
            {record.events.length === 0 ? (
              <EmptyState title="No event was extracted from this record">
                No dated or relative event wording matched in the analyzed text. Other extractors or
                more records may change this.
              </EmptyState>
            ) : (
              <ul className="space-y-4">
                {record.events.map((event, index) => (
                  <li key={event.id ?? index} className="border-b border-ink-100 pb-3 last:border-0 last:pb-0">
                    <p className="text-sm font-semibold text-ink-900">{event.title}</p>
                    {event.description ? (
                      <p className="mt-1 text-sm text-ink-700">{event.description}</p>
                    ) : null}
                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      <span
                        className={cx(
                          "dm-chip",
                          event.certainty === "recorded"
                            ? "border-teal-300 bg-teal-50 text-teal-900"
                            : event.certainty === "explicit"
                              ? "border-blue-300 bg-blue-50 text-blue-900"
                              : "border-ink-400 bg-ink-100 text-ink-900",
                        )}
                      >
                        certainty: {event.certainty}
                      </span>
                      <NeutralChip>precision: {event.precision}</NeutralChip>
                      <NeutralChip>{event.kind}</NeutralChip>
                      <NeutralChip>confidence {event.confidence.toFixed(2)}</NeutralChip>
                      <NeutralChip>{event.extractor}</NeutralChip>
                    </div>
                    {event.raw_expression ? (
                      <p className="mt-2 text-xs text-ink-700">
                        Raw wording: “{event.raw_expression}”
                      </p>
                    ) : null}
                    <p className="mt-1 text-xs text-ink-600">
                      {formatDateTime(event.start_at)}
                      {event.end_at ? ` → ${formatDateTime(event.end_at)}` : ""}
                      {event.start_char !== null && event.end_char !== null
                        ? ` · characters ${event.start_char}–${event.end_char}`
                        : ""}
                    </p>
                    {event.start_char !== null && event.end_char !== null ? (
                      <a
                        href={`#${recordLink(record.id, event.start_char, event.end_char)}`}
                        className="mt-2 inline-block text-xs font-medium text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
                      >
                        Show this wording in the body
                      </a>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="Record metadata" subtitle="Everything stored alongside the text">
            <dl>
              <Row term="Source type">
                <SourceBadge sourceType={record.source_type} />
              </Row>
              <Row term="Origin">{record.origin}</Row>
              <Row term="External id">{record.external_id ?? "none recorded"}</Row>
              <Row term="Occurred at">
                {record.occurred_at
                  ? formatDateTime(record.occurred_at)
                  : "no source timestamp — only date content inside the text is available"}
              </Row>
              <Row term="Imported at">{formatDateTime(record.imported_at)}</Row>
              <Row term="Participants">
                {record.participants.length > 0 ? record.participants.join(", ") : "none recorded"}
              </Row>
              <Row term="Retrieval chunks">
                {record.chunks.length} chunk(s):{" "}
                {record.chunks.map((chunk) => `${chunk.ordinal}:${chunk.start_char}–${chunk.end_char}`).join(", ") || "none"}
              </Row>
              {Object.keys(record.extra).length > 0 ? (
                <Row term="Extra fields">
                  <ul className="list-inside list-disc">
                    {Object.entries(record.extra).map(([key, value]) => (
                      <li key={key}>
                        <span className="font-mono text-xs">{key}</span>: {String(value)}
                      </li>
                    ))}
                  </ul>
                </Row>
              ) : null}
            </dl>
          </Card>
        </div>
      </div>
    </>
  );
}

function Row({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 border-b border-ink-100 py-2 last:border-0">
      <dt className="w-full text-xs font-semibold uppercase tracking-wide text-ink-500 sm:w-40">{term}</dt>
      <dd className="min-w-0 flex-1 text-sm text-ink-800">{children}</dd>
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
  tone,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  tone: "entity" | "risk" | "event";
}) {
  const dot =
    tone === "entity" ? "bg-teal-600" : tone === "risk" ? "bg-risk-600" : "bg-blue-600";
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={cx(
        "dm-btn",
        checked ? "border-ink-400 bg-ink-100 text-ink-900" : "text-ink-600",
      )}
    >
      <span
        aria-hidden="true"
        className={cx("inline-block h-2.5 w-2.5 rounded-full", checked ? dot : "bg-ink-300")}
      />
      {label}
      <span className="sr-only">{checked ? " (on)" : " (off)"}</span>
    </button>
  );
}

function IndicatorRow({ indicator, recordId }: { indicator: Indicator; recordId: number }) {
  return (
    <article
      className={cx(
        "rounded-md border px-4 py-3",
        indicator.present ? "border-risk-300 bg-risk-50" : "border-ink-200 bg-white",
      )}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-ink-900">{indicator.display}</h3>
        <span
          className={cx(
            "text-xs font-semibold uppercase tracking-wide",
            indicator.present ? "text-risk-800" : "text-ink-500",
          )}
        >
          {indicator.present ? "Present" : "Not present"}
        </span>
      </div>
      <p className="mt-1 text-xs text-ink-700">
        Score {formatScore(indicator.score)} · threshold {formatScore(indicator.threshold)} ·
        methods:{" "}
        {Object.keys(indicator.methods).length > 0
          ? Object.entries(indicator.methods)
              .map(([name, value]) => `${name} ${formatScore(value)}`)
              .join(", ")
          : "none"}{" "}
        {indicator.calibrated ? "· model output is calibrated" : "· rules output is not calibrated"}
      </p>
      <p className="mt-2 text-sm text-ink-800">{indicator.explanation}</p>
      {indicator.evidence.length > 0 ? (
        <ul className="mt-3 space-y-2">
          {indicator.evidence.map((item: Evidence, index: number) => (
            <li key={`${item.start_char}-${item.end_char}-${index}`}>
              <EvidenceQuote evidence={item} recordId={recordId} indicator={indicator.display} />
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-3 text-xs text-ink-600">
          No cited span. Nothing in the analyzed text produced evidence for this indicator.
        </p>
      )}
    </article>
  );
}

export function EvidenceQuote({
  evidence,
  recordId,
  indicator,
}: {
  evidence: Evidence;
  recordId: number;
  indicator?: string;
}) {
  const linkable =
    evidence.start_char !== null && evidence.end_char !== null && evidence.end_char > evidence.start_char;
  const inner = (
    <>
      <span className="block text-sm italic text-ink-900">
        “{evidence.text ?? "no specific span cited"}”
      </span>
      <span className="mt-1 block text-xs text-ink-600">
        {evidence.note ?? "no note supplied"}
        {evidence.polarity ? ` · ${evidence.polarity}` : ""}
        {evidence.weight !== null && evidence.weight !== undefined
          ? ` · weight ${evidence.weight}`
          : ""}
        {linkable ? ` · characters ${evidence.start_char}–${evidence.end_char}` : ""}
      </span>
      <span className="sr-only">
        {linkable ? "Open the record at this passage" : "No character range available for this evidence"}
        {indicator ? ` for ${indicator}` : ""}
      </span>
    </>
  );

  if (!linkable) {
    return (
      <div className="rounded-md border border-ink-300 border-dashed bg-white px-3 py-2">
        {inner}
      </div>
    );
  }
  return (
    <a
      href={`#${recordLink(recordId, evidence.start_char, evidence.end_char)}`}
      className="block rounded-md border border-ink-300 bg-white px-3 py-2 hover:border-teal-600 hover:bg-teal-50"
    >
      {inner}
    </a>
  );
}

function OverallEstimateCard({ record }: { record: RecordDetail }) {
  const estimate = record.overall_estimate;
  return (
    <Card
      title={
        <span className="inline-flex items-center gap-2">
          Overall estimate
          <span className="dm-chip border-ink-400 bg-ink-100 text-ink-900">estimate</span>
        </span>
      }
      subtitle="Shown for sorting only, together with the formula that produced it."
    >
      <div className="rounded-md border border-ink-300 bg-ink-50 px-4 py-4">
        <p className="dm-section-title">Estimate value</p>
        <p className="mt-1 text-3xl font-semibold tabular-nums text-ink-900">
          {estimate.value.toFixed(2)}
        </p>
        <p className="mt-1 text-xs text-ink-700">
          This is an explicitly derived <strong>estimate</strong> over{" "}
          {estimate.contributions.length} indicator(s), with {estimate.flagged_count} flagged. It is
          not a probability of fraud.
        </p>

        <div className="mt-4">
          <p className="dm-section-title">Formula</p>
          <p className="mt-1 rounded-md border border-ink-200 bg-white px-3 py-2 font-mono text-xs text-ink-800">
            {estimate.formula}
          </p>
        </div>

        <div className="mt-4">
          <p className="dm-section-title">Contribution per indicator</p>
          <ul className="mt-1 space-y-1.5">
            {estimate.contributions.map((item) => (
              <li key={item.label} className="rounded-md border border-ink-200 bg-white px-3 py-2">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="text-sm font-medium text-ink-900">
                    {riskLabelDisplay(item.label)}
                    <span
                      className={cx(
                        "ml-2 text-xs",
                        item.present ? "text-risk-800" : "text-ink-500",
                      )}
                    >
                      {item.present ? "present" : "not present"}
                    </span>
                  </span>
                  <span className="text-xs tabular-nums text-ink-700">
                    weight {item.weight.toFixed(2)} × score {item.score.toFixed(2)} = contribution{" "}
                    {item.contribution.toFixed(4)}
                  </span>
                </div>
                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-ink-200">
                  <div
                    className={cx("h-full", item.present ? "bg-risk-600" : "bg-ink-400")}
                    style={{ width: `${Math.min(100, Math.max(2, item.contribution * 300))}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-4">
          <p className="dm-section-title">Caveat</p>
          <p className="mt-1 text-sm text-ink-800">{estimate.caveat}</p>
        </div>

        {estimate.methods.length > 0 ? (
          <Disclosure summary="Methods that contributed to this estimate">
            <p>{estimate.methods.join(", ")}</p>
            <p className="mt-2">
              Judge each indicator on its cited text rather than on this number.
            </p>
          </Disclosure>
        ) : null}
      </div>
    </Card>
  );
}

export { Callout };

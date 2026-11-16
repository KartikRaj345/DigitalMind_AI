import { useCallback, useEffect, useMemo, useState } from "react";

import { getTimeline, type TimelineItem, type TimelineLane, type TimelineResult } from "../lib/api";
import { formatDate, formatDateTime, formatRange, riskLabelDisplay } from "../lib/format";
import { recordLink } from "../lib/router";
import {
  Card,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  SourceBadge,
  cx,
} from "../components/ui";

const LANES: { lane: TimelineLane; title: string; bar: string; text: string }[] = [
  {
    lane: "recorded",
    title: "Recorded",
    bar: "bg-teal-600",
    text: "text-teal-900",
  },
  {
    lane: "explicit",
    title: "Explicit dates",
    bar: "bg-blue-600",
    text: "text-blue-900",
  },
  {
    lane: "inferred",
    title: "Inferred ranges",
    bar: "bg-ink-400",
    text: "text-ink-900",
  },
];

interface DayGroup {
  day: string;
  recorded: TimelineItem[];
  explicit: TimelineItem[];
  inferred: TimelineItem[];
}

export function TimelineView() {
  const [data, setData] = useState<TimelineResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [showRiskOnly, setShowRiskOnly] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    getTimeline({ limit: 500 })
      .then(setData)
      .catch(setError);
  }, []);

  useEffect(load, [load]);

  const groups = useMemo<DayGroup[]>(() => {
    if (!data) return [];
    const visible = showRiskOnly ? data.items.filter((item) => item.has_risk) : data.items;
    const byDay = new Map<string, DayGroup>();
    for (const item of visible) {
      const day = dayKey(item);
      let group = byDay.get(day);
      if (!group) {
        group = { day, recorded: [], explicit: [], inferred: [] };
        byDay.set(day, group);
      }
      group[item.lane].push(item);
    }
    return Array.from(byDay.values())
      .sort((a, b) => a.day.localeCompare(b.day))
      .map((group) => ({
        ...group,
        recorded: group.recorded.sort(bySortKey),
        explicit: group.explicit.sort(bySortKey),
        inferred: group.inferred.sort(bySortKey),
      }));
  }, [data, showRiskOnly]);

  if (error) {
    return (
      <>
        <PageHeader title="Timeline" />
        <ErrorBlock error={error} onRetry={load} />
      </>
    );
  }

  if (!data) {
    return (
      <>
        <PageHeader title="Timeline" />
        <LoadingBlock label="Building the timeline" />
      </>
    );
  }

  const { summary } = data;

  return (
    <>
      <PageHeader
        eyebrow="Sequence"
        title="Timeline"
        description="Three separate lanes, because they mean different things. Nothing is moved between lanes to make a story line up."
        actions={
          <>
            <label className="flex items-center gap-2 text-sm text-ink-800">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-ink-400"
                checked={showRiskOnly}
                onChange={(event) => setShowRiskOnly(event.target.checked)}
              />
              Only items whose record carries an indicator
            </label>
            <button type="button" className="dm-btn" onClick={load}>
              Refresh
            </button>
          </>
        }
      />

      <Card
        className="mb-5"
        title="Legend"
        subtitle="Inferred items are ranges, not points in time."
      >
        <div className="grid gap-3 sm:grid-cols-3">
          {LANES.map((entry) => (
            <div key={entry.lane} className="rounded-md border border-ink-200 bg-white px-3 py-3">
              <div className="flex items-center gap-2">
                <span
                  aria-hidden="true"
                  className={cx(
                    "inline-block h-3 w-6 rounded-sm",
                    entry.lane === "inferred"
                      ? "bg-gradient-to-r from-ink-200 via-ink-400 to-ink-200"
                      : entry.bar,
                  )}
                />
                <p className={cx("text-sm font-semibold", entry.text)}>{entry.title}</p>
              </div>
              <p className="mt-1 text-xs text-ink-700">{summary.explanations[entry.lane]}</p>
              <p className="mt-1 text-xs font-medium tabular-nums text-ink-900">
                {summary.counts[entry.lane] ?? 0} item(s)
              </p>
            </div>
          ))}
        </div>
        <p className="mt-3 rounded-md border border-ink-300 bg-ink-50 px-3 py-2 text-xs text-ink-800">
          <strong>Inferred items are ranges, not points in time.</strong> {summary.note} An inferred
          bar shows the window the relative wording could fall in, resolved against the record's own
          timestamp.
        </p>
      </Card>

      {data.items.length === 0 ? (
        <EmptyState title="No timeline items">
          The timeline is built from records with a source timestamp and from dated wording inside the
          text. Nothing to place means nothing has been imported yet.
        </EmptyState>
      ) : groups.length === 0 ? (
        <EmptyState title="No items match the current filter">
          Uncheck the indicator filter to see the rest of the timeline.
        </EmptyState>
      ) : (
        <ol className="space-y-6">
          {groups.map((group) => (
            <li key={group.day}>
              <h2 className="sticky top-0 z-10 mb-2 inline-block rounded-md border border-ink-300 bg-white px-3 py-1 text-sm font-semibold text-ink-900">
                {formatDate(group.day)}
              </h2>
              <div className="space-y-3">
                {LANES.map((entry) => {
                  const items = group[entry.lane];
                  if (items.length === 0) return null;
                  return (
                    <div key={entry.lane} className="grid gap-2 sm:grid-cols-[9rem_minmax(0,1fr)]">
                      <div className="sm:pt-1">
                        <p className={cx("text-xs font-semibold uppercase tracking-wide", entry.text)}>
                          {entry.title}
                        </p>
                      </div>
                      <ul className="space-y-2">
                        {items.map((item) => (
                          <TimelineRow
                            key={`${item.record_id}-${item.lane}-${item.kind}-${item.start_at}-${item.start_char}`}
                            item={item}
                            lane={entry.lane}
                            barClass={entry.bar}
                            open={openId === itemKey(item)}
                            onToggle={() =>
                              setOpenId((current) => (current === itemKey(item) ? null : itemKey(item)))
                            }
                          />
                        ))}
                      </ul>
                    </div>
                  );
                })}
              </div>
            </li>
          ))}
        </ol>
      )}
    </>
  );
}

function itemKey(item: TimelineItem): string {
  return `${item.record_id}|${item.lane}|${item.kind}|${item.sort_key}|${item.start_char}`;
}

function bySortKey(a: TimelineItem, b: TimelineItem): number {
  return a.sort_key.localeCompare(b.sort_key);
}

function dayKey(item: TimelineItem): string {
  const value = item.start_at ?? item.end_at ?? item.sort_key;
  return value.slice(0, 10);
}

function TimelineRow({
  item,
  lane,
  barClass,
  open,
  onToggle,
}: {
  item: TimelineItem;
  lane: TimelineLane;
  barClass: string;
  open: boolean;
  onToggle: () => void;
}) {
  const isInferred = lane === "inferred";
  const linkable = item.start_char !== null && item.end_char !== null;

  return (
    <li
      className={cx(
        "rounded-md border bg-white",
        item.has_risk ? "border-risk-300" : "border-ink-200",
      )}
    >
      <div className="flex flex-wrap items-start gap-3 px-3 py-2">
        <span aria-hidden="true" className="mt-1.5 flex h-4 w-16 shrink-0 items-center">
          {isInferred ? (
            <span className="h-2 w-full rounded-full bg-gradient-to-r from-ink-200 via-ink-500 to-ink-200" />
          ) : (
            <span className={cx("h-2.5 w-3 rounded-full", barClass)} />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <a
              href={`#${recordLink(item.record_id, item.start_char, item.end_char)}`}
              className={cx(
                "text-sm font-medium underline underline-offset-2 hover:bg-ink-50",
                item.has_risk ? "text-risk-900 decoration-risk-400" : "text-teal-800 decoration-teal-400",
              )}
            >
              {item.title}
            </a>
            <SourceBadge sourceType={item.source_type} />
            {item.has_risk ? (
              <span className="dm-chip border-risk-300 bg-risk-50 text-risk-900">
                Potential manipulation risk detected ({item.risk_labels.map(riskLabelDisplay).join(", ")})
              </span>
            ) : null}
          </div>
          <p className="mt-1 text-xs text-ink-700">
            {isInferred ? (
              <>
                Range {formatDateTime(item.start_at)} → {formatDateTime(item.end_at)} · raw wording “
                {item.raw_expression}”
              </>
            ) : (
              <>{formatRange(item.start_at, item.end_at)}</>
            )}
          </p>
          <p className="mt-1 text-xs text-ink-500">
            certainty: {item.certainty} · precision: {item.precision} · kind: {item.kind} ·
            confidence {item.confidence.toFixed(2)}
          </p>
        </div>
        <button
          type="button"
          className="dm-btn"
          aria-expanded={open}
          onClick={onToggle}
        >
          {open ? "Hide explanation" : "Why this lane?"}
        </button>
      </div>
      {open ? (
        <div className="border-t border-ink-100 px-3 py-2 text-xs text-ink-700">
          <p>{item.explanation}</p>
          {item.raw_expression ? (
            <p className="mt-1">
              The text says “{item.raw_expression}”. Nothing in the record names a specific moment
              for this, which is why it is {isInferred ? "a range" : "placed in this lane"} rather
              than a point in time.
            </p>
          ) : null}
          {linkable ? (
            <a
              href={`#${recordLink(item.record_id, item.start_char, item.end_char)}`}
              className="mt-1 inline-block font-medium text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
            >
              Show the wording in the record (characters {item.start_char}–{item.end_char})
            </a>
          ) : (
            <p className="mt-1 text-ink-600">
              This item has no character range in the record, so there is nothing to point at.
            </p>
          )}
        </div>
      ) : null}
    </li>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";

import {
  getDashboard,
  listRecords,
  search,
  type RecordList,
  type RecordSummary,
  type RetrievalMode,
  type SearchResult,
} from "../lib/api";
import { formatDateTime, truncate } from "../lib/format";
import { recordLink } from "../lib/router";
import {
  Card,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  RiskChip,
  SourceBadge,
  cx,
} from "../components/ui";

const PAGE_SIZE = 20;

function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

export function RecordsView() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [sourceType, setSourceType] = useState("");
  const [sourceTypes, setSourceTypes] = useState<string[]>([]);
  const [onlyRisk, setOnlyRisk] = useState(false);
  const [offset, setOffset] = useState(0);

  const debouncedQuery = useDebounced(query.trim(), 300);

  const [list, setList] = useState<RecordList | null>(null);
  const [listError, setListError] = useState<unknown>(null);
  const [searchState, setSearchState] = useState<{
    loading: boolean;
    error: unknown;
    result: SearchResult | null;
  }>({ loading: false, error: null, result: null });

  const requestId = useRef(0);

  const loadList = useCallback(() => {
    const id = ++requestId.current;
    setListError(null);
    listRecords({
      q: debouncedQuery || undefined,
      source_type: sourceType || undefined,
      risk: onlyRisk ? true : undefined,
      limit: PAGE_SIZE,
      offset,
    })
      .then((value) => {
        if (requestId.current === id) setList(value);
      })
      .catch((error) => {
        if (requestId.current === id) setListError(error);
      });
  }, [debouncedQuery, sourceType, onlyRisk, offset]);

  useEffect(loadList, [loadList]);

  useEffect(() => {
    if (!debouncedQuery) {
      setSearchState({ loading: false, error: null, result: null });
      return;
    }
    let cancelled = false;
    setSearchState((prev) => ({ ...prev, loading: true, error: null }));
    search({ q: debouncedQuery, mode, limit: 8 })
      .then((result) => {
        if (!cancelled) setSearchState({ loading: false, error: null, result });
      })
      .catch((error) => {
        if (!cancelled) setSearchState({ loading: false, error, result: null });
      });
    return () => {
      cancelled = true;
    };
  }, [debouncedQuery, mode]);

  // Any filter change invalidates the current page window.
  const resetOffset = () => setOffset(0);

  // The filter list comes from the dashboard's per-source-type counts, not from
  // whichever page happens to be loaded, so the list does not change as you paginate.
  useEffect(() => {
    let cancelled = false;
    getDashboard()
      .then((dashboard) => {
        if (!cancelled) setSourceTypes(Object.keys(dashboard.records.by_source_type).sort());
      })
      .catch(() => {
        if (!cancelled) setSourceTypes([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const total = list?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE_SIZE, total);

  return (
    <>
      <PageHeader
        eyebrow="Inbox"
        title="Records"
        description="Every imported record, with the wording that triggered any indicator attached to it. Search runs against the local indexes."
      />

      <Card className="mb-5">
        <form
          role="search"
          aria-label="Search records"
          onSubmit={(event) => {
            event.preventDefault();
            resetOffset();
          }}
        >
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="flex-1">
              <label htmlFor="record-search" className="mb-1 block text-xs font-medium text-ink-700">
                Search the records
              </label>
              <input
                id="record-search"
                type="search"
                className="dm-input"
                placeholder="for example: payment, verification, supplier bank details"
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  resetOffset();
                }}
                autoComplete="off"
              />
            </div>
            <div>
              <label htmlFor="search-mode" className="mb-1 block text-xs font-medium text-ink-700">
                Retrieval mode
              </label>
              <select
                id="search-mode"
                className="dm-input sm:w-44"
                value={mode}
                onChange={(event) => setMode(event.target.value as RetrievalMode)}
              >
                <option value="hybrid">Hybrid</option>
                <option value="lexical">Lexical (BM25)</option>
                <option value="semantic">Semantic</option>
              </select>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-end gap-4">
            <div>
              <label htmlFor="source-filter" className="mb-1 block text-xs font-medium text-ink-700">
                Source type
              </label>
              <select
                id="source-filter"
                className="dm-input sm:w-48"
                value={sourceType}
                onChange={(event) => {
                  setSourceType(event.target.value);
                  resetOffset();
                }}
              >
                <option value="">All source types</option>
                {sourceTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2 pb-2">
              <input
                id="only-risk"
                type="checkbox"
                className="h-4 w-4 rounded border-ink-400 text-teal-700"
                checked={onlyRisk}
                onChange={(event) => {
                  setOnlyRisk(event.target.checked);
                  resetOffset();
                }}
              />
              <label htmlFor="only-risk" className="text-sm text-ink-800">
                Only records with indicators
              </label>
            </div>
          </div>
        </form>
      </Card>

      {debouncedQuery ? (
        <Card
          className="mb-5"
          title="Search result"
          subtitle={
            searchState.result
              ? `${searchState.result.hits.length} ranked passage(s) across ${searchState.result.total_records} record(s) · mode: ${searchState.result.mode}`
              : "Ranking passages across the local indexes"
          }
        >
          {searchState.loading ? <LoadingBlock label="Searching" /> : null}
          {searchState.error ? <ErrorBlock error={searchState.error} /> : null}
          {searchState.result && !searchState.loading ? (
            <>
              {searchState.result.degraded.length > 0 ? (
                <p className="mb-3 rounded-md border border-ink-300 bg-ink-50 px-3 py-2 text-xs text-ink-700">
                  Degraded channels: {searchState.result.degraded.join(", ")}. Results below come from
                  the channels that are available.
                </p>
              ) : null}
              <p className="mb-3 text-xs text-ink-600">{searchState.result.explanation}</p>
              {searchState.result.hits.length === 0 ? (
                <EmptyState title="No matching discussion was found in the records analyzed.">
                  <p>
                    {searchState.result.total_records} record(s) were searched. This only means the
                    records currently imported do not contain a matching discussion. It is not
                    evidence that no such discussion took place.
                  </p>
                </EmptyState>
              ) : (
                <ul className="space-y-3">
                  {searchState.result.hits.map((hit) => (
                    <li key={`${hit.record_id}-${hit.chunk_id}`} className="border-b border-ink-100 pb-3 last:border-0 last:pb-0">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <a
                          href={`#${recordLink(hit.record_id, hit.start_char, hit.end_char)}`}
                          className="text-sm font-medium text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
                        >
                          {hit.title ?? `Record ${hit.record_id}`}
                        </a>
                        <span className="text-xs text-ink-600">
                          {hit.source_type} · {formatDateTime(hit.occurred_at)} · score{" "}
                          {hit.score.toFixed(3)} · {hit.agreement.replace(/_/g, " ")}
                        </span>
                      </div>
                      <p className="mt-1 whitespace-pre-wrap rounded-md border border-ink-200 bg-ink-50 px-3 py-2 text-sm text-ink-800">
                        {hit.text}
                      </p>
                      <p className="mt-1 text-xs text-ink-500">
                        Characters {hit.start_char}–{hit.end_char} of this record.
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </>
          ) : null}
        </Card>
      ) : null}

      <Card
        title="All records"
        subtitle={
          listError
            ? "Could not load the list"
            : `${total} record(s) match the current filters · showing ${from}–${to}`
        }
        actions={
          <>
            <button
              type="button"
              className="dm-btn"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              Previous
            </button>
            <button
              type="button"
              className="dm-btn"
              disabled={total === 0 || to >= total}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              Next
            </button>
          </>
        }
      >
        {listError ? <ErrorBlock error={listError} onRetry={loadList} /> : null}
        {!listError && !list ? <LoadingBlock label="Loading records" /> : null}
        {list && list.items.length === 0 ? (
          <EmptyState title="No records match these filters">
            <p>
              Nothing in the currently imported data matches. Adjust the search text or clear the
              source-type filter. This is not evidence that the records you are looking for do not
              exist elsewhere.
            </p>
          </EmptyState>
        ) : null}
        {list && list.items.length > 0 ? (
          <ul className="divide-y divide-ink-100">
            {list.items.map((item) => (
              <RecordRow key={item.id} item={item} />
            ))}
          </ul>
        ) : null}
      </Card>
    </>
  );
}

function RecordRow({ item }: { item: RecordSummary }) {
  const hasRisk = item.risk_labels.length > 0 && item.severity !== "none";
  return (
    <li className="py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <SourceBadge sourceType={item.source_type} />
            <a
              href={`#/record/${item.id}`}
              className={cx(
                "text-sm font-semibold underline underline-offset-2 hover:bg-ink-50",
                hasRisk ? "text-risk-900 decoration-risk-400" : "text-ink-900 decoration-ink-300",
              )}
            >
              {item.title}
            </a>
          </div>
          <p className="mt-1 text-xs text-ink-600">
            {formatDateTime(item.occurred_at)} · {item.char_count} characters · imported as{" "}
            {item.origin}
            {item.participants.length > 0 ? ` · ${item.participants.join(", ")}` : ""}
          </p>
          <p className="mt-2 text-sm text-ink-700">{truncate(item.snippet, 240)}</p>
        </div>
        <div className="shrink-0">
          <RiskChip labels={item.risk_labels} severity={item.severity} />
        </div>
      </div>
    </li>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";

import { ask, type AskOut, type RetrievalMode } from "../lib/api";
import { formatDateTime, formatScore } from "../lib/format";
import { recordLink } from "../lib/router";
import {
  Callout,
  Card,
  Disclosure,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  SourceBadge,
  cx,
} from "../components/ui";

/**
 * The exact sentence the analysis service produces when it finds no match. It is
 * reproduced verbatim and never reworded: the point is that a gap in the imported
 * records is not a finding about the world.
 */
export const NO_MATCH_SENTENCE = "No matching discussion was found in the records analyzed.";

const EXAMPLES = [
  "What records mention Northwind Bank?",
  "What happened before this payment request?",
  "Which records mention supplier bank details?",
  "Who was involved in the account verification review?",
  "What records contain risk indicators?",
  "What did we discuss about the Q2 supplier reconciliation?",
];

export function AskView() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [answer, setAnswer] = useState<AskOut | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const resultRef = useRef<HTMLDivElement | null>(null);

  const submit = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) {
        setAnswer(null);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const result = await ask({ question: trimmed, mode, top_k: 8 });
        setAnswer(result);
      } catch (err) {
        setError(err);
        setAnswer(null);
      } finally {
        setLoading(false);
        resultRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      }
    },
    [mode],
  );

  useEffect(() => {
    resultRef.current?.scrollIntoView({ block: "start" });
  }, [answer]);

  return (
    <>
      <PageHeader
        eyebrow="Cited answers"
        title="Ask"
        description="Answers are assembled from retrieved text, never generated. Every claim is shown with the record and the character range it came from."
      />

      <Card className="mb-5">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit(question);
          }}
        >
          <label htmlFor="ask-question" className="mb-1 block text-sm font-medium text-ink-800">
            Your question
          </label>
          <div className="flex flex-col gap-3 sm:flex-row">
            <input
              id="ask-question"
              className="dm-input"
              type="text"
              maxLength={500}
              placeholder="for example: What records mention Northwind Bank?"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              autoComplete="off"
            />
            <div className="sm:w-44">
              <label htmlFor="ask-mode" className="sr-only">
                Retrieval mode
              </label>
              <select
                id="ask-mode"
                className="dm-input"
                value={mode}
                onChange={(event) => setMode(event.target.value as RetrievalMode)}
              >
                <option value="hybrid">Hybrid search</option>
                <option value="lexical">Lexical (BM25)</option>
                <option value="semantic">Semantic</option>
              </select>
            </div>
            <button type="submit" className="dm-btn dm-btn-primary sm:w-32" disabled={loading}>
              {loading ? "Searching…" : "Ask"}
            </button>
          </div>
        </form>

        <div className="mt-4 border-t border-ink-100 pt-3">
          <p className="dm-section-title">Example questions</p>
          <ul className="mt-2 flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <li key={example}>
                <button
                  type="button"
                  className="dm-chip border-teal-300 bg-teal-50 text-teal-900 hover:bg-teal-100"
                  onClick={() => {
                    setQuestion(example);
                    void submit(example);
                  }}
                >
                  {example}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </Card>

      <div ref={resultRef}>
        {error ? <ErrorBlock error={error} /> : null}
        {loading ? <LoadingBlock label="Retrieving and assembling an answer" /> : null}

        {!loading && !error && !answer ? (
          <EmptyState title="No question asked yet">
            Pick one of the example questions or type your own. DigitalMind answers only from the
            records currently imported on this machine.
          </EmptyState>
        ) : null}

        {answer ? <AnswerPanel answer={answer} /> : null}
      </div>
    </>
  );
}

function AnswerPanel({ answer }: { answer: AskOut }) {
  return (
    <div className="space-y-5">
      {answer.no_match ? (
        <div
          role="note"
          className="rounded-lg border-2 border-teal-400 bg-teal-50 px-5 py-5"
        >
          <p className="text-lg font-semibold text-teal-950">{NO_MATCH_SENTENCE}</p>
          <p className="mt-2 text-sm text-teal-950">{answer.answer}</p>
          <p className="mt-3 text-sm text-teal-900">
            This describes the records that were searched. It is not a statement that the
            conversation did not happen — the original channel may simply not be imported or
            connected yet.
          </p>
        </div>
      ) : (
        <Card
          title="Answer"
          subtitle={`Intent: ${answer.intent.replace(/_/g, " ")} · confidence ${formatScore(
            answer.confidence,
          )} (retrieval-side)`}
        >
          <AnswerText text={answer.answer} />
          <p className="mt-3 text-xs text-ink-600">
            <strong>Confidence basis:</strong> {answer.confidence_basis} This is a measure of how
            much of the analyzed corpus agrees, not a probability that the answer is true.
          </p>
          {answer.interpretation.length > 0 ? (
            <div className="mt-4 rounded-md border border-ink-200 bg-ink-50 px-3 py-2">
              <p className="dm-section-title">Interpretation</p>
              <ul className="mt-1 list-inside list-disc text-sm text-ink-800">
                {answer.interpretation.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </Card>
      )}

      <Card
        title={`Citations (${answer.citations.length})`}
        subtitle="Each citation is a verbatim quote with the record and character range it came from."
      >
        {answer.citations.length === 0 ? (
          <EmptyState title="No citation could be produced">
            Nothing in the searched records supported a claim, so no quotation is offered.
          </EmptyState>
        ) : (
          <ol className="space-y-3">
            {answer.citations.map((citation, index) => (
              <li
                key={`${citation.record_id}-${citation.start_char}-${index}`}
                className="rounded-md border border-ink-200 bg-white px-3 py-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-semibold text-ink-500">#{index + 1}</span>
                    <SourceBadge sourceType={citation.source_type} />
                    <span className="text-sm font-medium text-ink-900">{citation.title}</span>
                  </span>
                  <span className="text-xs text-ink-600">{formatDateTime(citation.occurred_at)}</span>
                </div>
                <blockquote className="mt-2 border-l-2 border-teal-600 bg-teal-50/60 px-3 py-2 text-sm italic text-ink-900">
                  {citation.quote || "no quote text supplied"}
                </blockquote>
                <p className="mt-1 text-xs text-ink-600">
                  Found by {citation.retriever} · characters {citation.start_char}–{citation.end_char}
                  {citation.score !== null ? ` · score ${citation.score.toFixed(3)}` : ""}
                </p>
                <a
                  href={`#${recordLink(citation.record_id, citation.start_char, citation.end_char)}`}
                  className="dm-btn mt-2"
                >
                  Open record at this passage
                </a>
              </li>
            ))}
          </ol>
        )}
      </Card>

      {answer.facts.length > 0 ? (
        <Card
          title={`Cited facts (${answer.facts.length})`}
          subtitle="The individual statements behind the answer, each linked to its source."
        >
          <ul className="space-y-2">
            {answer.facts.map((fact, index) => (
              <li key={`${fact.record_id}-${index}`} className="border-b border-ink-100 pb-2 last:border-0 last:pb-0">
                {typeof fact.start_char === "number" && typeof fact.end_char === "number" ? (
                  <a
                    href={`#${recordLink(fact.record_id, fact.start_char, fact.end_char)}`}
                    className="text-sm text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
                  >
                    “{fact.quote ?? "open this passage"}”
                  </a>
                ) : (
                  <span className="text-sm text-ink-800">
                    {fact.quote ?? `Record ${fact.record_id}`}
                  </span>
                )}
                <span className="ml-2 text-xs text-ink-600">record {fact.record_id}</span>
                {fact.note ? <p className="mt-0.5 text-xs text-ink-600">{fact.note}</p> : null}
              </li>
            ))}
          </ul>
        </Card>
      ) : null}

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="What was searched" subtitle="Coverage of the answer">
          <Coverage coverage={answer.coverage} />
        </Card>
        <Card title="Method" subtitle="How the answer was assembled">
          <Method method={answer.method} />
          <div className="mt-3 rounded-md border border-ink-300 bg-ink-50 px-3 py-2">
            <p className="dm-section-title">Caveat</p>
            <p className="mt-1 text-sm text-ink-800">{answer.caveat}</p>
          </div>
        </Card>
      </div>

      {!answer.no_match ? (
        <p className={cx("text-xs text-ink-600")}>
          Every sentence above was assembled from the citations listed. DigitalMind does not write
          claims that are not in the quoted text.
        </p>
      ) : null}
    </div>
  );
}

function AnswerText({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <div className="space-y-1 text-sm leading-6 text-ink-900">
      {lines.map((line, index) =>
        line.startsWith("- ") ? (
          <p key={index} className="pl-4">
            • {line.slice(2)}
          </p>
        ) : line.trim() === "" ? (
          <div key={index} className="h-1" />
        ) : (
          <p key={index}>{line}</p>
        ),
      )}
    </div>
  );
}

function Coverage({ coverage }: { coverage: Record<string, unknown> }) {
  const dateRange = coverage.date_range as { from: string; to: string } | null | undefined;
  const entries: [string, string][] = [
    ["Records in the database", String(coverage.records_in_database ?? "not stated")],
    ["Date range", dateRange ? `${formatDateTime(dateRange.from)} → ${formatDateTime(dateRange.to)}` : "no record carries a timestamp"],
    ["Source types searched", Array.isArray(coverage.source_types) ? coverage.source_types.join(", ") : "not stated"],
    ["Integrations", String(coverage.integrations ?? "not stated")],
    ["Query actually used", String(coverage.query_used ?? "not stated")],
  ];
  return (
    <dl className="space-y-2">
      {entries.map(([term, value]) => (
        <div key={term} className="flex flex-wrap items-baseline gap-x-3 border-b border-ink-100 pb-2 last:border-0">
          <dt className="w-full text-xs font-semibold uppercase tracking-wide text-ink-500 sm:w-48">
            {term}
          </dt>
          <dd className="min-w-0 flex-1 text-sm text-ink-800">{value}</dd>
        </div>
      ))}
      <Callout tone="neutral">
        <p>
          Only imported data is searched. No third-party channel is connected, so anything that was
          never imported cannot appear here.
        </p>
      </Callout>
    </dl>
  );
}

function Method({ method }: { method: Record<string, unknown> }) {
  const channels = method.channels as Record<string, number> | undefined;
  const degraded = Array.isArray(method.degraded) ? (method.degraded as string[]) : [];
  return (
    <div className="space-y-2">
      <dl className="space-y-2">
        <div className="flex flex-wrap items-baseline gap-x-3">
          <dt className="w-full text-xs font-semibold uppercase tracking-wide text-ink-500 sm:w-48">
            Mode
          </dt>
          <dd className="min-w-0 flex-1 text-sm text-ink-800">{String(method.mode ?? "not stated")}</dd>
        </div>
        <div className="flex flex-wrap items-baseline gap-x-3">
          <dt className="w-full text-xs font-semibold uppercase tracking-wide text-ink-500 sm:w-48">
            Description
          </dt>
          <dd className="min-w-0 flex-1 text-sm text-ink-800">
            {String(method.description ?? "not stated")}
          </dd>
        </div>
      </dl>
      {channels ? (
        <Disclosure summary="Channel contributions">
          <ul className="space-y-0.5">
            {Object.entries(channels).map(([name, count]) => (
              <li key={name}>
                {name}: {String(count)} result(s)
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}
      {typeof method.explanation === "string" && method.explanation ? (
        <p className="text-xs text-ink-700">{method.explanation}</p>
      ) : null}
      {degraded.length > 0 ? (
        <p className="rounded-md border border-ink-300 bg-ink-50 px-3 py-2 text-xs text-ink-700">
          Degraded channels: {degraded.join(", ")}. The answer uses only the channels that were
          available.
        </p>
      ) : null}
    </div>
  );
}

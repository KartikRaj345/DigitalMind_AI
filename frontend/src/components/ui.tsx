import { type ReactNode } from "react";

import { riskLabelDisplay } from "../lib/format";

export function cx(...values: (string | false | null | undefined)[]): string {
  return values.filter(Boolean).join(" ");
}

// ---------------------------------------------------------------------------
// layout primitives
// ---------------------------------------------------------------------------

export function Card({
  title,
  subtitle,
  actions,
  children,
  className,
  id,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section id={id} className={cx("dm-card", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-ink-200 px-4 py-3">
          <div className="min-w-0">
            {title ? <h2 className="text-sm font-semibold text-ink-900">{title}</h2> : null}
            {subtitle ? <p className="mt-1 text-xs text-ink-600">{subtitle}</p> : null}
          </div>
          {actions ? <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div> : null}
        </header>
      )}
      <div className="px-4 py-4">{children}</div>
    </section>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow ? <p className="dm-section-title">{eyebrow}</p> : null}
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink-900">{title}</h1>
        {description ? <p className="mt-2 max-w-3xl text-sm text-ink-600">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

export function LoadingBlock({ label = "Loading" }: { label?: string }) {
  return (
    <p role="status" aria-live="polite" className="dm-loading-dot flex items-center gap-2 text-sm text-ink-600">
      <span aria-hidden="true" className="inline-block h-2 w-2 rounded-full bg-teal-700" />
      {label}…
    </p>
  );
}

export function ErrorBlock({
  error,
  onRetry,
}: {
  error: unknown;
  onRetry?: () => void;
}) {
  const message =
    error instanceof Error ? error.message : typeof error === "string" ? error : "Unknown error";
  return (
    <div role="alert" className="rounded-md border border-risk-300 bg-risk-50 px-4 py-3 text-sm text-risk-900">
      <p className="font-semibold">Something went wrong talking to the local analysis service.</p>
      <p className="mt-1">{message}</p>
      {onRetry ? (
        <button type="button" className="dm-btn mt-3" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({
  title,
  children,
  tone = "neutral",
}: {
  title: string;
  children?: ReactNode;
  tone?: "neutral" | "risk" | "info";
}) {
  const toneClass =
    tone === "risk"
      ? "border-risk-300 bg-risk-50 text-risk-900"
      : tone === "info"
        ? "border-teal-300 bg-teal-50 text-teal-950"
        : "border-ink-200 bg-ink-50 text-ink-800";
  return (
    <div className={cx("rounded-md border px-4 py-5", toneClass)}>
      <p className="text-sm font-semibold">{title}</p>
      {children ? <div className="mt-2 text-sm">{children}</div> : null}
    </div>
  );
}

/**
 * Absence copy. Always states what was searched; never implies that an event did
 * not happen because no record was found.
 */
export function NoMatchNote({ children }: { children: ReactNode }) {
  return (
    <div
      role="note"
      className="rounded-md border border-teal-300 bg-teal-50 px-4 py-4 text-sm text-teal-950"
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// chips
// ---------------------------------------------------------------------------

const SOURCE_TONE: Record<string, string> = {
  email: "border-blue-300 bg-blue-50 text-blue-900",
  message: "border-teal-300 bg-teal-50 text-teal-900",
  calendar: "border-indigo-300 bg-indigo-50 text-indigo-900",
  document: "border-ink-300 bg-ink-100 text-ink-800",
  note: "border-ink-300 bg-ink-100 text-ink-800",
  task: "border-violet-300 bg-violet-50 text-violet-900",
};

export function SourceBadge({ sourceType }: { sourceType: string }) {
  return (
    <span className={cx("dm-chip", SOURCE_TONE[sourceType] ?? "border-ink-300 bg-ink-100 text-ink-800")}>
      {sourceType}
    </span>
  );
}

/**
 * Risk is the only amber element in the product. The chip always says what it is,
 * never a bare number.
 */
export function RiskChip({
  labels,
  severity,
  showCount = false,
}: {
  labels: string[];
  severity: "none" | "notice" | "warning";
  showCount?: boolean;
}) {
  if (labels.length === 0 || severity === "none") {
    return (
      <span className="dm-chip border-ink-300 bg-white text-ink-700">No indicators detected</span>
    );
  }
  const tone =
    severity === "warning"
      ? "border-risk-400 bg-risk-100 text-risk-900"
      : "border-risk-300 bg-risk-50 text-risk-900";
  const names = labels.map(riskLabelDisplay).join(", ");
  return (
    <span className={cx("dm-chip", tone)}>
      <span aria-hidden="true">▲</span>
      Potential manipulation risk detected
      <span className="sr-only">: {names}</span>
      <span className="font-normal">({names})</span>
      {showCount ? <span className="text-risk-800">· {labels.length} indicator(s)</span> : null}
    </span>
  );
}

export function SeverityBadge({ severity }: { severity: string }) {
  if (severity === "warning") {
    return (
      <span className="dm-chip border-risk-400 bg-risk-100 text-risk-900">Warning level</span>
    );
  }
  if (severity === "notice") {
    return <span className="dm-chip border-risk-200 bg-risk-50 text-risk-900">Notice level</span>;
  }
  return <span className="dm-chip border-ink-300 bg-white text-ink-700">No level</span>;
}

export function NeutralChip({ children }: { children: ReactNode }) {
  return <span className="dm-chip border-ink-300 bg-white text-ink-700">{children}</span>;
}

export function LaneChip({ lane }: { lane: string }) {
  const tone =
    lane === "recorded"
      ? "border-teal-300 bg-teal-50 text-teal-900"
      : lane === "explicit"
        ? "border-blue-300 bg-blue-50 text-blue-900"
        : "border-ink-400 bg-ink-100 text-ink-900";
  return <span className={cx("dm-chip", tone)}>{lane}</span>;
}

// ---------------------------------------------------------------------------
// data display
// ---------------------------------------------------------------------------

export function Stat({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: string | number;
  hint?: string;
  tone?: "neutral" | "risk" | "teal";
}) {
  const valueClass =
    tone === "risk"
      ? "text-risk-800"
      : tone === "teal"
        ? "text-teal-800"
        : "text-ink-900";
  return (
    <div className="rounded-md border border-ink-200 bg-white px-3 py-3">
      <p className="dm-section-title">{label}</p>
      <p className={cx("mt-1 text-2xl font-semibold tabular-nums", valueClass)}>{value}</p>
      {hint ? <p className="mt-1 text-xs text-ink-600">{hint}</p> : null}
    </div>
  );
}

export function DefinitionRow({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-ink-100 py-2 last:border-0">
      <dt className="w-full text-xs font-semibold uppercase tracking-wide text-ink-500 sm:w-56">
        {term}
      </dt>
      <dd className="min-w-0 flex-1 text-sm text-ink-800">{children}</dd>
    </div>
  );
}

export function AvailabilityRow({
  name,
  available,
  detail,
}: {
  name: string;
  available: boolean;
  detail?: ReactNode;
}) {
  return (
    <li className="flex flex-wrap items-start justify-between gap-2 border-b border-ink-100 py-2 last:border-0">
      <div className="min-w-0">
        <p className="text-sm font-medium text-ink-900">{name}</p>
        {detail ? <p className="mt-0.5 text-xs text-ink-600">{detail}</p> : null}
      </div>
      <span
        className={cx(
          "dm-chip shrink-0",
          available ? "border-teal-300 bg-teal-50 text-teal-900" : "border-ink-300 bg-ink-100 text-ink-700",
        )}
      >
        {available ? "available" : "not available"}
      </span>
    </li>
  );
}

export function Disclosure({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="group" open={defaultOpen}>
      <summary className="cursor-pointer list-none text-sm font-medium text-teal-800 marker:hidden">
        <span className="inline-flex items-center gap-2">
          <span aria-hidden="true" className="transition-transform group-open:rotate-90">
            ›
          </span>
          {summary}
        </span>
      </summary>
      <div className="mt-2 pl-4 text-sm text-ink-700">{children}</div>
    </details>
  );
}

/** Link that navigates to a record and highlights a character range. */
export function CitationLink({
  href,
  children,
  className,
  title,
}: {
  href: string;
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <a
      href={`#${href}`}
      title={title}
      className={cx(
        "rounded text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50",
        className,
      )}
    >
      {children}
    </a>
  );
}

export function Callout({
  tone = "neutral",
  title,
  children,
}: {
  tone?: "neutral" | "risk" | "teal";
  title?: ReactNode;
  children: ReactNode;
}) {
  const toneClass =
    tone === "risk"
      ? "border-risk-300 bg-risk-50 text-risk-950"
      : tone === "teal"
        ? "border-teal-300 bg-teal-50 text-teal-950"
        : "border-ink-200 bg-ink-50 text-ink-800";
  return (
    <div className={cx("rounded-md border px-3 py-3 text-sm", toneClass)}>
      {title ? <p className="text-sm font-semibold">{title}</p> : null}
      <div className={cx(title ? "mt-1" : "")}>{children}</div>
    </div>
  );
}

/** Presentation helpers shared by the views. */

export const RISK_LABEL_DISPLAY: Record<string, string> = {
  urgency: "Urgency",
  threat_language: "Threat language",
  emotional_pressure: "Emotional pressure",
  payment_request: "Payment request",
};

/** The only order the indicators are ever shown in. */
export const RISK_LABEL_ORDER = [
  "payment_request",
  "threat_language",
  "urgency",
  "emotional_pressure",
] as const;

export function riskLabelDisplay(label: string): string {
  return RISK_LABEL_DISPLAY[label] ?? label.replace(/_/g, " ");
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return "no timestamp";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "unknown date";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(date);
}

export function formatRange(start: string | null, end: string | null): string {
  if (!start) return "no timestamp";
  const from = formatDateTime(start);
  if (!end) return from;
  return `${from} → ${formatDateTime(end)}`;
}

export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / 1024 ** index;
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  return value.toFixed(2);
}

export function titleCase(value: string): string {
  return value
    .split(/[\s_]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

/** Escape-free truncation for long record bodies. */
export function truncate(value: string, limit: number): string {
  const collapsed = value.replace(/\s+/g, " ").trim();
  if (collapsed.length <= limit) return collapsed;
  return `${collapsed.slice(0, limit - 1).trimEnd()}…`;
}

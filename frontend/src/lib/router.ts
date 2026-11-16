/**
 * Tiny hash router. No dependency.
 *
 * Routes look like `#/records/12?span=100-140`. `useRoute()` returns the parsed
 * path segments plus the query string so views can read state from the URL, which
 * keeps citation links (record + span) shareable and back/forward friendly.
 */

import { useCallback, useEffect, useState } from "react";

export type RouteName =
  | "dashboard"
  | "records"
  | "record"
  | "graph"
  | "timeline"
  | "risk"
  | "ask"
  | "insights"
  | "import"
  | "settings";

export interface Route {
  name: RouteName;
  /** Path segments after the route name, e.g. ["12"] for `#/records/12`. */
  params: string[];
  query: URLSearchParams;
  hash: string;
}

const ROUTE_NAMES: RouteName[] = [
  "dashboard",
  "records",
  "record",
  "graph",
  "timeline",
  "risk",
  "ask",
  "insights",
  "import",
  "settings",
];

function isRouteName(value: string): value is RouteName {
  return (ROUTE_NAMES as string[]).includes(value);
}

export function parseRoute(hash: string): Route {
  const raw = hash.replace(/^#/, "");
  const [pathPart = "", queryPart = ""] = raw.split("?");
  const segments = pathPart.split("/").filter((s) => s.length > 0);
  const first = segments[0] ?? "";
  const name: RouteName = isRouteName(first) ? first : "dashboard";
  // `/records/12` and `/record/12` both address a single record.
  const params = segments.slice(1);
  return { name, params, query: new URLSearchParams(queryPart), hash: `#${raw}` };
}

export function navigate(to: string, options: { replace?: boolean } = {}): void {
  const next = to.startsWith("#") ? to : `#${to}`;
  if (window.location.hash === next) return;
  if (options.replace) {
    const url = `${window.location.pathname}${window.location.search}${next}`;
    window.history.replaceState(null, "", url);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    window.location.hash = next;
  }
}

/** Build a record-detail link, optionally focusing a character range. */
export function recordLink(recordId: number, start?: number | null, end?: number | null): string {
  const params = new URLSearchParams();
  if (typeof start === "number" && typeof end === "number" && end > start) {
    params.set("span", `${start}-${end}`);
  }
  const query = params.toString();
  return `/record/${recordId}${query ? `?${query}` : ""}`;
}

export function useRoute(): Route {
  const [hash, setHash] = useState(() => window.location.hash || "#/dashboard");

  useEffect(() => {
    const onChange = () => setHash(window.location.hash || "#/dashboard");
    window.addEventListener("hashchange", onChange);
    if (!window.location.hash) window.location.hash = "#/dashboard";
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return parseRoute(hash);
}

/** Convenience hook for the record-id route. */
export function useRecordId(route: Route): number | null {
  const raw = route.params[0];
  if (raw === undefined) return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

export function useNavigate(): (to: string, options?: { replace?: boolean }) => void {
  return useCallback((to: string, options?: { replace?: boolean }) => {
    navigate(to, options);
  }, []);
}

import { useEffect, useState } from "react";

import { getHealth, getPrivacyNotice, type Health, type PrivacyNotice } from "../lib/api";
import { navigate, useRoute, type RouteName } from "../lib/router";
import { cx } from "./ui";

interface NavItem {
  name: RouteName;
  path: string;
  label: string;
  hint: string;
}

const NAV: NavItem[] = [
  { name: "dashboard", path: "/dashboard", label: "Dashboard", hint: "Counts, indicators, privacy" },
  { name: "records", path: "/records", label: "Records", hint: "Search and filter every record" },
  { name: "timeline", path: "/timeline", label: "Timeline", hint: "Recorded, explicit and inferred" },
  { name: "graph", path: "/graph", label: "Graph", hint: "Entities and records" },
  { name: "risk", path: "/risk", label: "Risk analysis", hint: "Per-indicator evidence" },
  { name: "ask", path: "/ask", label: "Ask", hint: "Cited answers from your records" },
  { name: "insights", path: "/insights", label: "Insights", hint: "Cross-record conclusions" },
  { name: "import", path: "/import", label: "Import", hint: "Demo data and file upload" },
  { name: "settings", path: "/settings", label: "Settings", hint: "Models, privacy, deletion" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const route = useRoute();
  const [health, setHealth] = useState<Health | null>(null);
  const [notice, setNotice] = useState<PrivacyNotice | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((value) => {
        if (!cancelled) setHealth(value);
      })
      .catch(() => {
        if (!cancelled) setHealth(null);
      });
    getPrivacyNotice()
      .then((value) => {
        if (!cancelled) setNotice(value);
      })
      .catch(() => {
        if (!cancelled) setNotice(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setMenuOpen(false);
  }, [route.hash]);

  const llmEnabled = health?.llm_enabled ?? false;

  return (
    <div className="flex min-h-screen flex-col bg-ink-50">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-white focus:px-3 focus:py-2 focus:text-sm focus:font-semibold focus:text-ink-900 focus:shadow"
      >
        Skip to main content
      </a>

      <header className="border-b border-ink-200 bg-white">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6">
          <a
            href="#/dashboard"
            className="flex items-center gap-2 rounded text-lg font-semibold tracking-tight text-ink-900"
          >
            <span aria-hidden="true" className="inline-block h-3 w-3 rounded-full bg-teal-700" />
            DigitalMind
          </a>

          <nav aria-label="Primary" className="hidden flex-1 lg:block">
            <ul className="flex flex-wrap items-center gap-1">
              {NAV.map((item) => (
                <li key={item.name}>
                  <a
                    href={`#${item.path}`}
                    aria-current={route.name === item.name ? "page" : undefined}
                    title={item.hint}
                    className={cx(
                      "inline-block rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors",
                      route.name === item.name
                        ? "bg-teal-700 text-white"
                        : "text-ink-700 hover:bg-ink-100",
                    )}
                  >
                    {item.label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <span className="hidden text-xs text-ink-600 sm:inline">
              {health ? `v${health.version} · ${health.records} record(s)` : "connecting…"}
            </span>
            <button
              type="button"
              className="dm-btn lg:hidden"
              aria-expanded={menuOpen}
              aria-controls="primary-menu"
              onClick={() => setMenuOpen((open) => !open)}
            >
              {menuOpen ? "Close menu" : "Menu"}
            </button>
          </div>
        </div>

        <nav
          id="primary-menu"
          aria-label="Primary (compact)"
          hidden={!menuOpen}
          className="border-t border-ink-200 bg-white px-4 py-3 lg:hidden"
        >
          <ul className="flex flex-col gap-1">
            {NAV.map((item) => (
              <li key={item.name}>
                <a
                  href={`#${item.path}`}
                  aria-current={route.name === item.name ? "page" : undefined}
                  className={cx(
                    "block rounded-md px-3 py-2 text-sm font-medium",
                    route.name === item.name
                      ? "bg-teal-700 text-white"
                      : "text-ink-700 hover:bg-ink-100",
                  )}
                >
                  <span className="block">{item.label}</span>
                  <span
                    className={cx(
                      "block text-xs",
                      route.name === item.name ? "text-teal-50" : "text-ink-500",
                    )}
                  >
                    {item.hint}
                  </span>
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </header>

      <main id="main" className="mx-auto w-full max-w-[1600px] flex-1 px-4 py-6 sm:px-6">
        {children}
      </main>

      <footer className="mt-6 border-t border-ink-200 bg-white">
        <div className="mx-auto max-w-[1600px] space-y-2 px-4 py-5 text-xs text-ink-700 sm:px-6">
          <p className="font-medium text-ink-900">
            Analysis runs locally. No record content leaves this machine.
          </p>
          <p>
            {notice?.notice ??
              "No record content leaves this machine. DigitalMind runs entirely locally."}{" "}
            {llmEnabled
              ? "An external language model endpoint is configured, so record text can be sent to it when you use features that need it."
              : "The optional external LLM layer is off by default."}
          </p>
          <p className="text-ink-500">
            DigitalMind reports indicators, not verdicts. A positive indicator means “review this
            record”, not “this is an attack”.
          </p>
        </div>
      </footer>
    </div>
  );
}

/** Small helper used by views that need to jump home without a link. */
export function goToDashboard(): void {
  navigate("/dashboard");
}

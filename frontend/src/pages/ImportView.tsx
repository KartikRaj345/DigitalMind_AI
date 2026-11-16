import { useCallback, useEffect, useRef, useState } from "react";

import {
  getImportStatus,
  importDemo,
  importFiles,
  type ImportResult,
  type ImportStatus,
} from "../lib/api";
import { formatBytes } from "../lib/format";
import {
  Callout,
  Card,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  PageHeader,
  cx,
} from "../components/ui";

const ACCEPT = ".txt,.md,.eml,.csv,.json,.ics";

export function ImportView({ onImported }: { onImported: () => void }) {
  const [status, setStatus] = useState<ImportStatus | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [demoBusy, setDemoBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [pending, setPending] = useState<File[]>([]);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const load = useCallback(() => {
    setError(null);
    getImportStatus()
      .then(setStatus)
      .catch(setError);
  }, []);

  useEffect(load, [load]);

  const loadDemo = async () => {
    setDemoBusy(true);
    setError(null);
    try {
      const outcome = await importDemo(true);
      setResult(outcome);
      onImported();
    } catch (err) {
      setError(err);
    } finally {
      setDemoBusy(false);
    }
  };

  const upload = async (files: File[]) => {
    if (files.length === 0) return;
    setUploadBusy(true);
    setError(null);
    try {
      const outcome = await importFiles(files, true);
      setResult(outcome);
      setPending([]);
      onImported();
    } catch (err) {
      setError(err);
    } finally {
      setUploadBusy(false);
    }
  };

  const supported = status?.supported_files ?? [];
  const acceptAttr = supported.length > 0 ? supported.join(",") : ACCEPT;

  return (
    <>
      <PageHeader
        eyebrow="Data in"
        title="Import"
        description="DigitalMind reads files you upload and its own demo corpus. There is no third-party connection to enable, because nothing is connected."
      />

      {error ? <ErrorBlock error={error} /> : null}
      {!status && !error ? <LoadingBlock label="Loading import status" /> : null}

      <div className="grid gap-5 lg:grid-cols-2">
        <Card
          title="1 · Demo data"
          subtitle="A hand-written narrative used to demonstrate the analysis. It replaces any previously imported demo records."
          actions={
            <button
              type="button"
              className="dm-btn dm-btn-primary"
              onClick={loadDemo}
              disabled={demoBusy}
            >
              {demoBusy ? "Loading…" : "Load demo data"}
            </button>
          }
        >
          <p className="text-sm text-ink-700">
            The demo corpus contains two connected scenarios: an escalation that ends in a demand
            for money, and a message that refers to a discussion the records do not contain. Both
            are analysed with the same code path as your own uploads.
          </p>
          <p className="mt-2 text-xs text-ink-600">
            Loading the demo replaces previously imported demo records. Uploaded files are not
            touched.
          </p>
        </Card>

        <Card
          title="2 · Upload files"
          subtitle={`Parsed locally. Accepted types: ${supported.join(", ") || ACCEPT}`}
        >
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              const files = Array.from(event.dataTransfer.files ?? []);
              setPending(files);
              void upload(files);
            }}
            className={cx(
              "rounded-md border-2 border-dashed px-4 py-8 text-center",
              dragActive ? "border-teal-600 bg-teal-50" : "border-ink-300 bg-ink-50",
            )}
          >
            <p className="text-sm font-medium text-ink-900">Drop files here</p>
            <p className="mt-1 text-xs text-ink-600">
              Files are read from disk by the local analysis service. Nothing is uploaded to an
              external host.
            </p>
            <div className="mt-3 flex flex-wrap justify-center gap-2">
              <button
                type="button"
                className="dm-btn"
                onClick={() => inputRef.current?.click()}
                disabled={uploadBusy}
              >
                {uploadBusy ? "Uploading…" : "Choose files"}
              </button>
            </div>
            <input
              ref={inputRef}
              id="file-upload"
              type="file"
              multiple
              accept={acceptAttr}
              className="sr-only"
              onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                setPending(files);
                void upload(files);
                event.target.value = "";
              }}
            />
          </div>

          {pending.length > 0 ? (
            <ul className="mt-3 space-y-1 text-xs text-ink-700">
              {pending.map((file) => (
                <li key={`${file.name}-${file.size}`}>
                  {file.name} · {formatBytes(file.size)}
                  {!supported.some((ext) => file.name.toLowerCase().endsWith(ext)) ? (
                    <span className="ml-2 text-risk-800">unsupported type — the service will report it</span>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}
        </Card>
      </div>

      {result ? (
        <Card
          className="mt-5"
          title="Result of the last import"
          subtitle={`${result.inserted} record(s) inserted · ${result.skipped} skipped`}
        >
          {result.files.length > 0 ? (
            <ul className="space-y-1 text-sm text-ink-800">
              {result.files.map((file) => (
                <li key={file.filename} className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-100 pb-1 last:border-0">
                  <span>{file.filename}</span>
                  <span className="text-xs tabular-nums text-ink-600">{file.records} record(s)</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-ink-700">
              {result.inserted} record(s) were added and re-analysed. Derived entities, events, risk
              assessments, graph edges and insights were rebuilt from them.
            </p>
          )}
          {result.errors.length > 0 ? (
            <div className="mt-3">
              <p className="dm-section-title">Errors</p>
              <ul className="mt-1 list-inside list-disc text-sm text-risk-900">
                {result.errors.map((message) => (
                  <li key={message}>{message}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {result.record_ids.length > 0 ? (
            <p className="mt-3 text-xs text-ink-600">
              Record ids: {result.record_ids.join(", ")}
            </p>
          ) : null}
          <a href="#/records" className="dm-btn mt-3">
            Review the imported records
          </a>
        </Card>
      ) : null}

      <Card
        className="mt-5"
        title="3 · Third-party integrations"
        subtitle="None of these are connected. Export the data and upload it instead."
      >
        {status ? (
          <>
            <ul className="space-y-3">
              {status.integrations.map((integration) => (
                <li
                  key={integration.key}
                  className="rounded-md border border-ink-200 bg-white px-3 py-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-sm font-medium text-ink-900">{integration.label}</span>
                    <span className="dm-chip border-ink-300 bg-ink-100 text-ink-800">
                      not connected
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-ink-700">{integration.note}</p>
                </li>
              ))}
            </ul>
            <div className="mt-4 rounded-md border border-teal-300 bg-teal-50 px-3 py-3">
              <p className="text-sm font-medium text-teal-950">Why nothing is connected</p>
              <p className="mt-1 text-sm text-teal-950">
                A live connection would mean sending your records to another company's servers.
                DigitalMind does not do that, so there is nothing to connect. Use each service's own
                export and upload the file here.
              </p>
            </div>
            {status.privacy_note ? (
              <Callout tone="neutral">
                <p>{status.privacy_note}</p>
              </Callout>
            ) : null}
          </>
        ) : error ? (
          <EmptyState title="Import status unavailable">
            The local analysis service did not answer. Start it and reload this page.
          </EmptyState>
        ) : (
          <LoadingBlock label="Loading integrations" />
        )}
      </Card>
    </>
  );
}

/**
 * Typed API client for DigitalMind.
 *
 * Every interface here mirrors a response shape in `backend/app/schemas.py` (or a
 * literal dict returned by a route in `backend/app/api.py`). Field names are copied
 * from the backend, never invented.
 */

const BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: init?.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...init,
  });
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && payload !== null && "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`;
    throw new ApiError(response.status, detail);
  }
  return payload as T;
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

// ---------------------------------------------------------------------------
// health / meta
// ---------------------------------------------------------------------------

export interface Health {
  status: string;
  version: string;
  records: number;
  spacy_available: boolean;
  transformers_available: boolean;
  sentence_transformers_available: boolean;
  faiss_available: boolean;
  llm_enabled: boolean;
  external_transmission: boolean;
}

export interface PrivacyNotice {
  local_first: boolean;
  external_transmission: boolean;
  notice: string;
  integrations: string;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

export function getPrivacyNotice(): Promise<PrivacyNotice> {
  return request<PrivacyNotice>("/privacy-notice");
}

// ---------------------------------------------------------------------------
// dashboard
// ---------------------------------------------------------------------------

export interface ModelComponent {
  available: boolean;
  description?: string;
  thresholds?: Record<string, number> | null;
  calibration?: string | null;
}

export interface ModelsStatus {
  risk: {
    rules: { available: boolean; description: string };
    tfidf_lr: ModelComponent;
    distilbert: ModelComponent;
    notes: string[];
  };
  ner: { available: boolean; model: string; note: string };
  retrieval: {
    bm25: { available: boolean; indexed_chunks: number };
    embeddings: {
      available: boolean;
      model: string | null;
      indexed: number;
      faiss: boolean;
      fallback: string | null;
    };
  };
}

export interface Dashboard {
  records: { total: number; by_source_type: Record<string, number> };
  risk: {
    records_with_indicators: number;
    warning_records: number;
    notice_records: number;
    indicator_counts: Record<string, number>;
    disclaimer: string;
  };
  graph: { entities: number; relations: number };
  timeline: { events: number };
  insights: { active: number; dismissed: number; corrected: number };
  models: ModelsStatus;
  privacy: { external_transmission: boolean; summary: string };
}

export function getDashboard(): Promise<Dashboard> {
  return request<Dashboard>("/dashboard");
}

// ---------------------------------------------------------------------------
// records
// ---------------------------------------------------------------------------

export interface RecordSummary {
  id: number;
  source_type: string;
  title: string;
  occurred_at: string | null;
  imported_at: string;
  origin: string;
  participants: string[];
  char_count: number;
  risk_labels: string[];
  severity: "none" | "notice" | "warning";
  snippet: string;
}

export interface RecordList {
  total: number;
  items: RecordSummary[];
}

export interface Span {
  text: string;
  label: string;
  start_char: number;
  end_char: number;
  confidence: number;
  extractor: string;
  normalized: string | null;
  entity_id: number | null;
  entity_name: string | null;
}

export interface Event {
  id: number | null;
  title: string;
  description: string;
  kind: string;
  certainty: string;
  start_at: string | null;
  end_at: string | null;
  precision: string;
  raw_expression: string;
  start_char: number | null;
  end_char: number | null;
  confidence: number;
  extractor: string;
  note: string;
}

export interface Evidence {
  start_char: number | null;
  end_char: number | null;
  text: string | null;
  note: string | null;
  polarity: string | null;
  weight: number | null;
}

export interface Indicator {
  label: string;
  display: string;
  present: boolean;
  score: number;
  threshold: number;
  methods: Record<string, number>;
  explanation: string;
  evidence: Evidence[];
  calibrated: boolean;
}

export interface EstimateContribution {
  label: string;
  display: string;
  weight: number;
  score: number;
  present: boolean;
  contribution: number;
}

export interface OverallEstimate {
  is_estimate: boolean;
  value: number;
  formula: string;
  contributions: EstimateContribution[];
  caveat: string;
  flagged_count: number;
  methods: string[];
}

export interface RecordEntity {
  id: number;
  name: string;
  type: string;
  mention_count: number;
  resolution_confidence: number;
  surface_forms: string[];
}

export interface Backreference {
  start_char: number;
  end_char: number;
  text: string;
}

export interface ChunkRange {
  ordinal: number;
  start_char: number;
  end_char: number;
}

export interface RecordDetail {
  id: number;
  source_type: string;
  title: string;
  body: string;
  occurred_at: string | null;
  imported_at: string;
  origin: string;
  external_id: string | null;
  participants: string[];
  extra: Record<string, unknown>;
  spans: Span[];
  events: Event[];
  indicators: Indicator[];
  overall_estimate: OverallEstimate;
  entities: RecordEntity[];
  backreferences: Backreference[];
  chunks: ChunkRange[];
}

export function listRecords(params: {
  q?: string;
  source_type?: string;
  risk?: boolean;
  limit?: number;
  offset?: number;
}): Promise<RecordList> {
  return request<RecordList>(`/records${qs(params)}`);
}

export function getRecord(id: number): Promise<RecordDetail> {
  return request<RecordDetail>(`/records/${id}`);
}

export interface DeleteRecordResult {
  deleted: {
    record_id: number;
    removed: Record<string, number>;
    orphaned_entities_removed: number;
    index_invalidated: boolean;
    verified: boolean;
    problems: string[];
  };
  message: string;
}

export function deleteRecord(id: number): Promise<DeleteRecordResult> {
  return request<DeleteRecordResult>(`/records/${id}`, { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// search / ask
// ---------------------------------------------------------------------------

export type RetrievalMode = "lexical" | "semantic" | "hybrid";

export interface SearchHit {
  record_id: number;
  chunk_id: number;
  text: string;
  start_char: number;
  end_char: number;
  score: number;
  rank: number;
  lexical_rank: number | null;
  semantic_rank: number | null;
  agreement: string;
  title: string | null;
  source_type: string | null;
  occurred_at: string | null;
}

export interface SearchResult {
  query: string;
  mode: string;
  hits: SearchHit[];
  channels: Record<string, number>;
  degraded: string[];
  explanation: string;
  total_records: number;
}

export function search(params: {
  q: string;
  mode?: RetrievalMode;
  limit?: number;
}): Promise<SearchResult> {
  return request<SearchResult>(`/search${qs(params)}`);
}

export interface Citation {
  record_id: number;
  source_type: string;
  title: string;
  occurred_at: string | null;
  start_char: number;
  end_char: number;
  quote: string;
  retriever: string;
  score: number | null;
}

export interface AskFact {
  record_id: number;
  start_char?: number | null;
  end_char?: number | null;
  quote?: string | null;
  note?: string | null;
  label?: string;
  indicator?: string;
  lane?: string;
}

export interface AskOut {
  question: string;
  intent: string;
  answer: string;
  citations: Citation[];
  facts: AskFact[];
  interpretation: string[];
  coverage: Record<string, unknown>;
  method: Record<string, unknown>;
  caveat: string;
  no_match: boolean;
  confidence: number;
  confidence_basis: string;
}

export function ask(payload: {
  question: string;
  mode?: RetrievalMode;
  top_k?: number;
}): Promise<AskOut> {
  return request<AskOut>("/ask", {
    method: "POST",
    body: JSON.stringify({ question: payload.question, mode: payload.mode ?? "hybrid", top_k: payload.top_k ?? 8 }),
  });
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string;
  kind: string;
  label: string;
  entity_type: string | null;
  mention_count: number | null;
  source_type: string | null;
  occurred_at: string | null;
  record_id: number | null;
  resolution_confidence: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation_type: string;
  confidence: number;
  record_id: number | null;
  start_char: number | null;
  end_char: number | null;
  method: string;
  weight: number;
}

export interface GraphResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  centrality: Record<string, Record<string, number>>;
  explanation: string;
}

export function getGraph(params?: { record_ids?: string; max_nodes?: number }): Promise<GraphResult> {
  return request<GraphResult>(`/graph${qs(params ?? {})}`);
}

export interface EntityMention {
  record_id: number;
  text: string;
  start_char: number;
  end_char: number;
  confidence: number;
  extractor: string;
}

export interface EntityDetail {
  id: number;
  name: string;
  type: string;
  mention_count: number;
  resolution_confidence: number;
  aliases: string[];
  merges: { target: string; alias: string; kind: string; score: number; detail: string }[];
  mentions: EntityMention[];
  caveat: string;
}

export function getEntity(id: number): Promise<EntityDetail> {
  return request<EntityDetail>(`/entities/${id}`);
}

// ---------------------------------------------------------------------------
// timeline
// ---------------------------------------------------------------------------

export type TimelineLane = "recorded" | "explicit" | "inferred";

export interface TimelineItem {
  record_id: number;
  title: string;
  source_type: string;
  lane: TimelineLane;
  kind: string;
  start_at: string | null;
  end_at: string | null;
  sort_key: string;
  certainty: string;
  precision: string;
  raw_expression: string;
  confidence: number;
  explanation: string;
  start_char: number | null;
  end_char: number | null;
  has_risk: boolean;
  risk_labels: string[];
}

export interface TimelineResult {
  items: TimelineItem[];
  summary: {
    counts: Record<string, number>;
    explanations: Record<string, string>;
    note: string;
  };
}

export function getTimeline(params?: { record_ids?: string; limit?: number }): Promise<TimelineResult> {
  return request<TimelineResult>(`/timeline${qs(params ?? {})}`);
}

// ---------------------------------------------------------------------------
// risk
// ---------------------------------------------------------------------------

export interface RiskRecord {
  record_id: number;
  title: string;
  source_type: string;
  occurred_at: string | null;
  severity: "none" | "notice" | "warning";
  headline: string;
  indicators: Indicator[];
  overall_estimate: OverallEstimate;
}

export interface RiskOverview {
  records: RiskRecord[];
  models: ModelsStatus["risk"];
  method_note: string;
  benign_note: string;
}

export function getRisk(): Promise<RiskOverview> {
  return request<RiskOverview>("/risk");
}

// ---------------------------------------------------------------------------
// insights
// ---------------------------------------------------------------------------

export type InsightStatus = "active" | "dismissed" | "corrected";

export interface Insight {
  id: number;
  kind: string;
  severity: string;
  title: string;
  statement: string;
  facts: AskFact[];
  evidence: AskFact[];
  confidence: number;
  confidence_basis: string;
  status: InsightStatus;
  user_note: string | null;
  record_ids: number[];
  entity_ids: number[];
  created_at: string;
}

export function listInsights(params?: {
  status?: "active" | "dismissed" | "corrected" | "all";
  kind?: string;
}): Promise<Insight[]> {
  return request<Insight[]>(`/insights${qs(params ?? {})}`);
}

export function updateInsight(
  id: number,
  payload: { status?: InsightStatus; user_note?: string | null },
): Promise<Insight> {
  return request<Insight>(`/insights/${id}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// import
// ---------------------------------------------------------------------------

export interface ImportResult {
  inserted: number;
  skipped: number;
  record_ids: number[];
  files: { filename: string; records: number }[];
  errors: string[];
}

export function importDemo(reanalyze = true): Promise<ImportResult> {
  return request<ImportResult>(`/import/demo${qs({ reanalyze })}`, { method: "POST" });
}

export function importFiles(files: File[], reanalyze = true): Promise<ImportResult> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  form.append("reanalyze", String(reanalyze));
  return request<ImportResult>("/import/files", { method: "POST", body: form });
}

export interface ImportStatus {
  integrations: { key: string; label: string; configured: boolean; status: string; note: string }[];
  supported_files: string[];
  privacy_note: string;
}

export function getImportStatus(): Promise<ImportStatus> {
  return request<ImportStatus>("/import/status");
}

// ---------------------------------------------------------------------------
// settings / privacy / analysis
// ---------------------------------------------------------------------------

export interface PrivacyReport {
  counts: Record<string, number>;
  database_bytes: number;
  database_path: string;
  external_transmission: boolean;
  llm: {
    enabled: boolean;
    endpoint: string | null;
    model: string | null;
    disclosure: string;
    integrations_configured: boolean;
    note: string;
  };
  log_policy: string;
  delete_semantics: string;
}

export interface DataSource {
  key: string;
  label: string;
  kind: string;
  configured: boolean;
  record_count: number;
}

export interface SettingsOut {
  retrieval_mode: RetrievalMode;
  min_evidence_confidence: number;
  risk_models: ModelsStatus["risk"];
  spacy: ModelsStatus["ner"];
  embeddings: ModelsStatus["retrieval"]["embeddings"];
  privacy: PrivacyReport;
  data_sources: DataSource[];
}

export function getSettings(): Promise<SettingsOut> {
  return request<SettingsOut>("/settings");
}

export function updateSettings(payload: {
  retrieval_mode?: RetrievalMode;
  min_evidence_confidence?: number;
  allow_external_llm?: boolean;
}): Promise<SettingsOut> {
  return request<SettingsOut>("/settings", { method: "PATCH", body: JSON.stringify(payload) });
}

export function getPrivacy(): Promise<PrivacyReport> {
  return request<PrivacyReport>("/privacy");
}

export function deleteAllData(): Promise<{ deleted: Record<string, number>; message: string }> {
  return request<{ deleted: Record<string, number>; message: string }>("/settings/delete-all", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function reindex(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/analysis/reindex", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function rerunAnalysis(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/analysis/rerun", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function rerunResolution(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/analysis/resolve", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

// /src/lib/api.ts
// Centralized API client for Next.js frontend
//
// Fixes:
//  - Do NOT default to 0.0.0.0 (not a routable host)
//  - Do NOT use relative fetch("/api/...") for backend calls (hits Next server)
//  - One base URL used everywhere

export function getApiBaseUrl() {
  // Prefer env in dev/build pipelines
  const envUrl = (process.env.NEXT_PUBLIC_API_URL || "").trim();

  // Optional runtime override (useful for your future .exe)
  // You can set this from UI settings and persist it in localStorage.
  const runtimeUrl =
    (typeof window !== "undefined" && localStorage.getItem("AION_API_BASE_URL")) || "";

  const raw = (runtimeUrl || envUrl || "http://127.0.0.1:8000").trim();

  // Normalize: remove trailing slashes
  return raw.replace(/\/+$/, "");
}

async function request(path: string, init: RequestInit = {}) {
  const base = getApiBaseUrl();
  const url = path.startsWith("http")
    ? path
    : `${base}${path.startsWith("/") ? "" : "/"}${path}`;

  const res = await fetch(url, {
    ...init,
    // Next.js: avoid caching for live dashboards
    cache: "no-store",
  });

  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }

  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) ? (data.detail || data.message) : text;
    throw new Error(`${init.method || "GET"} ${url} failed: ${res.status} ${res.statusText} — ${detail}`);
  }

  return data;
}

async function get(path: string) {
  return request(path, { method: "GET" });
}

async function post(path: string, body?: any) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

// ---- Your exported API wrappers (examples) ----

export async function getIntradaySnapshot(limit: number = 120) {
  // IMPORTANT: use absolute backend base, not relative /api/*
  return get(`/api/intraday/snapshot?limit=${limit}`);
}

// ---- Settings / Configuration API ----

/**
 * Get knobs.env file content for EOD/Nightly configuration
 */
export async function getKnobs(): Promise<{ content: string }> {
  return get("/api/settings/knobs");
}

/**
 * Save knobs.env file content
 */
export async function saveKnobs(content: string): Promise<{ ok: boolean; message: string }> {
  return post("/api/settings/knobs", { content });
}

/**
 * Get dt_knobs.env file content for Intraday/day-trading configuration
 */
export async function getDtKnobs(): Promise<{ content: string }> {
  return get("/api/settings/dt-knobs");
}

/**
 * Save dt_knobs.env file content
 */
export async function saveDtKnobs(content: string): Promise<{ ok: boolean; message: string }> {
  return post("/api/settings/dt-knobs", { content });
}

/**
 * Get API keys and settings from .env
 */
export async function getSettingsKeys(): Promise<{ keys: Record<string, string> }> {
  return get("/api/settings/keys");
}

/**
 * Update API keys / settings in .env
 */
export async function updateSettings(updates: Record<string, string>): Promise<{ status: string }> {
  return post("/api/settings/update", { updates });
}

export const api = {
  health: () => get("/health"),

  // Settings & Configuration
  settings: {
    getKeys: getSettingsKeys,
    updateKeys: updateSettings,
    getKnobs,
    saveKnobs,
    getDtKnobs,
    saveDtKnobs,
  },

  // NOTE: these endpoints look legacy (not in your openapi dump).
  // Keep them only if they actually exist on the backend.
  // Otherwise they will throw and break pages that call them.
  metrics: (n = 20) => get(`/metrics?n=${n}`),
  buildML: () => post("/build-ml-data"),
  monitorDrift: () => post("/monitor_drift"),
  featureColumns: () => get("/feature-columns"),
};

export type MetricLog = Record<
  string,
  { R2?: number; MAE?: number; RMSE?: number; ACC?: number; F1?: number; AUC?: number }
>;

// ---- Type Definitions for API Responses ----

export interface KnobsResponse {
  content: string;
}

export interface SaveKnobsResponse {
  ok: boolean;
  message: string;
}

export interface SettingsKeysResponse {
  keys: Record<string, string>;
}

export interface UpdateSettingsResponse {
  status: string;
}

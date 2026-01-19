// /frontend/lib/api.ts
// Centralized API client for Next.js frontend
//
// Architecture:
//  - Uses Next.js API proxy routes (/api/backend/* and /api/dt/*)
//  - Proxy routes forward to backend servers avoiding CORS
//  - Backend has NO cache (returns fresh data always)
//  - Client implements localStorage cache with TTL
//  - SSE pushes auto-invalidate cache for real-time updates

import { fetchWithCache } from "./clientCache";

/**
 * Get the base URL for main backend API calls.
 * Always uses Next.js proxy route to avoid CORS.
 */
export function getApiBaseUrl() {
  // In browser: always use Next.js proxy route
  if (typeof window !== "undefined") {
    return "/api/backend";
  }
  
  // Server-side: use configured backend URL or proxy
  return process.env.NEXT_PUBLIC_BACKEND_URL || "/api/backend";
}

/**
 * Get the base URL for DT (intraday) backend API calls.
 * Always uses Next.js proxy route to avoid CORS.
 */
export function getDtApiBaseUrl() {
  // In browser: always use Next.js proxy route
  if (typeof window !== "undefined") {
    return "/api/dt";
  }
  
  // Server-side: use configured DT backend URL or proxy
  return process.env.NEXT_PUBLIC_DT_BACKEND_URL || "/api/dt";
}

/**
 * Make a request to the backend API.
 * Automatically uses the appropriate backend (main or DT).
 */
async function request(path: string, init: RequestInit = {}, useDt: boolean = false) {
  const base = useDt ? getDtApiBaseUrl() : getApiBaseUrl();
  const url = path.startsWith("http")
    ? path
    : `${base}${path.startsWith("/") ? path : `/${path}`}`;

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

/**
 * GET request with optional client-side caching.
 */
async function get(path: string, options?: { cache?: boolean; ttl?: number; useDt?: boolean }) {
  const base = options?.useDt ? getDtApiBaseUrl() : getApiBaseUrl();
  const url = path.startsWith("http")
    ? path
    : `${base}${path.startsWith("/") ? path : `/${path}`}`;
  
  // Support client-side caching for GET requests
  if (options?.cache && typeof window !== "undefined") {
    return fetchWithCache(url, { ttl: options.ttl });
  }
  
  return request(path, { method: "GET" }, options?.useDt);
}

/**
 * POST request with optional body.
 */
async function post(path: string, body?: any, useDt: boolean = false) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  }, useDt);
}

// ---- Exported API wrappers ----

/**
 * Get intraday snapshot data.
 */
export async function getIntradaySnapshot(limit: number = 120, useCache: boolean = false) {
  const path = `/api/intraday/snapshot?limit=${limit}`;
  return get(path, { cache: useCache, ttl: 5000 });
}

/**
 * Get bots page data (unified status for all bot families).
 */
export async function getBotsPage(useCache: boolean = false) {
  const path = "/api/bots/page";
  return get(path, { cache: useCache, ttl: 5000 });
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

// ---- Health & Metrics API ----

/**
 * Get system health status.
 */
export async function getHealth() {
  return get("/health");
}

/**
 * Get model metrics.
 * @param n - Number of metrics to return (default: 20)
 */
export async function getMetrics(n: number = 20) {
  return get(`/api/metrics/overview?limit=${n}`);
}

// ---- Portfolio API ----

/**
 * Get portfolio data (holdings, equity, performance).
 */
export async function getPortfolio() {
  return get("/api/portfolio/holdings/top/1w");
}

// ---- Unified API Object ----

export const api = {
  // Health
  health: getHealth,

  // Settings & Configuration
  settings: {
    getKeys: getSettingsKeys,
    updateKeys: updateSettings,
    getKnobs,
    saveKnobs,
    getDtKnobs,
    saveDtKnobs,
  },

  // Metrics
  metrics: getMetrics,

  // Portfolio
  portfolio: getPortfolio,

  // Legacy endpoints (may not exist on backend, kept for compatibility)
  // These will throw errors if called but not implemented on backend
  buildML: () => post("/api/model/train"),
  monitorDrift: () => get("/api/metrics/drift"),
  featureColumns: () => get("/api/model/status"),
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

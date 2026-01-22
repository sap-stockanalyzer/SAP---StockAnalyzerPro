// /frontend/lib/api.ts
// Centralized API client for Next.js frontend
//
// Architecture:
//  - Uses Next.js API proxy routes (/api/backend/* and /api/dt/*)
//  - Proxy routes forward to backend servers avoiding CORS
//  - Backend has NO cache (returns fresh data always)
//  - Client implements localStorage cache with TTL
//  - SSE pushes auto-invalidate cache for real-time updates
//
// ===== ENDPOINT MAPPINGS =====
//
// Frontend Call                          → Proxy Route              → Backend Endpoint
// ----------------------------------------------------------------------------------------------------
// /api/backend/bots/page                → /api/bots/page          → GET /api/bots/page
// /api/backend/dashboard/metrics        → /dashboard/metrics      → GET /dashboard/metrics (no /api)
// /api/backend/portfolio/holdings/top/1w → /api/portfolio/...     → GET /api/portfolio/holdings/top/1w
// /api/backend/cache/unified            → /api/cache/unified      → GET /api/cache/unified
// /api/backend/settings/keys            → /api/settings/keys      → GET /api/settings/keys
// /api/backend/settings/knobs           → /api/settings/knobs     → GET /api/settings/knobs
// /api/backend/settings/dt-knobs        → /api/settings/dt-knobs  → GET /api/settings/dt-knobs
// /api/backend/metrics/overview         → /api/metrics/overview   → GET /api/metrics/overview
// /api/backend/models/list              → /api/models/list        → GET /api/models/list
// /api/backend/reports/list             → /api/reports/list       → GET /api/reports/list
// /api/backend/admin/*                  → /admin/*                → /admin/* (no /api prefix)
//
// Note: The Next.js proxy in /app/api/backend/[...path]/route.ts automatically:
//       - Adds /api prefix for most routes
//       - Skips /api prefix for dashboard/* and admin/* routes
//       - Preserves query strings and request bodies

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
 * NEW CONSOLIDATED ENDPOINT: /api/page/bots
 */
export async function getBotsPage(useCache: boolean = false) {
  const path = "/api/page/bots";
  return get(path, { cache: useCache, ttl: 5000 });
}

/**
 * Get profile/portfolio page data.
 * NEW CONSOLIDATED ENDPOINT: /api/page/profile
 */
export async function getProfilePage(useCache: boolean = false) {
  const path = "/api/page/profile";
  return get(path, { cache: useCache, ttl: 5000 });
}

/**
 * Get dashboard page data.
 * NEW CONSOLIDATED ENDPOINT: /api/page/dashboard
 */
export async function getDashboardPage(useCache: boolean = false) {
  const path = "/api/page/dashboard";
  return get(path, { cache: useCache, ttl: 5000 });
}

/**
 * Get predict page data.
 * NEW CONSOLIDATED ENDPOINT: /api/page/predict
 */
export async function getPredictPage(useCache: boolean = false) {
  const path = "/api/page/predict";
  return get(path, { cache: useCache, ttl: 5000 });
}

/**
 * Get tools/admin page data.
 * NEW CONSOLIDATED ENDPOINT: /api/page/tools
 */
export async function getToolsPage(useCache: boolean = false) {
  const path = "/api/page/tools";
  return get(path, { cache: useCache, ttl: 5000 });
}

// ========== Admin Consolidated API ==========

/**
 * Get admin system status.
 * NEW CONSOLIDATED ENDPOINT: /api/admin/status
 */
export async function getAdminStatus() {
  return get("/api/admin/status");
}

/**
 * Get admin logs.
 * NEW CONSOLIDATED ENDPOINT: /api/admin/logs
 */
export async function getAdminLogs(logType: string = "backend", lines: number = 100) {
  return get(`/api/admin/logs?log_type=${logType}&lines=${lines}`);
}

/**
 * Execute admin action.
 * NEW CONSOLIDATED ENDPOINT: /api/admin/action/{action}
 */
export async function executeAdminAction(action: string) {
  return post(`/api/admin/action/${action}`);
}

/**
 * Get replay status for a backend.
 * NEW CONSOLIDATED ENDPOINT: /api/admin/replay/{backend}/status
 */
export async function getReplayStatus(backendType: string) {
  return get(`/api/admin/replay/${backendType}/status`);
}

/**
 * Get admin metrics.
 * NEW CONSOLIDATED ENDPOINT: /api/admin/metrics
 */
export async function getAdminMetrics() {
  return get("/api/admin/metrics");
}

// ========== Optimizer API ==========

/**
 * Run portfolio optimization with specified parameters
 */
export async function runPortfolioOptimizer(params: any): Promise<any> {
  return post("/api/optimizer/run", params);
}

/**
 * Get efficient frontier data for visualization
 */
export async function getEfficientFrontier(): Promise<any> {
  return get("/api/optimizer/frontier");
}

// ========== Reports API ==========

/**
 * List all available reports
 */
export async function listReports(): Promise<any[]> {
  return get("/api/reports/list");
}

/**
 * Get detailed information about a specific report
 */
export async function getReportDetails(reportId: string): Promise<any> {
  return get(`/api/reports/${reportId}`);
}

/**
 * Download a report in the specified format
 */
export async function downloadReport(
  reportId: string,
  format: "pdf" | "csv" | "json"
): Promise<Blob> {
  const base = getApiBaseUrl();
  const url = `${base}/api/reports/${reportId}/download?format=${format}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download report: ${res.status} ${res.statusText}`);
  }
  return res.blob();
}

/**
 * Generate a new report with specified parameters
 */
export async function generateReport(params: any): Promise<any> {
  return post("/api/reports/generate", params);
}

// ========== Model Registry API ==========

/**
 * List all models in the registry
 */
export async function listModels(): Promise<any[]> {
  return get("/api/models/list");
}

/**
 * Get detailed information about a specific model
 */
export async function getModelDetails(modelId: string): Promise<any> {
  return get(`/api/models/${modelId}`);
}

/**
 * Get performance metrics for a model
 */
export async function getModelPerformance(modelId: string): Promise<any> {
  return get(`/api/models/${modelId}/performance`);
}

/**
 * Upload/create a new model
 */
export async function uploadModel(modelData: any): Promise<any> {
  const formData = new FormData();
  formData.append("file", modelData.file);
  formData.append("name", modelData.name);
  formData.append("type", modelData.type);
  if (modelData.description) {
    formData.append("description", modelData.description);
  }
  if (modelData.hyperparameters) {
    formData.append("hyperparameters", JSON.stringify(modelData.hyperparameters));
  }
  
  const base = getApiBaseUrl();
  const url = `${base}/api/models/upload`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  
  if (!res.ok) {
    throw new Error(`Failed to upload model: ${res.status} ${res.statusText}`);
  }
  
  return res.json();
}

/**
 * Get model training history
 */
export async function getModelTrainingHistory(modelId: string): Promise<any[]> {
  return get(`/api/models/${modelId}/history`);
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

  // Page Data (NEW CONSOLIDATED)
  pages: {
    bots: getBotsPage,
    profile: getProfilePage,
    dashboard: getDashboardPage,
    predict: getPredictPage,
    tools: getToolsPage,
  },

  // Admin (NEW CONSOLIDATED)
  admin: {
    status: getAdminStatus,
    logs: getAdminLogs,
    action: executeAdminAction,
    replayStatus: getReplayStatus,
    metrics: getAdminMetrics,
  },

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

  // Optimizer
  optimizer: {
    run: runPortfolioOptimizer,
    getFrontier: getEfficientFrontier,
  },

  // Reports
  reports: {
    list: listReports,
    getDetails: getReportDetails,
    download: downloadReport,
    generate: generateReport,
  },

  // Model Registry
  models: {
    list: listModels,
    getDetails: getModelDetails,
    getPerformance: getModelPerformance,
    upload: uploadModel,
    getHistory: getModelTrainingHistory,
  },

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

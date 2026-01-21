// /frontend/lib/dtApi.ts
// Centralized API client for DT (day-trading/intraday) backend
//
// Architecture:
//  - Uses Next.js API proxy route (/api/dt/*)
//  - Proxy route forwards to DT backend (port 8010) avoiding CORS
//  - Mirrors the structure of main API client (api.ts)
//  - Provides type-safe functions for all DT backend endpoints
//
// ===== ENDPOINT MAPPINGS =====
//
// Frontend Call                          → Proxy Route              → DT Backend Endpoint
// ----------------------------------------------------------------------------------------------------
// /api/dt/health                        → /health                 → GET /health
// /api/dt/health/ready                  → /health/ready           → GET /health/ready
// /api/dt/health/live                   → /health/live            → GET /health/live
// /api/dt/dt/learning/metrics           → /dt/learning/metrics    → GET /dt/learning/metrics
// /api/dt/replay/start                  → /replay/start           → POST /replay/start
// /api/dt/replay/status                 → /replay/status          → GET /replay/status
// /api/dt/jobs/cycle                    → /jobs/cycle             → POST /jobs/cycle
// /api/dt/jobs/status                   → /jobs/status            → GET /jobs/status
// /api/dt/data/rolling                  → /data/rolling           → GET /data/rolling
// /api/dt/data/positions                → /data/positions         → GET /data/positions
// /api/dt/data/metrics                  → /data/metrics           → GET /data/metrics
//
// Note: The Next.js proxy in /app/api/dt/[...path]/route.ts forwards paths as-is,
//       without adding any prefix. DT backend uses different routing conventions.

import { getDtApiBaseUrl } from "./api";
import type {
  HealthResponse,
  ReadyResponse,
  LiveResponse,
  LearningMetrics,
  ReplayResponse,
  ReplayStatusResponse,
  JobResponse,
  JobStatusResponse,
  RollingDataResponse,
  PositionsResponse,
  MetricsResponse,
} from "./dtTypes";

/**
 * Make a request to the DT backend API.
 * Automatically uses the DT proxy route.
 */
async function request(path: string, init: RequestInit = {}) {
  const base = getDtApiBaseUrl();
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
 * GET request to DT backend.
 */
async function get(path: string) {
  return request(path, { method: "GET" });
}

/**
 * POST request to DT backend with optional body.
 */
async function post(path: string, body?: any) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

// ---- Health Check API ----

/**
 * Get DT backend health status.
 */
export async function getDtHealth(): Promise<HealthResponse> {
  return get("/health");
}

/**
 * Get DT backend readiness status.
 */
export async function getDtHealthReady(): Promise<ReadyResponse> {
  return get("/health/ready");
}

/**
 * Get DT backend liveness status.
 */
export async function getDtHealthLive(): Promise<LiveResponse> {
  return get("/health/live");
}

// ---- Learning Metrics API ----

/**
 * Get DT learning metrics (performance, model health, missed opportunities).
 * Used by the DT Learning dashboard page.
 */
export async function getDtLearningMetrics(): Promise<LearningMetrics> {
  return get("/dt/learning/metrics");
}

// ---- Replay API ----

/**
 * Start DT replay for the specified number of weeks.
 * @param weeks - Number of weeks to replay (default: 4)
 */
export async function startDtReplay(weeks: number = 4): Promise<ReplayResponse> {
  return post("/replay/start", { weeks });
}

/**
 * Get DT replay status and progress.
 */
export async function getDtReplayStatus(): Promise<ReplayStatusResponse> {
  return get("/replay/status");
}

// ---- Job Management API ----

/**
 * Trigger a DT trading cycle job.
 */
export async function triggerDtCycle(): Promise<JobResponse> {
  return post("/jobs/cycle");
}

/**
 * Get DT job status.
 */
export async function getDtJobStatus(): Promise<JobStatusResponse> {
  return get("/jobs/status");
}

// ---- Data Access API ----

/**
 * Get DT rolling cache data.
 */
export async function getDtRollingData(): Promise<RollingDataResponse> {
  return get("/data/rolling");
}

/**
 * Get current DT positions.
 */
export async function getDtPositions(): Promise<PositionsResponse> {
  return get("/data/positions");
}

/**
 * Get DT metrics.
 */
export async function getDtMetrics(): Promise<MetricsResponse> {
  return get("/data/metrics");
}

// ---- Utility Functions ----

/**
 * Get the base URL for DT backend API calls.
 * Useful for components that need to construct custom URLs.
 */
export function getDtBaseUrl(): string {
  return getDtApiBaseUrl();
}

// ---- Unified DT API Object ----

/**
 * Unified DT API object for organized access to all DT endpoints.
 * Usage: import { dtApi } from "@/lib/dtApi"; dtApi.health.getStatus();
 */
export const dtApi = {
  // Health checks
  health: {
    getStatus: getDtHealth,
    getReady: getDtHealthReady,
    getLive: getDtHealthLive,
  },

  // Learning
  learning: {
    getMetrics: getDtLearningMetrics,
  },

  // Replay
  replay: {
    start: startDtReplay,
    getStatus: getDtReplayStatus,
  },

  // Jobs
  jobs: {
    triggerCycle: triggerDtCycle,
    getStatus: getDtJobStatus,
  },

  // Data
  data: {
    getRolling: getDtRollingData,
    getPositions: getDtPositions,
    getMetrics: getDtMetrics,
  },

  // Utility
  getBaseUrl: getDtBaseUrl,
};

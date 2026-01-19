/**
 * Centralized Bots API Client
 * 
 * This module provides focused functions for interacting with the bots API.
 * All requests go through Next.js proxy routes to avoid CORS and localhost issues.
 * 
 * API Flow:
 * 1. Frontend calls functions in this module
 * 2. Functions make relative fetch calls to /api/backend/*
 * 3. Next.js proxy route forwards to actual backend
 * 4. Response is returned to frontend
 * 
 * This ensures:
 * - No hardcoded localhost URLs
 * - No CORS errors
 * - Works with remote backends
 * - Centralized error handling
 */

import type { 
  BotsPageBundle, 
  EodBotConfig, 
  BotDraft,
  UpdateBotConfigRequest,
  UpdateBotConfigResponse 
} from "./botsTypes";

// ============================================
// Helper Functions
// ============================================

/**
 * Add cache-busting timestamp to URL
 */
function withBust(url: string): string {
  const bust = `_ts=${Date.now()}`;
  return url.includes("?") ? `${url}&${bust}` : `${url}?${bust}`;
}

/**
 * Generic GET request through proxy
 */
async function apiGet<T>(url: string, options?: { bustCache?: boolean }): Promise<T> {
  const finalUrl = options?.bustCache ? withBust(url) : url;
  const response = await fetch(finalUrl, { 
    method: "GET", 
    cache: "no-store",
  });
  
  if (!response.ok) {
    throw new Error(`GET ${url} failed: ${response.status} ${response.statusText}`);
  }
  
  return await response.json() as T;
}

/**
 * Generic POST request through proxy
 */
async function apiPostJson<T>(url: string, body: any): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  
  if (!response.ok) {
    throw new Error(`POST ${url} failed: ${response.status} ${response.statusText}`);
  }
  
  return await response.json() as T;
}

/**
 * Try multiple URLs in order, return first success
 */
async function tryGetFirst<T>(urls: string[]): Promise<{ url: string; data: T } | null> {
  for (const url of urls) {
    try {
      const data = await apiGet<T>(url);
      return { url, data };
    } catch {
      // Continue to next URL
    }
  }
  return null;
}

// ============================================
// Bots Page Data
// ============================================

/**
 * Get combined swing + intraday bot data
 * 
 * Tries multiple endpoints for compatibility:
 * 1. /api/backend/bots/page (proxied to backend)
 * 2. /api/bots/page (fallback)
 * 
 * @throws Error if no endpoint responds successfully
 */
export async function getBotsPageBundle(): Promise<BotsPageBundle> {
  const result = await tryGetFirst<BotsPageBundle>([
    "/api/backend/bots/page",
    "/api/bots/page",
  ]);
  
  if (!result) {
    throw new Error("Bots bundle endpoint not found (check backend router mount).");
  }
  
  return result.data;
}

/**
 * Determine which API prefix to use based on successful endpoint
 * 
 * This function tries to fetch from the bots page endpoint and returns
 * the prefix that worked. This is useful for save operations.
 * 
 * @returns "/api/backend" or "/api" depending on which works
 */
export async function detectApiPrefix(): Promise<string> {
  const result = await tryGetFirst<BotsPageBundle>([
    "/api/backend/bots/page",
    "/api/bots/page",
  ]);
  
  if (!result) {
    // Default to /api/backend if detection fails
    return "/api/backend";
  }
  
  return result.url.startsWith("/api/backend") ? "/api/backend" : "/api";
}

// ============================================
// Bot Configuration Updates
// ============================================

/**
 * Update EOD (swing) bot configuration
 * 
 * @param botKey - Bot identifier (e.g., "swing_bot_1")
 * @param config - Bot configuration object
 * @param apiPrefix - API prefix to use ("/api/backend" or "/api")
 */
export async function updateEodBotConfig(
  botKey: string,
  config: EodBotConfig | BotDraft,
  apiPrefix: string = "/api/backend"
): Promise<UpdateBotConfigResponse> {
  return apiPostJson<UpdateBotConfigResponse>(
    `${apiPrefix}/eod/configs`,
    { bot_key: botKey, config }
  );
}

/**
 * Update intraday (day trading) bot configuration
 * 
 * @param botKey - Bot identifier (e.g., "intraday_engine")
 * @param config - Bot configuration object
 * @param apiPrefix - API prefix to use ("/api/backend" or "/api")
 */
export async function updateIntradayBotConfig(
  botKey: string,
  config: EodBotConfig | BotDraft,
  apiPrefix: string = "/api/backend"
): Promise<UpdateBotConfigResponse> {
  return apiPostJson<UpdateBotConfigResponse>(
    `${apiPrefix}/intraday/configs`,
    { bot_key: botKey, config }
  );
}

// ============================================
// Server-Sent Events (SSE)
// ============================================

/**
 * Get the SSE endpoint URL for real-time bot updates
 * 
 * The SSE stream provides real-time updates for both swing and intraday bots.
 * Updates are pushed approximately every 5 seconds.
 * 
 * @returns SSE endpoint URL (relative path through proxy)
 */
export function getBotsSSEUrl(): string {
  return "/api/backend/events/bots";
}

// ============================================
// Cache Keys
// ============================================

/**
 * Cache keys for client-side caching (used with SSE invalidation)
 */
export const BOTS_CACHE_KEYS = {
  PAGE_BACKEND: "api:/api/backend/bots/page",
  PAGE_DIRECT: "api:/api/bots/page",
} as const;

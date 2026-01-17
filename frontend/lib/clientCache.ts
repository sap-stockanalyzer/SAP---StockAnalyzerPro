/**
 * Client-side cache with TTL (Time To Live) support
 * 
 * Implements browser localStorage-based caching with automatic expiration.
 * Cache entries are automatically invalidated when:
 * - TTL expires
 * - SSE push event with matching key is received
 * - Manual invalidation is called
 * 
 * Architecture:
 * - Backend has NO caching (returns fresh data always)
 * - Frontend caches responses in localStorage
 * - SSE pushes invalidate cache (force fresh fetch)
 * - TTL ensures cache doesn't go stale
 */

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number; // milliseconds
}

interface CacheOptions {
  ttl?: number; // Time to live in milliseconds (default: 5000ms = 5s)
  key: string; // Cache key
}

const CACHE_PREFIX = "aion_cache_";
const DEFAULT_TTL = 5000; // 5 seconds

/**
 * Get cached data if it exists and is not expired
 */
export function getCached<T>(key: string): T | null {
  if (typeof window === "undefined") {
    return null; // SSR - no localStorage
  }

  try {
    const cacheKey = CACHE_PREFIX + key;
    const cached = localStorage.getItem(cacheKey);
    
    if (!cached) {
      return null;
    }

    const entry: CacheEntry<T> = JSON.parse(cached);
    const now = Date.now();
    const age = now - entry.timestamp;

    // Check if expired
    if (age > entry.ttl) {
      localStorage.removeItem(cacheKey);
      return null;
    }

    return entry.data;
  } catch (error) {
    console.warn(`[clientCache] Failed to get cache for ${key}:`, error);
    return null;
  }
}

/**
 * Store data in cache with TTL
 */
export function setCached<T>(
  key: string,
  data: T,
  options: { ttl?: number } = {}
): void {
  if (typeof window === "undefined") {
    return; // SSR - no localStorage
  }

  try {
    const cacheKey = CACHE_PREFIX + key;
    const ttl = options.ttl ?? DEFAULT_TTL;

    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      ttl,
    };

    localStorage.setItem(cacheKey, JSON.stringify(entry));
  } catch (error) {
    console.warn(`[clientCache] Failed to set cache for ${key}:`, error);
    // Quota exceeded or other error - fail silently
  }
}

/**
 * Invalidate (remove) cached data
 */
export function invalidateCache(key: string): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    const cacheKey = CACHE_PREFIX + key;
    localStorage.removeItem(cacheKey);
  } catch (error) {
    console.warn(`[clientCache] Failed to invalidate cache for ${key}:`, error);
  }
}

/**
 * Clear all AION cache entries
 */
export function clearAllCache(): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    const keys = Object.keys(localStorage);
    keys.forEach((key) => {
      if (key.startsWith(CACHE_PREFIX)) {
        localStorage.removeItem(key);
      }
    });
  } catch (error) {
    console.warn("[clientCache] Failed to clear all cache:", error);
  }
}

/**
 * Fetch with cache support
 * 
 * Usage:
 *   const data = await fetchWithCache("/api/bots/page", { ttl: 5000 });
 *   const data = await fetchWithCache("/api/bots/page", { key: "custom_key", ttl: 5000 });
 * 
 * - First checks cache
 * - If cache miss or expired, fetches from API
 * - Stores response in cache with TTL
 * - Returns data
 */
export async function fetchWithCache<T>(
  url: string,
  options: Partial<CacheOptions> & RequestInit = {}
): Promise<T> {
  const { key, ttl, ...fetchOptions } = options;
  const cacheKey = key || url; // Default to URL if no key provided

  // Try cache first
  const cached = getCached<T>(cacheKey);
  if (cached !== null) {
    return cached;
  }

  // Cache miss - fetch from API
  const response = await fetch(url, {
    ...fetchOptions,
    cache: "no-store", // Disable browser cache
  });

  if (!response.ok) {
    throw new Error(
      `Fetch failed for ${url}: ${response.status} ${response.statusText}`
    );
  }

  const data: T = await response.json();

  // Store in cache
  setCached(cacheKey, data, { ttl });

  return data;
}

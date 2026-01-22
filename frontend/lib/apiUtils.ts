/**
 * Shared API utility functions
 * 
 * Common helpers for API calls across components
 */

/**
 * Try multiple URLs in parallel, return first success
 * 
 * Instead of sequential execution (timeout per URL), this:
 * 1. Races all URLs in parallel with configurable timeout
 * 2. Returns the first successful response
 * 3. Fails fast if no URL responds within timeout
 * 4. Properly cleans up all pending requests using AbortController
 * 
 * @param urls - Array of URLs to try
 * @param timeoutMs - Timeout per request in milliseconds (default: 3000)
 */
export async function tryGetFirst<T>(
  urls: string[], 
  timeoutMs: number = 3000
): Promise<{ url: string; data: T } | null> {
  if (urls.length === 0) return null;

  const controllers: AbortController[] = [];

  const promises = urls.map(async (url) => {
    const controller = new AbortController();
    controllers.push(controller);
    
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, timeoutMs);

    try {
      const response = await fetch(url, { 
        method: "GET", 
        cache: "no-store",
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`GET ${url} failed: ${response.status}`);
      }
      
      const data = await response.json() as T;
      return { url, data };
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  });

  try {
    const result = await Promise.any(promises);
    controllers.forEach(c => c.abort());
    return result;
  } catch (error) {
    controllers.forEach(c => c.abort());
    return null;
  }
}

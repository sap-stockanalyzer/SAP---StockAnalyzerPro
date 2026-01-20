# API Integration Testing Guide

This document provides manual testing steps to verify the frontend-backend API integration fixes.

## Changes Made

### 1. Fixed `tryGetFirst()` Function
- **Location**: `/frontend/lib/botsApi.ts`
- **Change**: Replaced sequential execution with parallel requests using `Promise.any()`
- **Benefit**: Reduces timeout from 48s to 3s when first endpoint fails

### 2. Fixed Profile Page API Endpoint
- **Location**: `/frontend/app/profile/page.tsx`
- **Change**: Fixed double `/api/api` prefix → `/api/backend/cache/unified`
- **Benefit**: Profile page now fetches data correctly

## Manual Testing Steps

### Test 1: Bots Page Fast Load (No 48s Delay)

**Test the `tryGetFirst()` fix:**

1. **Setup**: Ensure backend is running but first endpoint `/api/backend/bots/page` returns 404
   ```bash
   # Temporarily disable the endpoint or use network throttling
   ```

2. **Test**:
   - Navigate to `/bots` page in browser
   - Open browser DevTools (F12) → Network tab
   - Refresh the page
   - Observe the timing

3. **Expected Result**:
   - ✅ Page loads within 3-5 seconds (not 48 seconds)
   - ✅ Request to fallback endpoint `/api/bots/page` succeeds
   - ✅ No console errors related to timeouts

4. **Failure Indicators**:
   - ❌ Page takes 48+ seconds to load
   - ❌ Multiple sequential requests with long delays
   - ❌ Browser shows "waiting for response" for extended period

### Test 2: Profile Page Loads Correctly

**Test the unified cache endpoint fix:**

1. **Test**:
   - Navigate to `/profile` page in browser
   - Open browser DevTools (F12) → Network tab
   - Refresh the page
   - Check the API request

2. **Expected Result**:
   - ✅ Request to `/api/backend/cache/unified` succeeds (200 OK)
   - ✅ Profile page displays portfolio data, equity curve, and holdings
   - ✅ No 404 errors in console

3. **Failure Indicators**:
   - ❌ Request to `/api/backend/api/cache/unified` (double api prefix)
   - ❌ 404 Not Found error
   - ❌ Profile page shows "Unable to load portfolio" error message

### Test 3: Portfolio Page Loads Correctly

**Verify portfolio page API calls:**

1. **Test**:
   - Navigate to `/portfolio` page in browser
   - Open browser DevTools (F12) → Network tab
   - Refresh the page

2. **Expected Result**:
   - ✅ Request to `/api/backend/bots/page` succeeds (200 OK)
   - ✅ Portfolio page displays total equity, active bots, and performance
   - ✅ Individual bot cards show equity values

3. **Failure Indicators**:
   - ❌ 404 or 502 errors in console
   - ❌ "Failed to fetch" error message
   - ❌ Empty portfolio data

### Test 4: Parallel Request Behavior

**Verify parallel execution of tryGetFirst():**

1. **Test**:
   - Navigate to `/bots` page
   - Open browser DevTools (F12) → Network tab
   - Filter by "Fetch/XHR"
   - Refresh the page
   - Observe the waterfall chart

2. **Expected Result**:
   - ✅ Requests to both `/api/backend/bots/page` and `/api/bots/page` start simultaneously
   - ✅ Requests run in parallel (overlapping bars in waterfall)
   - ✅ Page displays data from whichever endpoint responds first

3. **Failure Indicators**:
   - ❌ Requests run sequentially (one after another in waterfall)
   - ❌ Long gap between request start times
   - ❌ Second request only starts after first completes

### Test 5: Fast Failure on 404

**Verify timeout works correctly:**

1. **Setup**: Temporarily configure backend to return 404 for all endpoints
   ```bash
   # Or use browser DevTools to block requests
   ```

2. **Test**:
   - Navigate to `/bots` page
   - Measure time to error message

3. **Expected Result**:
   - ✅ Error message appears within 3-5 seconds
   - ✅ "Bots bundle endpoint not found" error displayed
   - ✅ No 48+ second hang

4. **Failure Indicators**:
   - ❌ Page hangs for 48+ seconds
   - ❌ Browser becomes unresponsive
   - ❌ Multiple timeout messages in console

### Test 6: SSE Real-Time Updates

**Verify SSE endpoint is accessible:**

1. **Test**:
   - Navigate to `/bots` page
   - Enable "Live" mode (toggle switch)
   - Open browser DevTools (F12) → Network tab
   - Look for EventSource connection

2. **Expected Result**:
   - ✅ Connection to `/api/backend/events/bots` (status: pending/101)
   - ✅ Bot data auto-updates every 5 seconds
   - ✅ No SSE connection errors

3. **Failure Indicators**:
   - ❌ No EventSource connection in Network tab
   - ❌ 404 or connection errors
   - ❌ Bot data doesn't auto-update

### Test 7: All API Endpoints Work

**Comprehensive endpoint verification:**

1. **Test Pages**:
   - `/bots` - Bots management page
   - `/portfolio` - Portfolio overview
   - `/profile` - Profile and holdings
   - `/insights` - Insights and predictions
   - `/bots/config` - Bot configuration
   - `/tools/api-keys` - API keys settings

2. **Expected Result**:
   - ✅ All pages load without errors
   - ✅ No 404 errors in console
   - ✅ No 502 Bad Gateway errors
   - ✅ Data displays correctly on all pages

3. **Failure Indicators**:
   - ❌ Any page shows error message
   - ❌ Console shows API errors
   - ❌ Empty or missing data

## API Endpoint Reference

### Main Backend Endpoints (via `/api/backend/*`)

| Frontend Path | Proxy Forwards To | Backend Route |
|--------------|-------------------|---------------|
| `/api/backend/bots/page` | `/api/bots/page` | `GET /api/bots/page` |
| `/api/backend/bots/overview` | `/api/bots/overview` | `GET /api/bots/overview` |
| `/api/backend/eod/status` | `/api/eod/status` | `GET /api/eod/status` |
| `/api/backend/eod/configs` | `/api/eod/configs` | `GET /api/eod/configs` |
| `/api/backend/intraday/configs` | `/api/intraday/configs` | `POST /api/intraday/configs` |
| `/api/backend/cache/unified` | `/api/cache/unified` | `GET /api/cache/unified` |
| `/api/backend/events/bots` | `/api/events/bots` | `GET /api/events/bots` (SSE) |
| `/api/backend/insights/predictions/latest` | `/api/insights/predictions/latest` | `GET /api/insights/predictions/latest` |
| `/api/backend/portfolio/holdings/top/1w` | `/api/portfolio/holdings/top/1w` | `GET /api/portfolio/holdings/top/1w` |
| `/api/backend/settings/keys` | `/api/settings/keys` | `GET /api/settings/keys` |
| `/api/backend/settings/knobs` | `/api/settings/knobs` | `GET /api/settings/knobs` |

### Special Routes (No /api prefix added)

| Frontend Path | Proxy Forwards To | Backend Route |
|--------------|-------------------|---------------|
| `/api/backend/dashboard/*` | `/dashboard/*` | `/dashboard/*` |
| `/api/backend/admin/*` | `/admin/*` | `/admin/*` |

### DT Backend Endpoints (via `/api/dt/*`)

| Frontend Path | Proxy Forwards To | DT Backend Route |
|--------------|-------------------|------------------|
| `/api/dt/health` | `/health` | `GET /health` |
| `/api/dt/jobs/status` | `/jobs/status` | `GET /jobs/status` |
| `/api/dt/data/positions` | `/data/positions` | `GET /data/positions` |

## Troubleshooting

### Issue: Page hangs for 48+ seconds
- **Cause**: `tryGetFirst()` is still using sequential execution
- **Fix**: Ensure changes in `/frontend/lib/botsApi.ts` are deployed
- **Verify**: Check browser source maps or add `console.log()` in `tryGetFirst()`

### Issue: Profile page shows 404 error
- **Cause**: Double `/api/api` prefix in unified cache call
- **Fix**: Ensure `/frontend/app/profile/page.tsx` uses `/api/backend/cache/unified`
- **Verify**: Check Network tab for actual request URL

### Issue: Bots page shows "endpoint not found" error
- **Cause**: Backend route not mounted or proxy misconfigured
- **Fix**: Verify backend router is included in `backend_service.py`
- **Verify**: Test backend endpoint directly: `curl http://localhost:8000/api/bots/page`

### Issue: SSE connection fails
- **Cause**: SSE endpoint not accessible or CORS issue
- **Fix**: Check backend events router is mounted
- **Verify**: Check browser console for SSE connection errors

## Success Criteria

All tests pass with:
- ✅ Bots page loads in < 5 seconds (not 48+ seconds)
- ✅ All pages load without 404 errors
- ✅ Profile page displays portfolio data
- ✅ Portfolio page displays bot equity
- ✅ SSE connections work for real-time updates
- ✅ Parallel requests execute simultaneously
- ✅ Fast failure on unavailable endpoints (< 3s)
- ✅ No console errors in browser DevTools
- ✅ No 502 Bad Gateway errors

## Automated Testing (Future)

Consider adding automated tests:

```typescript
// Example Jest test for tryGetFirst()
describe('tryGetFirst', () => {
  it('should try URLs in parallel', async () => {
    const urls = ['/api/backend/bots/page', '/api/bots/page'];
    const startTime = Date.now();
    await tryGetFirst(urls);
    const elapsed = Date.now() - startTime;
    expect(elapsed).toBeLessThan(5000); // Should complete in < 5s
  });

  it('should return first successful response', async () => {
    const result = await tryGetFirst(['/404-endpoint', '/api/bots/page']);
    expect(result).not.toBeNull();
    expect(result?.url).toBe('/api/bots/page');
  });

  it('should timeout after 3 seconds per request', async () => {
    const startTime = Date.now();
    const result = await tryGetFirst(['/slow-endpoint']);
    const elapsed = Date.now() - startTime;
    expect(elapsed).toBeLessThan(4000);
    expect(result).toBeNull();
  });
});
```

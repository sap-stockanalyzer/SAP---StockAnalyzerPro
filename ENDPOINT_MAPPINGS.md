# Frontend-Backend API Endpoint Mappings

This document provides a comprehensive mapping of all frontend API calls to their corresponding backend routes.

## Overview

The AION Analytics application uses a Next.js frontend with proxy routes to communicate with two backend services:

1. **Main Backend** (port 8000) - Handles swing/EOD bots, portfolio, settings, etc.
2. **DT Backend** (port 8010) - Handles day-trading/intraday operations

All frontend requests go through Next.js proxy routes (`/api/backend/*` and `/api/dt/*`) to avoid CORS issues and centralize error handling.

## Proxy Route Behavior

### Main Backend Proxy (`/app/api/backend/[...path]/route.ts`)

**Routing Logic:**
- Dashboard routes (`/dashboard/*`) → Forward as-is (NO /api prefix)
- Admin routes (`/admin/*`) → Forward as-is (NO /api prefix)
- Routes starting with `/api/*` → Forward as-is (NO additional /api prefix)
- All other routes → Prepend `/api` prefix

**Examples:**
```
Frontend: /api/backend/bots/page     → Backend: /api/bots/page
Frontend: /api/backend/dashboard     → Backend: /dashboard (no /api)
Frontend: /api/backend/admin/login   → Backend: /admin/login (no /api)
Frontend: /api/backend/api/test      → Backend: /api/test (no duplicate)
```

### DT Backend Proxy (`/app/api/dt/[...path]/route.ts`)

**Routing Logic:**
- Forward path as-is (NO prefix manipulation)

**Examples:**
```
Frontend: /api/dt/health             → DT Backend: /health
Frontend: /api/dt/jobs/status        → DT Backend: /jobs/status
Frontend: /api/dt/data/positions     → DT Backend: /data/positions
```

## Complete Endpoint Mappings

### Bots Management

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/bots/page` | `/api/bots/page` | `GET /api/bots/page` | GET | Combined swing + intraday bot data bundle |
| `/api/backend/bots/overview` | `/api/bots/overview` | `GET /api/bots/overview` | GET | Alias for bots page (same data) |
| `/api/backend/eod/status` | `/api/eod/status` | `GET /api/eod/status` | GET | EOD (swing) bot status |
| `/api/backend/eod/configs` | `/api/eod/configs` | `GET /api/eod/configs` | GET | List EOD bot configurations |
| `/api/backend/eod/configs` | `/api/eod/configs` | `POST /api/eod/configs` | POST | Update EOD bot configuration |
| `/api/backend/intraday/status` | `/api/intraday/status` | `GET /api/intraday/status` | GET | Intraday bot status |
| `/api/backend/intraday/configs` | `/api/intraday/configs` | `GET /api/intraday/configs` | GET | List intraday bot configurations |
| `/api/backend/intraday/configs` | `/api/intraday/configs` | `POST /api/intraday/configs` | POST | Update intraday bot configuration |

**Router:** `backend/routers/bots_page_router.py` (prefix: `/api/bots`)
**Router:** `backend/routers/eod_bots_router.py` (prefix: `/api/eod`)
**Router:** `backend/routers/intraday_logs_router.py` (prefix: `/api/intraday`)

### Server-Sent Events (SSE)

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/events/bots` | `/api/events/bots` | `GET /api/events/bots` | GET | Real-time bot updates stream |

**Router:** `backend/routers/events_router.py` (prefix: `/api/events`)
**Update Frequency:** ~5 seconds

### Portfolio & Holdings

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/portfolio/holdings/top/1w` | `/api/portfolio/holdings/top/1w` | `GET /api/portfolio/holdings/top/1w` | GET | Top holdings over 1 week |
| `/api/backend/portfolio/holdings/top/1m` | `/api/portfolio/holdings/top/1m` | `GET /api/portfolio/holdings/top/1m` | GET | Top holdings over 1 month |
| `/api/backend/portfolio/holdings/top/3m` | `/api/portfolio/holdings/top/3m` | `GET /api/portfolio/holdings/top/3m` | GET | Top holdings over 3 months |

**Router:** `backend/routers/portfolio_router.py` (prefix: `/api/portfolio`)

### Unified Cache

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/cache/unified` | `/api/cache/unified` | `GET /api/cache/unified` | GET | Single endpoint for all frontend data |
| `/api/backend/cache/unified/refresh` | `/api/cache/unified/refresh` | `POST /api/cache/unified/refresh` | POST | Manually trigger cache refresh |
| `/api/backend/cache/unified/age` | `/api/cache/unified/age` | `GET /api/cache/unified/age` | GET | Get cache age in seconds |

**Router:** `backend/routers/unified_cache_router.py` (prefix: `/api/cache`)
**Update Frequency:** Auto-updated every 5 seconds by background job

### Insights & Predictions

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/insights/predictions/latest` | `/api/insights/predictions/latest` | `GET /api/insights/predictions/latest` | GET | Latest prediction feed with targets |

**Router:** `backend/routers/insights_router.py` (prefix: `/api/insights`)

### Settings & Configuration

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/settings/keys` | `/api/settings/keys` | `GET /api/settings/keys` | GET | Get API keys and settings from .env |
| `/api/backend/settings/update` | `/api/settings/update` | `POST /api/settings/update` | POST | Update API keys / settings in .env |
| `/api/backend/settings/knobs` | `/api/settings/knobs` | `GET /api/settings/knobs` | GET | Get knobs.env file content (EOD config) |
| `/api/backend/settings/knobs` | `/api/settings/knobs` | `POST /api/settings/knobs` | POST | Save knobs.env file content |
| `/api/backend/settings/dt-knobs` | `/api/settings/dt-knobs` | `GET /api/settings/dt-knobs` | GET | Get dt_knobs.env file content (DT config) |
| `/api/backend/settings/dt-knobs` | `/api/settings/dt-knobs` | `POST /api/settings/dt-knobs` | POST | Save dt_knobs.env file content |
| `/api/backend/settings/status` | `/api/settings/status` | `GET /api/settings/status` | GET | Get settings status and validation |
| `/api/backend/settings/test` | `/api/settings/test` | `POST /api/settings/test` | POST | Test API keys validity |

**Router:** `backend/routers/settings_router.py` (prefix: `/api/settings`)

### Metrics & System Status

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/metrics/overview` | `/api/metrics/overview` | `GET /api/metrics/overview` | GET | System metrics overview |
| `/api/backend/system/status` | `/api/system/status` | `GET /api/system/status` | GET | System status and health |
| `/api/backend/system/run/{task}` | `/api/system/run/{task}` | `POST /api/system/run/{task}` | POST | Run system task |

**Router:** `backend/routers/metrics_router.py` (prefix: `/api/metrics`)
**Router:** `backend/routers/system_status_router.py` (prefix: `/api/system`)

### Dashboard

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/dashboard/metrics` | `/dashboard/metrics` | `GET /dashboard/metrics` | GET | Dashboard metrics (no /api prefix) |

**Router:** `backend/routers/dashboard_router.py` (prefix: `/dashboard`)
**Note:** Dashboard routes do NOT have `/api` prefix on backend

### Admin

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/admin/login` | `/admin/login` | `POST /admin/login` | POST | Admin authentication |
| `/api/backend/admin/replay/status` | `/admin/replay/status` | `GET /admin/replay/status` | GET | Replay system status |
| `/api/backend/admin/replay/start` | `/admin/replay/start` | `POST /admin/replay/start` | POST | Start historical replay |
| `/api/backend/admin/tools/logs` | `/admin/tools/logs` | `GET /admin/tools/logs` | GET | Get system logs |
| `/api/backend/admin/tools/clear-locks` | `/admin/tools/clear-locks` | `POST /admin/tools/clear-locks` | POST | Clear system locks |
| `/api/backend/admin/tools/git-pull` | `/admin/tools/git-pull` | `POST /admin/tools/git-pull` | POST | Pull latest code from git |
| `/api/backend/admin/system/restart` | `/admin/system/restart` | `POST /admin/system/restart` | POST | Restart system |
| `/api/backend/admin/tools/refresh-universes` | `/admin/tools/refresh-universes` | `POST /admin/tools/refresh-universes` | POST | Refresh trading universes |

**Router:** `backend/admin/routes.py` (prefix: `/admin`)
**Note:** Admin routes do NOT have `/api` prefix on backend

### Logs

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/logs/nightly/runs` | `/api/logs/nightly/runs` | `GET /api/logs/nightly/runs` | GET | List nightly log runs |
| `/api/backend/logs/nightly/{day}` | `/api/logs/nightly/{day}` | `GET /api/logs/nightly/{day}` | GET | Get nightly log for specific day |

**Router:** `backend/routers/nightly_logs_router.py` (prefix: `/api/logs`)

### Model Registry

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/models/list` | `/api/models/list` | `GET /api/models/list` | GET | List all models in registry |
| `/api/backend/models/{id}` | `/api/models/{id}` | `GET /api/models/{id}` | GET | Get model details |
| `/api/backend/models/{id}/performance` | `/api/models/{id}/performance` | `GET /api/models/{id}/performance` | GET | Get model performance metrics |
| `/api/backend/models/upload` | `/api/models/upload` | `POST /api/models/upload` | POST | Upload new model |
| `/api/backend/models/{id}/history` | `/api/models/{id}/history` | `GET /api/models/{id}/history` | GET | Get model training history |

**Router:** `backend/routers/model_router.py` (prefix: `/api/models`)

### Reports

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/reports/list` | `/api/reports/list` | `GET /api/reports/list` | GET | List all available reports |
| `/api/backend/reports/{id}` | `/api/reports/{id}` | `GET /api/reports/{id}` | GET | Get report details |
| `/api/backend/reports/{id}/download` | `/api/reports/{id}/download` | `GET /api/reports/{id}/download` | GET | Download report (PDF/CSV/JSON) |
| `/api/backend/reports/generate` | `/api/reports/generate` | `POST /api/reports/generate` | POST | Generate new report |

**Router:** `backend/routers/reports_router.py` (prefix: `/api/reports`)

### Replay

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/replay/jobs` | `/api/replay/jobs` | `GET /api/replay/jobs` | GET | List replay jobs |
| `/api/backend/replay/days` | `/api/replay/days` | `GET /api/replay/days` | GET | List available replay days |

**Router:** `backend/routers/replay_router.py` (prefix: `/api/replay`)

### Health

| Frontend Call | Proxy Forwards To | Backend Route | Method | Description |
|--------------|-------------------|---------------|--------|-------------|
| `/api/backend/health` | `/health` | `GET /health` | GET | Backend health check |

**Router:** `backend/routers/health_router.py` (prefix: `/health`)

## DT Backend Endpoints

### Health

| Frontend Call | Proxy Forwards To | DT Backend Route | Method | Description |
|--------------|-------------------|------------------|--------|-------------|
| `/api/dt/health` | `/health` | `GET /health` | GET | DT backend health status |
| `/api/dt/health/ready` | `/health/ready` | `GET /health/ready` | GET | DT backend readiness check |
| `/api/dt/health/live` | `/health/live` | `GET /health/live` | GET | DT backend liveness check |

### Learning Metrics

| Frontend Call | Proxy Forwards To | DT Backend Route | Method | Description |
|--------------|-------------------|------------------|--------|-------------|
| `/api/dt/api/dt/learning/metrics` | `/api/dt/learning/metrics` | `GET /api/dt/learning/metrics` | GET | DT learning metrics |

### Replay

| Frontend Call | Proxy Forwards To | DT Backend Route | Method | Description |
|--------------|-------------------|------------------|--------|-------------|
| `/api/dt/api/replay/start` | `/api/replay/start` | `POST /api/replay/start` | POST | Start DT replay |
| `/api/dt/api/replay/status` | `/api/replay/status` | `GET /api/replay/status` | GET | Get DT replay status |

### Jobs

| Frontend Call | Proxy Forwards To | DT Backend Route | Method | Description |
|--------------|-------------------|------------------|--------|-------------|
| `/api/dt/jobs/cycle` | `/jobs/cycle` | `POST /jobs/cycle` | POST | Trigger DT trading cycle |
| `/api/dt/jobs/status` | `/jobs/status` | `GET /jobs/status` | GET | Get DT job status |

### Data Access

| Frontend Call | Proxy Forwards To | DT Backend Route | Method | Description |
|--------------|-------------------|------------------|--------|-------------|
| `/api/dt/data/rolling` | `/data/rolling` | `GET /data/rolling` | GET | Get DT rolling cache data |
| `/api/dt/data/positions` | `/data/positions` | `GET /data/positions` | GET | Get current DT positions |
| `/api/dt/data/metrics` | `/data/metrics` | `GET /data/metrics` | GET | Get DT metrics |

## Frontend Page → API Calls Mapping

### `/bots` - Bots Management Page

**API Calls Made:**
1. `GET /api/backend/bots/page` - Fetch bot data bundle
2. `POST /api/backend/eod/configs` - Update swing bot config (when saving)
3. `POST /api/backend/intraday/configs` - Update DT bot config (when saving)
4. `EventSource /api/backend/events/bots` - Real-time updates (when Live mode enabled)

**Response Structure:**
```json
{
  "as_of": "2024-01-20T04:13:37Z",
  "swing": {
    "status": { "bots": {...}, "running": true, "last_update": "..." },
    "configs": { "configs": {...} },
    "log_days": [...]
  },
  "intraday": {
    "status": { "bots": {...}, "running": true },
    "configs": { "configs": {...} },
    "pnl_last_day": { "bots": {...}, "total": {...} },
    "signals_latest": { "signals": [...] },
    "fills_recent": { "fills": [...] }
  }
}
```

### `/portfolio` - Portfolio Overview Page

**API Calls Made:**
1. `GET /api/backend/bots/page` - Fetch bot equity data

**Response Structure:** Same as bots page (extracts equity from bot status)

### `/profile` - Profile & Holdings Page

**API Calls Made:**
1. `GET /api/backend/cache/unified` - Fetch unified cache data

**Response Structure:**
```json
{
  "timestamp": "2024-01-20T04:13:37Z",
  "cache_age_seconds": 2.5,
  "version": "1.0",
  "data": {
    "bots": { /* bots page bundle */ },
    "portfolio": { /* top holdings */ },
    "system": { /* system status */ }
  },
  "errors": {}
}
```

### `/insights` - Insights & Predictions Page

**API Calls Made:**
1. `GET /api/backend/insights/predictions/latest` - Fetch prediction feed

**Response Structure:**
```json
{
  "timestamp": "2024-01-20T04:13:37Z",
  "symbols": {
    "AAPL": {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "sector": "Technology",
      "price": 150.25,
      "targets": {
        "1w": { "expected_return": 0.05, "confidence": 0.85 }
      }
    }
  }
}
```

### `/bots/config` - Bot Configuration Page

**API Calls Made:**
1. `GET /api/backend/settings/knobs` - Fetch knobs.env content
2. `GET /api/backend/settings/dt-knobs` - Fetch dt_knobs.env content
3. `POST /api/backend/settings/knobs` - Save knobs.env (when saving)
4. `POST /api/backend/settings/dt-knobs` - Save dt_knobs.env (when saving)

### `/tools/api-keys` - API Keys Management Page

**API Calls Made:**
1. `GET /api/backend/settings/keys` - Fetch API keys
2. `GET /api/backend/settings/status` - Fetch settings status
3. `POST /api/backend/settings/update` - Update API keys (when saving)
4. `POST /api/backend/settings/test` - Test API keys (when testing)

## Common Issues & Solutions

### Issue: Double `/api/api` Prefix

**Symptom:** 404 errors, requests to `/api/api/...`

**Cause:** Frontend code includes `/api/` in the path when calling `/api/backend/*`

**Solution:** Remove the extra `/api/` prefix from frontend calls

**Example:**
```typescript
// ❌ Wrong - double prefix
fetch("/api/backend/api/cache/unified")

// ✅ Correct
fetch("/api/backend/cache/unified")
```

### Issue: 48-Second Timeout on Sequential Requests

**Symptom:** Page hangs for 48+ seconds when first endpoint fails

**Cause:** `tryGetFirst()` uses sequential execution with default fetch timeout

**Solution:** Use parallel requests with explicit timeout (already fixed in this PR)

### Issue: SSE Connection Fails

**Symptom:** Real-time updates don't work, EventSource connection errors

**Cause:** SSE endpoint not accessible or CORS issue

**Solution:** Ensure events router is mounted and accessible through proxy

### Issue: Admin/Dashboard Routes Return 404

**Symptom:** Admin and dashboard pages don't load

**Cause:** Proxy is adding `/api` prefix to routes that shouldn't have it

**Solution:** Proxy should NOT add `/api` prefix for `dashboard/*` and `admin/*` routes (already configured correctly)

## Testing Endpoints

### Using curl

```bash
# Test main backend health
curl http://localhost:8000/health

# Test bots page endpoint
curl http://localhost:8000/api/bots/page

# Test unified cache
curl http://localhost:8000/api/cache/unified

# Test DT backend health
curl http://localhost:8010/health
```

### Using Next.js Dev Server

```bash
cd frontend
npm run dev

# Navigate to:
# http://localhost:3000/bots
# http://localhost:3000/portfolio
# http://localhost:3000/profile
```

### Using Browser DevTools

1. Open page (e.g., `/bots`)
2. Press F12 to open DevTools
3. Go to Network tab
4. Filter by "Fetch/XHR"
5. Refresh page
6. Check request URLs and responses

## Environment Variables

### Backend Configuration

```bash
# .env
BACKEND_URL=http://localhost:8000
DT_BACKEND_URL=http://localhost:8010
```

### Frontend Configuration

```bash
# frontend/.env.local
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_DT_BACKEND_URL=http://localhost:8010
```

**Note:** In production, these should point to the actual backend servers, not localhost.

## Related Documentation

- `/frontend/lib/botsApi.ts` - Bots API client with endpoint documentation
- `/frontend/lib/api.ts` - Main API client with endpoint documentation
- `/frontend/lib/dtApi.ts` - DT API client with endpoint documentation
- `TESTING_API_INTEGRATION.md` - Manual testing guide for API integration
- `/frontend/app/api/backend/[...path]/route.ts` - Main backend proxy implementation
- `/frontend/app/api/dt/[...path]/route.ts` - DT backend proxy implementation

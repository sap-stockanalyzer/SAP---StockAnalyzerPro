# Legacy Router Files - Safe to Delete

## Overview

This document lists legacy router files that have been replaced by consolidated routers in Phase 3 of the router consolidation project. These files are no longer imported or used by `backend_service.py` and can be safely deleted once Phase 3 is verified stable in production.

## Status: NOT DELETED YET

**Why:** These files are preserved to allow for easy rollback if issues arise during Phase 3 verification.

**When to delete:** After Phase 3 has been verified stable in production for at least 1-2 weeks.

## Files Safe to Delete

### Data Router Files (Replaced by `page_data_router.py`)

- ✅ `backend/routers/bots_page_router.py` - Replaced by `/api/page/bots`
- ✅ `backend/routers/bots_hub_router.py` - Functionality merged into page_data_router
- ✅ `backend/routers/dashboard_router.py` - Replaced by `/api/page/dashboard`
- ✅ `backend/routers/portfolio_router.py` - Replaced by `/api/page/profile`
- ✅ `backend/routers/insights_router.py` - Merged into page_data_router
- ✅ `backend/routers/live_prices_router.py` - Merged into page_data_router
- ✅ `backend/routers/eod_bots_router.py` - Merged into page_data_router

### Admin Router Files (Replaced by `admin_consolidated_router.py`)

- ✅ `backend/routers/system_status_router.py` - Replaced by `/api/admin/status`
- ✅ `backend/routers/diagnostics_router.py` - Merged into admin_consolidated_router
- ✅ `backend/routers/metrics_router.py` - Replaced by `/api/admin/metrics`
- ✅ `backend/routers/replay_router.py` - Replaced by `/api/admin/replay/{backend}/status`
- ✅ `backend/routers/swing_replay_router.py` - Merged into admin_consolidated_router
- ✅ `backend/routers/nightly_logs_router.py` - Replaced by `/api/admin/logs`
- ✅ `backend/routers/intraday_logs_router.py` - Merged into admin_consolidated_router

### Settings Router Files (Replaced by `settings_consolidated_router.py`)

- ✅ `backend/routers/settings_router.py` - Replaced by `/api/settings/*`
- ✅ `backend/routers/model_router.py` - Configuration parts merged into settings_consolidated_router

### Intraday Router Files (Replaced by consolidated routers)

- ✅ `backend/routers/intraday_router.py` - Merged into page_data_router
- ✅ `backend/routers/intraday_stream_router.py` - Merged into events_router or admin_consolidated_router
- ✅ `backend/routers/intraday_tape_router.py` - Merged into page_data_router

## Files to KEEP

These files are still in use and should NOT be deleted:

- ✅ `backend/routers/page_data_router.py` - NEW consolidated router
- ✅ `backend/routers/admin_consolidated_router.py` - NEW consolidated router
- ✅ `backend/routers/settings_consolidated_router.py` - NEW consolidated router
- ✅ `backend/routers/health_router.py` - Essential health checks
- ✅ `backend/routers/events_router.py` - SSE streaming endpoints
- ✅ `backend/routers/unified_cache_router.py` - Unified cache service
- ✅ `backend/routers/testing_router.py` - Testing endpoints
- ✅ `backend/routers/system_run_router.py` - Still used by frontend overrides pages
- ✅ `backend/admin/routes.py` - Legacy admin routes (backward compat)
- ✅ `backend/admin/admin_tools_router.py` - Admin tools

## Deletion Command (Run After Phase 3 Verification)

```bash
# Navigate to repository root
cd /path/to/SAP---StockAnalyzerPro

# Delete legacy router files (run this ONLY after verifying Phase 3 is stable)
rm -f backend/routers/bots_page_router.py
rm -f backend/routers/bots_hub_router.py
rm -f backend/routers/dashboard_router.py
rm -f backend/routers/portfolio_router.py
rm -f backend/routers/system_status_router.py
rm -f backend/routers/diagnostics_router.py
rm -f backend/routers/insights_router.py
rm -f backend/routers/live_prices_router.py
rm -f backend/routers/intraday_router.py
rm -f backend/routers/model_router.py
rm -f backend/routers/metrics_router.py
rm -f backend/routers/settings_router.py
rm -f backend/routers/nightly_logs_router.py
rm -f backend/routers/replay_router.py
rm -f backend/routers/swing_replay_router.py
rm -f backend/routers/intraday_logs_router.py
rm -f backend/routers/intraday_stream_router.py
rm -f backend/routers/intraday_tape_router.py
rm -f backend/routers/eod_bots_router.py

# Commit the deletion
git add -A
git commit -m "Phase 3: Delete legacy router files"
git push
```

## Verification Before Deletion

Before deleting these files, verify:

1. ✅ All frontend pages load correctly
2. ✅ Bots page shows data from `/api/page/bots`
3. ✅ Admin pages show data from `/api/admin/*`
4. ✅ Settings pages work with `/api/settings/*`
5. ✅ No console errors about missing endpoints
6. ✅ No backend errors in logs
7. ✅ Production has been stable for 1-2 weeks

## Impact Analysis

**Total files to delete:** 19 router files
**Total size saved:** ~200 KB of code
**Maintenance reduction:** 19 fewer files to maintain
**Code complexity reduction:** Significant (from 25+ routers to 10)

## Rollback Plan

If issues arise after deletion:
1. Restore files from git history: `git checkout HEAD~1 backend/routers/`
2. Re-add imports to `backend_service.py`
3. Add routers to ROUTERS list
4. Restart backend

## Timeline

- **Phase 3 Completion:** [Current Date]
- **Verification Period:** 1-2 weeks
- **Estimated Deletion Date:** [Date + 2 weeks]

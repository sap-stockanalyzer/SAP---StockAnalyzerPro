# backend/backend_service.py — v2.1.2 (clean primary + quiet UI workers)
"""
AION Analytics — Backend Service

Goal
----
- Allow a "PRIMARY" backend (single worker) that owns scheduler/nightly/heartbeat.
- Allow a "UI" backend (multi-worker) that only serves fast read endpoints (no background threads, no noisy startup).

Key env flags
-------------
AION_ROLE: primary | ui (optional but recommended)
QUIET_STARTUP: 1 to suppress startup banners + skip non-essential startup work
ENABLE_SCHEDULER: 1 only on primary
ENABLE_HEARTBEAT: 1 only on primary
ENABLE_CLOUDSYNC: 1 only on primary
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.core.config import PATHS, TIMEZONE
from config import ensure_project_structure

# Scheduler runner (optional fallback)
try:
    from backend.scheduler_runner import main as scheduler_main
except ImportError:
    try:
        from backend.services.scheduler_runner import main as scheduler_main  # type: ignore
    except Exception:
        scheduler_main = None  # type: ignore

# ----------------------------------------------
# Routers (NEW CONSOLIDATED — v2.2.0)
# ----------------------------------------------
# NEW: 3 consolidated routers (replaces 25+ old routers)
from backend.routers.page_data_router import router as page_data_router
from backend.routers.admin_consolidated_router import router as admin_consolidated_router
from backend.routers.settings_consolidated_router import router as settings_consolidated_router

# KEEP: Essential routers still needed
from backend.routers.health_router import router as health_router
from backend.routers.events_router import router as events_router  # SSE endpoints
from backend.routers.unified_cache_router import router as unified_cache_router  # Existing cache

# OPTIONAL: Testing router (may have import issues)
testing_router = None  # Will be set if import succeeds
try:
    from backend.routers.testing_router import router as testing_router
except ImportError as e:
    print(f"[Backend] ⚠️ Testing router not available: {e}")

# KEEP: Legacy admin routers (for backward compat)
from backend.admin.routes import router as admin_router
from backend.admin.admin_tools_router import router as admin_tools_router

# KEEP: system_run_router for backward compatibility with /app/tools/overrides and /app/system/overrides
# Frontend still uses /api/system/run/{task} endpoint
from backend.routers.system_run_router import router as system_run_router

# Optional cloud sync
try:
    from backend import cloud_sync  # type: ignore
except Exception:
    cloud_sync = None


# -------------------------------------------------
# Feature gates (env controlled)
# -------------------------------------------------

def _env_flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "on")


ROLE = (os.getenv("AION_ROLE", "") or "").strip().lower() or "unspecified"

# QUIET_STARTUP is the big one: use it for the multi-worker UI backend so it doesn't spam logs
QUIET_STARTUP = _env_flag("QUIET_STARTUP", "0")

# Only PRIMARY should ever run these:
ENABLE_SCHEDULER = _env_flag("ENABLE_SCHEDULER", "0")
ENABLE_HEARTBEAT = _env_flag("ENABLE_HEARTBEAT", "1")
ENABLE_CLOUDSYNC = _env_flag("ENABLE_CLOUDSYNC", "1")


# -------------------------------------------------
# FastAPI App Configuration
# -------------------------------------------------

app = FastAPI(
    title="AION Analytics Backend",
    description="Backend API + scheduler for AION Analytics.",
    version="2.1.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Add gzip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# -------------------------------------------------
# Mount Routers
# -------------------------------------------------

ROUTERS = [
    # NEW: 3 consolidated routers (v2.2.0)
    page_data_router,           # Replaces: bots_page, dashboard, portfolio, insights, etc.
    admin_consolidated_router,  # Replaces: system_status, diagnostics, metrics, replay, etc.
    settings_consolidated_router,  # Replaces: settings_router
    
    # KEEP: Essential routers
    health_router,              # Health checks
    testing_router,             # Testing endpoints (optional, may be None)
    events_router,              # SSE endpoints
    unified_cache_router,       # Existing unified cache
    
    # KEEP: Legacy routers (backward compat)
    admin_router,               # Legacy admin routes
    admin_tools_router,         # Admin tools
    system_run_router,          # /api/system/run/{task} - still used by frontend overrides pages
]

# Filter out None routers (e.g., testing_router if import failed)
ROUTERS = [r for r in ROUTERS if r is not None]

# Mount all routers with logging
for r in ROUTERS:
    app.include_router(r)

# -------------------------------------------------
# Router Verification on Startup
# -------------------------------------------------

def _log_mounted_routers():
    """Log all mounted routers for verification."""
    if not QUIET_STARTUP:
        print(f"[Backend] 📋 Mounted {len(ROUTERS)} routers:", flush=True)
        for r in ROUTERS:
            prefix = getattr(r, "prefix", "")
            tags = getattr(r, "tags", [])
            print(f"  • {prefix or '/'} ({', '.join(tags) if tags else 'no tags'})", flush=True)

# -------------------------------------------------
# Basic Endpoints
# -------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "AION Analytics Backend",
        "version": "2.1.2",
        "time": datetime.now(TIMEZONE).isoformat(),
        "role": ROLE,
        "quiet_startup": QUIET_STARTUP,
        "enable_scheduler": ENABLE_SCHEDULER,
        "enable_heartbeat": ENABLE_HEARTBEAT,
    }


@app.post("/api/test-alerts")
def test_all_alerts():
    """Test all Slack channels."""
    from backend.monitoring.alerting import (
        alert_error,
        alert_critical,
        alert_dt,
        alert_swing,
        alert_nightly,
        alert_pnl,
        alert_report,
    )
    
    # Test each channel
    alert_error("Test: Error Alert", "Testing #errors-tracebacks")
    alert_critical("Test: Trading Alert", "Testing #trading-alerts", channel="trading")
    alert_dt("Test: DT Alert", "Testing #day_trading")
    alert_swing("Test: Swing Alert", "Testing #swing_trading")
    alert_nightly("Test: Nightly Alert", "Testing #nightly-logs-summary")
    alert_pnl("Test: PnL Alert", "Testing #daily-pnl")
    alert_report("Test: Report Alert", "Testing #reports")
    
    return {
        "status": "ok",
        "message": "Test alerts sent to all configured channels",
        "channels": ["errors", "trading", "dt", "swing", "nightly", "pnl", "reports"],
    }

# -------------------------------------------------
# Background Threads
# -------------------------------------------------

def _backend_heartbeat():
    """Print heartbeat every hour."""
    while True:
        now = datetime.now(TIMEZONE)
        print(f"[Backend] 💓 Heartbeat — {now.isoformat()}", flush=True)
        time.sleep(3600)


def _cache_updater_thread():
    """Run unified cache updater every 5 seconds."""
    try:
        from backend.jobs.cache_updater_job import update_frontend_cache
        print("[CacheUpdater] 🔄 Starting unified cache updater...", flush=True)
        
        while True:
            try:
                update_frontend_cache()
            except Exception as e:
                print(f"[CacheUpdater] ⚠️ Update failed: {e}", flush=True)
            time.sleep(5)
            
    except Exception as e:
        print(f"[CacheUpdater] ❌ Cache updater crashed: {e}", flush=True)


def _rolling_optimizer_thread():
    """Run rolling optimizer every 30 seconds."""
    try:
        from backend.services.rolling_optimizer import optimize_rolling_data
        print("[RollingOptimizer] 🔄 Starting rolling optimizer...", flush=True)
        
        # Initial run
        time.sleep(5)  # Wait for system to stabilize
        
        while True:
            try:
                result = optimize_rolling_data()
                if result.get("status") == "success":
                    stats = result.get("stats", {})
                    print(f"[RollingOptimizer] ✅ Optimized: {stats}", flush=True)
                else:
                    print(f"[RollingOptimizer] ⚠️ Optimization failed: {result.get('errors')}", flush=True)
            except Exception as e:
                print(f"[RollingOptimizer] ⚠️ Optimization error: {e}", flush=True)
            time.sleep(30)  # Run every 30 seconds
            
    except Exception as e:
        print(f"[RollingOptimizer] ❌ Optimizer crashed: {e}", flush=True)


def _scheduler_thread():
    """Run scheduler if available."""
    if scheduler_main is None:
        print("[Scheduler] ⚠️ scheduler_runner not found; skipping.", flush=True)
        return
    try:
        print("[Scheduler] 🧭 Starting scheduler runner…", flush=True)
        scheduler_main()
    except Exception as e:
        print(f"[Scheduler] ❌ Scheduler crashed: {e}", flush=True)


def _print_root_path():
    ensure_project_structure()
    try:
        root = PATHS.get("root")
        if root:
            print(f"[Backend] 📦 Root path: {root}", flush=True)
    except Exception:
        pass

# -------------------------------------------------
# Startup Hook
# -------------------------------------------------

@app.on_event("startup")
async def on_startup():
    # In multi-worker "UI" mode, do not spam banners and do not start any background threads.
    if not QUIET_STARTUP:
        print("────────────────────────────────────────", flush=True)
        print("🚀 AION Analytics Backend — Starting…", flush=True)
        print("────────────────────────────────────────", flush=True)
        _print_root_path()
        _log_mounted_routers()

    # IMPORTANT:
    # Bot bootstrap writes files. If you run it in 2 workers you can race and get noise.
    # So: only do it when NOT quiet.
    if not QUIET_STARTUP:
        try:
            from backend.services.bot_bootstrapper import bootstrap_bots_for_ui  # type: ignore
            bootstrap_bots_for_ui()
            print("[Backend] 🤖 Bot UI bootstrap complete.", flush=True)
        except Exception as e:
            print(f"[Backend] ⚠️ Bot UI bootstrap skipped: {e}", flush=True)

    # Cloud sync: only in primary / non-quiet
    if cloud_sync and ENABLE_CLOUDSYNC and not QUIET_STARTUP:
        try:
            print("[CloudSync] ☁️ Starting background sync tasks…", flush=True)
            cloud_sync.start_background_tasks()
        except Exception as e:
            print(f"[CloudSync] ⚠️ Cloud sync init failed: {e}", flush=True)

    # Heartbeat: only in primary / non-quiet
    if ENABLE_HEARTBEAT and not QUIET_STARTUP:
        threading.Thread(target=_backend_heartbeat, daemon=True).start()

    # Cache updater: run in primary only (avoid duplicate updates)
    if not QUIET_STARTUP:
        threading.Thread(target=_cache_updater_thread, daemon=True).start()
        print("[Backend] 🔄 Cache updater thread enabled.", flush=True)
    
    # Rolling optimizer: run in primary only (new in v2.2.0)
    if not QUIET_STARTUP:
        threading.Thread(target=_rolling_optimizer_thread, daemon=True).start()
        print("[Backend] 🔄 Rolling optimizer thread enabled.", flush=True)

    # Scheduler: MUST be gated (primary only)
    if ENABLE_SCHEDULER and not QUIET_STARTUP:
        threading.Thread(target=_scheduler_thread, daemon=True).start()
        print("[Backend] 🧭 Scheduler thread enabled.", flush=True)
    elif not QUIET_STARTUP:
        print("[Backend] 🧭 Scheduler thread disabled.", flush=True)

    if not QUIET_STARTUP:
        print("[Backend] ✅ Startup complete — ready for requests.", flush=True)

# -------------------------------------------------
# CLI Entrypoint (dev only)
# -------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Dev only: reload=True is fine locally, but do NOT use reload with multi-worker plans.
    uvicorn.run("backend.backend_service:app", host="127.0.0.1", port=8000, reload=True)

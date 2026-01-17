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
# Routers (centralized from backend/routers/)
# ----------------------------------------------
from backend.routers.health_router import router as health_router
from backend.routers.testing_router import router as testing_router
from backend.routers.system_status_router import router as system_router
from backend.routers.diagnostics_router import router as diagnostics_router
from backend.routers.insights_router import router as insights_router
from backend.routers.live_prices_router import router as live_prices_router
from backend.routers.intraday_router import router as intraday_router
from backend.routers.model_router import router as model_router
from backend.routers.metrics_router import router as metrics_router
from backend.routers.settings_router import router as settings_router
from backend.routers.nightly_logs_router import router as nightly_logs_router
from backend.routers.bots_page_router import router as bots_page_router
from backend.routers.bots_hub_router import router as bots_hub_router
from backend.routers.replay_router import router as replay_router
from backend.routers.swing_replay_router import router as swing_replay_router
from backend.routers.intraday_logs_router import router as intraday_logs_router
from backend.routers.dashboard_router import router as dashboard_router
from backend.routers.intraday_stream_router import router as stream_router
from backend.routers.system_run_router import router as system_run_router
from backend.routers.eod_bots_router import router as eod_bots_router
from backend.routers.intraday_tape_router import router as intraday_tape_router
from backend.routers.portfolio_router import router as portfolio_router

from backend.admin.routes import router as admin_router
from backend.admin.admin_tools_router import router as admin_tools_router

# Events router for SSE
from backend.routers.events_router import router as events_router

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
    health_router,
    testing_router,
    system_router,
    diagnostics_router,
    insights_router,
    live_prices_router,
    intraday_router,
    model_router,
    metrics_router,
    settings_router,
    nightly_logs_router,
    bots_page_router,
    bots_hub_router,
    replay_router,
    swing_replay_router,
    dashboard_router,
    portfolio_router,
    intraday_logs_router,
    stream_router,
    system_run_router,
    admin_router,
    admin_tools_router,
    eod_bots_router,
    intraday_tape_router,
    events_router,  # SSE endpoints
]

for r in ROUTERS:
    app.include_router(r)

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

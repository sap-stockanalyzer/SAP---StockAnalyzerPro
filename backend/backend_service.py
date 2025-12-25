# backend/backend_service.py — v2.1.0
"""
AION Analytics — Backend Service

Responsibilities:
    • Mount all FastAPI routers
    • Start scheduler + heartbeat in background threads
    • Provide health & root diagnostic endpoints
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.core.config import PATHS, TIMEZONE

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
from backend.routers.system_status_router import router as system_router
from backend.routers.insights_router import router as insights_router
from backend.routers.live_prices_router import router as live_prices_router
from backend.routers.intraday_router import router as intraday_router
from backend.routers.model_router import router as model_router
from backend.routers.metrics_router import router as metrics_router
from backend.routers.settings_router import router as settings_router
from backend.routers.eod_bots_router import router as eod_router
from backend.routers.replay_router import router as replay_router
from backend.routers.swing_replay_router import router as swing_replay_router
from backend.routers.intraday_logs_router import router as intraday_logs_router
from backend.routers.dashboard_router import router as dashboard_router
from backend.routers.intraday_stream_router import router as stream_router
from backend.routers.system_run_router import router as system_run_router
from backend.admin.routes import router as admin_router
from backend.admin.admin_tools_router import router as admin_tools_router

# Optional cloud sync
try:
    from backend import cloud_sync  # type: ignore
except Exception:
    cloud_sync = None

# -------------------------------------------------
# FastAPI App Configuration
# -------------------------------------------------

app = FastAPI(
    title="AION Analytics Backend",
    description="Backend API + scheduler for AION Analytics.",
    version="2.1.0",
)

# Allow your frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -------------------------------------------------
# Mount Routers (all already include /api/* prefixes)
# -------------------------------------------------

app.include_router(system_router)
app.include_router(insights_router)
app.include_router(live_prices_router)
app.include_router(intraday_router)
app.include_router(model_router)
app.include_router(metrics_router)
app.include_router(settings_router)
app.include_router(eod_router)
app.include_router(replay_router)
app.include_router(swing_replay_router)
app.include_router(dashboard_router)
app.include_router(intraday_logs_router)
app.include_router(stream_router)
app.include_router(system_run_router)
app.include_router(admin_router)
app.include_router(admin_tools_router)

# -------------------------------------------------
# Basic Endpoints
# -------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "AION Analytics Backend",
        "version": "2.1.0",
        "time": datetime.now(TIMEZONE).isoformat(),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.now(TIMEZONE).isoformat(),
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
    try:
        root = PATHS.get("root")
        print(f"[Backend] 📦 Root path: {root}", flush=True)
    except Exception:
        pass

# -------------------------------------------------
# Startup Hook
# -------------------------------------------------

@app.on_event("startup")
async def on_startup():
    print("────────────────────────────────────────", flush=True)
    print("🚀 AION Analytics Backend — Starting…", flush=True)
    print("────────────────────────────────────────", flush=True)

    _print_root_path()

    # Optional cloud sync
    if cloud_sync:
        try:
            print("[CloudSync] ☁️ Starting background sync tasks…", flush=True)
            cloud_sync.start_background_tasks()
        except Exception as e:
            print(f"[CloudSync] ⚠️ Cloud sync init failed: {e}", flush=True)

    # Start heartbeat + scheduler threads
    threading.Thread(target=_backend_heartbeat, daemon=True).start()
    threading.Thread(target=_scheduler_thread, daemon=True).start()

    print("[Backend] ✅ Startup complete — ready for requests.", flush=True)

# -------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("[Backend] Running at http://127.0.0.1:8000", flush=True)
    uvicorn.run("backend.backend_service:app", host="127.0.0.1", port=8000, reload=True)

"""
backend_service.py — v1.8.2 (Fixed for current routers)
Main FastAPI backend service for AION Analytics. 
Updated to work with current consolidated routers.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import threading
import subprocess
import time
from datetime import datetime
import pytz
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

# Config paths
from backend.core.config import PATHS, TIMEZONE

# FastAPI app
app = FastAPI(title="AION Analytics Backend", version="2.1.2")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# ROUTERS (Current working routers)
# ======================================================
from backend.routers.admin_consolidated_router import router as admin_consolidated_router
from backend.routers.health_router import router as health_router
from backend.routers.events_router import router as events_router
from backend.routers.unified_cache_router import router as unified_cache_router
from backend.admin.routes import router as admin_router
from backend.admin.admin_tools_router import router as admin_tools_router
from backend.routers.system_run_router import router as system_run_router
from backend.routers.pnl_dashboard_router import router as pnl_dashboard_router

# Commented out non-existent routers (for future use)
# from backend.routers.page_data_router import router as page_data_router
# from backend.routers.settings_consolidated_router import router as settings_consolidated_router

# Include all routers
ROUTERS = [
    admin_consolidated_router,
    health_router,
    events_router,
    unified_cache_router,
    admin_router,
    admin_tools_router,
    system_run_router,
    pnl_dashboard_router,
]

for router in ROUTERS:
    app.include_router(router)

# Root endpoint
@app.get("/")
def root():
    return {
        "service": "AION Analytics Backend",
        "version": "2.1.2",
        "status": "online",
        "message": "AION — Predict, Learn, Evolve."
    }

# ======================================================
# Background Threads
# ======================================================

def _backend_heartbeat():
    """Hourly heartbeat"""
    tz = pytz.timezone("America/New_York")
    while True:
        now = datetime.now(tz)
        print(f"[Backend] ❤️ Alive — {now:%H:%M %Z}", flush=True)
        time.sleep(3600)

@app.on_event("startup")
def on_startup():
    print("[Backend] 🚀 Startup sequence.. .", flush=True)
    threading.Thread(target=_backend_heartbeat, daemon=True).start()
    print("[Backend] ✅ Ready!", flush=True)

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run("backend.backend_service:app", host="0.0.0.0", port=8000, workers=1)

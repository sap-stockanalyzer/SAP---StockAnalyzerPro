"""Bots Hub Router

One-stop endpoints for the redesigned Bots page.

Why this exists:
  - Your UI wants to render *both* swing (EOD) and intraday bot state,
    configs, and logs without juggling a bunch of endpoints.
  - We keep the existing routers intact, and *aggregate* them here.

This router is intentionally thin: it calls the existing router functions
directly (no HTTP round-trips), so it stays deterministic and cheap.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter

from backend.core.config import TIMEZONE

# Reuse existing router implementations (call their endpoint functions directly)
from backend.routers import eod_bots_router
from backend.routers import intraday_logs_router


router = APIRouter(prefix="/api/bots", tags=["bots-hub"])


@router.get("/overview")
async def bots_overview() -> Dict[str, Any]:
    """Aggregated bot data for the UI.

    Returns:
      {
        "as_of": ...,
        "eod": {"status": ..., "configs": ..., "log_days": ...},
        "intraday": {"status": ..., "configs": ..., "log_days": ..., "pnl_last_day": ...}
      }
    """

    # --- Swing / EOD ---
    eod_status = await eod_bots_router.eod_status()
    eod_configs = await eod_bots_router.list_eod_bot_configs()
    eod_days = await eod_bots_router.eod_log_days()

    # --- Intraday ---
    intraday_status = await intraday_logs_router.intraday_status()
    intraday_configs = await intraday_logs_router.intraday_configs()
    intraday_days = await intraday_logs_router.list_log_days()

    # PnL snapshot is optional (404 if none) — we keep it graceful
    pnl_last_day = None
    try:
        pnl_last_day = await intraday_logs_router.get_last_day_pnl_summary()
    except Exception:
        pnl_last_day = None

    return {
        "as_of": datetime.now(TIMEZONE).isoformat(),
        "eod": {
            "status": eod_status,
            "configs": eod_configs,
            "log_days": eod_days,
        },
        "intraday": {
            "status": intraday_status,
            "configs": intraday_configs,
            "log_days": intraday_days,
            "pnl_last_day": pnl_last_day,
        },
    }

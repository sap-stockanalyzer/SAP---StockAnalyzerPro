"""backend/routers/bots_page_router.py

Unified Bots Page API — AION Analytics

This router exists to make the frontend simple: one call can fetch
all Bot-related data needed to render the Bots page.

Endpoint:
  GET /api/bots/page

The response bundles:
  - swing (EOD) status/configs/log days
  - intraday status/configs/log days

Everything is best-effort: failures in a sub-call return an error object
instead of killing the whole response.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(prefix="/api/bots", tags=["bots"])


def _err(e: Exception) -> Dict[str, Any]:
    return {
        "error": f"{type(e).__name__}: {e}",
        "trace": traceback.format_exc()[-2000:],
    }


@router.get("/page")
async def bots_page_bundle() -> Dict[str, Any]:
    """Return a single payload that contains everything the Bots page needs."""

    out: Dict[str, Any] = {
        "swing": {},
        "intraday": {},
    }

    # Import lazily to avoid circular imports at module load time.
    try:
        from backend.routers import eod_bots_router as eod

        try:
            out["swing"]["status"] = await eod.eod_status()
        except Exception as e:
            out["swing"]["status"] = _err(e)

        try:
            out["swing"]["configs"] = await eod.list_eod_bot_configs()
        except Exception as e:
            out["swing"]["configs"] = _err(e)

        try:
            out["swing"]["log_days"] = await eod.eod_log_days()
        except Exception as e:
            out["swing"]["log_days"] = _err(e)
    except Exception as e:
        out["swing"] = _err(e)

    try:
        from backend.routers import intraday_logs_router as dt

        try:
            out["intraday"]["status"] = await dt.intraday_status()
        except Exception as e:
            out["intraday"]["status"] = _err(e)

        try:
            out["intraday"]["configs"] = await dt.intraday_configs()
        except Exception as e:
            out["intraday"]["configs"] = _err(e)

        try:
            out["intraday"]["log_days"] = await dt.list_log_days()
        except Exception as e:
            out["intraday"]["log_days"] = _err(e)
    except Exception as e:
        out["intraday"] = _err(e)

    return out

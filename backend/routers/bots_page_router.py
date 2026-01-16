"""
backend/routers/bots_page_router.py

Unified Bots Page API — AION Analytics

One endpoint the frontend can hit:
  GET /api/bots/page   (alias: /api/bots/overview)

It bundles:
  - swing (EOD): status/configs/log days
  - intraday: status/configs/log days/pnl last day
  - best-effort: intraday signals + recent fills (file-based)
"""

from __future__ import annotations

import gzip
import importlib
import inspect
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter

try:
    from backend.core.config import PATHS, TIMEZONE
except Exception:
    # tolerate older config import shape
    from backend.config import PATHS, TIMEZONE  # type: ignore

from backend.core.cache_utils import timed_lru_cache


router = APIRouter(prefix="/api/bots", tags=["bots-page"])


# -------------------------
# Helpers
# -------------------------

def _err(e: Exception) -> Dict[str, Any]:
    return {
        "error": f"{type(e).__name__}: {e}",
        "trace": traceback.format_exc()[-2000:],
    }


def _import_module(name: str):
    """Import a module by absolute path, bypassing backend.routers __init__ aliases."""
    return importlib.import_module(name)


async def _call(mod: Any, fn_name: str, *args, **kwargs):
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Module '{getattr(mod, '__name__', mod)}' has no callable '{fn_name}'")
    out = fn(*args, **kwargs)
    if inspect.isawaitable(out):
        return await out
    return out


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_gz_json(path: Path) -> Optional[Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _mtime_iso(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=TIMEZONE).isoformat()
    except Exception:
        return None


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


# -------------------------
# Intraday file helpers (best-effort)
# -------------------------

def _paths_dt() -> Tuple[Path, Path]:
    """
    Return (ML_DATA_DT, DATA_DT) with safe fallbacks.
    """
    root = Path(PATHS.get("root", ".")).resolve()
    ml_data_dt = Path(PATHS.get("ml_data_dt", root / "ml_data_dt"))
    data_dt = Path(PATHS.get("data_dt", root / "data_dt"))
    return ml_data_dt, data_dt


def _load_intraday_signals_best_effort(limit: int = 50) -> Dict[str, Any]:
    """
    Try to read intraday signals from known output files.
    Your logs show:
      /home/aion/aion/Aion_Analytics/data_dt/signals/intraday/predictions/intraday_predictions.json
      /home/aion/aion/Aion_Analytics/ml_data_dt/signals/intraday/ranks/prediction_rank_fetch.json.gz
    """
    ml_data_dt, data_dt = _paths_dt()

    candidates = [
        data_dt / "signals" / "intraday" / "predictions" / "intraday_predictions.json",
        ml_data_dt / "signals" / "intraday" / "predictions" / "intraday_predictions.json",
        ml_data_dt / "signals" / "intraday" / "ranks" / "prediction_rank_fetch.json.gz",
    ]

    p = _first_existing(candidates)
    if not p:
        return {"updated_at": None, "signals": []}

    js = _read_gz_json(p) if p.suffix.endswith(".gz") else _read_json(p)
    items: List[Dict[str, Any]] = []

    # Normalize to a list of dict rows
    if isinstance(js, list):
        items = [x for x in js if isinstance(x, dict)]
    elif isinstance(js, dict):
        for k in ("signals", "results", "items", "top"):
            v = js.get(k)
            if isinstance(v, list):
                items = [x for x in v if isinstance(x, dict)]
                break
        if not items:
            # sometimes the file itself is {symbol: {...}}
            if all(isinstance(v, dict) for v in js.values()):
                items = [v for v in js.values() if isinstance(v, dict)]

    return {
        "source": str(p),
        "updated_at": _mtime_iso(p),
        "signals": items[: max(0, int(limit))],
    }


def _latest_intraday_day(sim_logs: Path) -> Optional[str]:
    """
    sim_logs contains files like YYYY-MM-DD_botname.json
    """
    try:
        days = set()
        for f in sim_logs.glob("*.json"):
            nm = f.name
            parts = nm.split("_")
            if len(parts) >= 2 and len(parts[0]) == 10:
                days.add(parts[0])
        return sorted(days)[-1] if days else None
    except Exception:
        return None


def _extract_fill_like_rows(payload: Any) -> List[Dict[str, Any]]:
    """
    Best-effort extraction from unknown shapes.
    Accepts list/dict and tries to find 'fills'/'trades'/'orders' arrays.
    """
    if payload is None:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for k in ("fills", "trades", "orders", "executions", "items"):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _load_intraday_fills_best_effort(limit: int = 50) -> Dict[str, Any]:
    """
    Try common places:
      - ml_data_dt/sim_logs (derive from per-bot logs)
      - ml_data_dt/*fills*.json (if you later add one)
    """
    ml_data_dt, _ = _paths_dt()

    sim_logs = ml_data_dt / "sim_logs"
    candidates = [
        ml_data_dt / "paper_fills.json",
        ml_data_dt / "broker_fills.json",
        ml_data_dt / "fills.json",
        ml_data_dt / "paper_trades.json",
    ]

    fills: List[Dict[str, Any]] = []
    src: Optional[str] = None
    updated_at: Optional[str] = None

    p = _first_existing(candidates)
    if p:
        js = _read_json(p)
        fills = _extract_fill_like_rows(js)
        src = str(p)
        updated_at = _mtime_iso(p)

    # fallback: derive from sim_logs latest day
    if not fills and sim_logs.exists():
        day = _latest_intraday_day(sim_logs)
        if day:
            rows: List[Dict[str, Any]] = []
            latest_m = 0.0
            for f in sim_logs.glob(f"{day}_*.json"):
                js = _read_json(f)
                rows.extend(_extract_fill_like_rows(js))
                try:
                    latest_m = max(latest_m, f.stat().st_mtime)
                except Exception:
                    pass

            # crude normalization: keep only rows that look trade-ish
            cleaned = []
            for r in rows:
                sym = r.get("symbol") or r.get("sym") or r.get("ticker")
                side = r.get("side") or r.get("action") or r.get("intent")
                if sym or side:
                    cleaned.append(r)

            fills = cleaned
            src = str(sim_logs)
            if latest_m > 0:
                updated_at = datetime.fromtimestamp(latest_m, tz=TIMEZONE).isoformat()

    return {
        "source": src,
        "updated_at": updated_at,
        "fills": fills[: max(0, int(limit))],
    }


# -------------------------
# Main bundle endpoint(s)
# -------------------------

@router.get("/page")
@timed_lru_cache(seconds=5, maxsize=10)
async def bots_page_bundle() -> Dict[str, Any]:
    """
    Single payload that contains everything the Bots page needs.
    Best-effort: sub-call failures become error objects instead of 500'ing the whole response.
    Cached for 5 seconds to reduce file reads and computation.
    """
    out: Dict[str, Any] = {
        "as_of": datetime.now(TIMEZONE).isoformat(),
        "swing": {},
        "intraday": {},
    }

    # --- Swing / EOD ---
    try:
        eod = _import_module("backend.routers.eod_bots_router")
        try:
            out["swing"]["status"] = await _call(eod, "eod_status")
        except Exception as e:
            out["swing"]["status"] = _err(e)

        try:
            out["swing"]["configs"] = await _call(eod, "list_eod_bot_configs")
        except Exception as e:
            out["swing"]["configs"] = _err(e)

        try:
            out["swing"]["log_days"] = await _call(eod, "eod_log_days")
        except Exception as e:
            out["swing"]["log_days"] = _err(e)
    except Exception as e:
        out["swing"] = _err(e)

    # --- Intraday ---
    try:
        dt = _import_module("backend.routers.intraday_logs_router")
        try:
            out["intraday"]["status"] = await _call(dt, "intraday_status")
        except Exception as e:
            out["intraday"]["status"] = _err(e)

        try:
            out["intraday"]["configs"] = await _call(dt, "intraday_configs")
        except Exception as e:
            out["intraday"]["configs"] = _err(e)

        try:
            out["intraday"]["log_days"] = await _call(dt, "list_log_days")
        except Exception as e:
            out["intraday"]["log_days"] = _err(e)

        # Optional PnL summary (from sim_summary.json)
        try:
            out["intraday"]["pnl_last_day"] = await _call(dt, "get_last_day_pnl_summary")
        except Exception as e:
            out["intraday"]["pnl_last_day"] = _err(e)

    except Exception as e:
        out["intraday"] = _err(e)

    # --- Best-effort “live-ish” artifacts ---
    try:
        out["intraday"]["signals_latest"] = _load_intraday_signals_best_effort(limit=50)
    except Exception as e:
        out["intraday"]["signals_latest"] = _err(e)

    try:
        out["intraday"]["fills_recent"] = _load_intraday_fills_best_effort(limit=50)
    except Exception as e:
        out["intraday"]["fills_recent"] = _err(e)

    return out


@router.get("/overview")
async def bots_overview() -> Dict[str, Any]:
    # alias for convenience
    return await bots_page_bundle()

# backend/routers/intraday_logs_router.py
"""
Intraday Day-Trading Bot Logs API — AION Analytics

Exposes:
    • /api/intraday/logs/last-day
    • /api/intraday/logs/days
    • /api/intraday/logs/{day}
    • /api/intraday/logs/{day}/{bot_name}
    • /api/intraday/pnl/last-day

Reads from:
    ml_data_dt/sim_logs/
    ml_data_dt/sim_summary.json
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.config import PATHS
from settings import BOT_KNOBS_DEFAULTS

router = APIRouter(prefix="/api/intraday", tags=["intraday-bot"])

ML_DATA_DT = Path(PATHS.get("ml_data_dt", "ml_data_dt"))
SIM_LOG_DIR = ML_DATA_DT / "sim_logs"
SIM_SUMMARY_FILE = ML_DATA_DT / "sim_summary.json"

# Optional intraday bot config store (UI-facing)
CONFIG_DIR = ML_DATA_DT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
INTRADAY_CONFIG_FILE = CONFIG_DIR / "intraday_bots_ui.json"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _get_available_days() -> List[str]:
    """Return sorted list of unique YYYY-MM-DD available in sim_logs."""
    if not SIM_LOG_DIR.exists():
        return []
    days = set()
    for f in SIM_LOG_DIR.glob("*.json"):
        name = f.name
        # expected: YYYY-MM-DD_botname.json
        parts = name.split("_")
        if len(parts) >= 2:
            day = parts[0]
            days.add(day)
    return sorted(days)

def _get_latest_day() -> Optional[str]:
    days = _get_available_days()
    return days[-1] if days else None


def _list_bot_names() -> List[str]:
    """Best-effort bot name discovery."""
    day = _get_latest_day()
    if not day:
        return []
    files = list(SIM_LOG_DIR.glob(f"{day}_*.json"))
    bots: List[str] = []
    for f in files:
        nm = f.stem.replace(f"{day}_", "")
        if nm:
            bots.append(nm)
    return sorted(set(bots))


def _load_intraday_ui_configs() -> Dict[str, Any]:
    try:
        if INTRADAY_CONFIG_FILE.exists():
            with INTRADAY_CONFIG_FILE.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    return {}


def _save_intraday_ui_configs(obj: Dict[str, Any]) -> None:
    try:
        INTRADAY_CONFIG_FILE.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    except Exception:
        return


def _infer_intraday_bot_names(last_day: Optional[str]) -> List[str]:
    """Infer bot names from most recent day's log files."""
    if not last_day or not SIM_LOG_DIR.exists():
        return []
    
    bots: List[str] = []
    for f in SIM_LOG_DIR.glob(f"{last_day}_*.json"):
        bot_name = f.stem.replace(f"{last_day}_", "")
        if bot_name:
            bots.append(bot_name)
    return sorted(set(bots))


# Default UI configuration for intraday bots
_DEFAULT_INTRADAY_UI: Dict[str, Any] = {
    "enabled": True,
    "type": "intraday",
    "horizon": "1d",
    "risk": "moderate",
    "rules": BOT_KNOBS_DEFAULTS.get("intraday", {}),
}


def _ensure_intraday_ui_defaults(bots: List[str], store: Dict[str, Any]) -> None:
    """Ensure all discovered bots have default UI configs."""
    if "bots" not in store or not isinstance(store["bots"], dict):
        store["bots"] = {}
    
    changed = False
    for bot_name in bots:
        if bot_name not in store["bots"]:
            store["bots"][bot_name] = dict(_DEFAULT_INTRADAY_UI)
            changed = True
    
    if changed:
        _save_intraday_ui_configs(store)


def _ensure_intraday_defaults() -> None:
    """Create default UI configs for discovered intraday bots."""
    bots = _list_bot_names()
    if not bots:
        return

    store = _load_intraday_ui_configs()
    if not isinstance(store, dict):
        store = {}

    _ensure_intraday_ui_defaults(bots, store)


class IntradayConfigUpdateRequest(BaseModel):
    bot_name: str
    config: Dict[str, Any]


@router.get("/status")
async def intraday_status() -> Dict[str, Any]:
    """Lightweight status for the intraday day-trading bot engine.

    Best-effort, file-based inspection. Always returns JSON (no 500s unless something
    truly wild happens).
    """
    try:
        last_day = _get_latest_day()

        # Discover bots primarily from UI config store (so the page always has rows),
        # falling back to log-derived names.
        store = _load_intraday_ui_configs()
        ui_bots = list((store.get("bots") or {}).keys())
        discovered = _infer_intraday_bot_names(last_day)

        bot_keys = ui_bots or discovered

        # "Running" heuristic: do we have fresh-ish artifacts?
        freshest_path = None
        freshest_mtime = 0.0
        candidates = []
        if SIM_SUMMARY_FILE.exists():
            candidates.append(SIM_SUMMARY_FILE)
        if last_day:
            candidates.extend(list(SIM_LOG_DIR.glob(f"{last_day}_*.json")))

        for p in candidates:
            try:
                mt = p.stat().st_mtime
                if mt > freshest_mtime:
                    freshest_mtime = mt
                    freshest_path = p
            except Exception:
                continue

        import time as _time
        age_s = float(_time.time() - freshest_mtime) if freshest_mtime > 0 else None
        running = bool(age_s is not None and age_s < 60 * 30)

        # Pull a best-effort equity/positions snapshot from sim_summary.json
        summary = _read_json(SIM_SUMMARY_FILE) or {}
        latest_day_node = None
        try:
            days = summary.get("days") or []
            if isinstance(days, list) and days:
                latest_day_node = days[-1]
        except Exception:
            latest_day_node = None

        summary_bots = (latest_day_node or {}).get("bots") if isinstance(latest_day_node, dict) else None
        if not isinstance(summary_bots, dict):
            summary_bots = {}

        bots_out: Dict[str, Any] = {}
        for bot_key in bot_keys:
            ui = (store.get("bots") or {}).get(bot_key, {}) if isinstance(store, dict) else {}
            enabled = bool(ui.get("enabled", True))

            node = summary_bots.get(bot_key, {}) if isinstance(summary_bots, dict) else {}
            if not isinstance(node, dict):
                node = {}

            equity = None
            try:
                equity = float(node.get("equity")) if node.get("equity") is not None else None
            except Exception:
                equity = None

            positions = None
            try:
                positions = int(node.get("positions")) if node.get("positions") is not None else None
            except Exception:
                positions = None

            # Latest per-bot artifact time
            bot_mtime = None
            if last_day:
                for p in SIM_LOG_DIR.glob(f"{last_day}_{bot_key}.json"):
                    try:
                        bot_mtime = max(bot_mtime or 0.0, p.stat().st_mtime)
                    except Exception:
                        continue

            bots_out[bot_key] = {
                "enabled": enabled,
                "equity": equity,
                "holdings_count": positions,
                "last_update": datetime.fromtimestamp(bot_mtime).isoformat() if bot_mtime else None,
                # UI rules (best effort)
                "rules": ui.get("rules", {}),
                "type": ui.get("type"),
                "horizon": ui.get("horizon"),
                "risk": ui.get("risk"),
            }

        return {
            "running": running,
            "last_update": datetime.fromtimestamp(freshest_mtime).isoformat() if freshest_mtime else None,
            "last_day": last_day,
            "bot_count": len(bots_out),
            "bots": bots_out,
            "freshest_artifact": str(freshest_path) if freshest_path else None,
            "freshness_age_s": age_s,
        }
    except Exception as e:
        import traceback
        return {
            "running": False,
            "last_update": None,
            "last_day": _get_latest_day(),
            "bot_count": 0,
            "bots": {},
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-2000:],
        }


@router.get("/configs")
async def intraday_configs() -> Dict[str, Any]:
    """Return UI-facing intraday bot configs.

    Stored as ML_DATA_DT/config/intraday_bots_ui.json.
    """
    try:
        store = _load_intraday_ui_configs()
        last_day = _get_latest_day()
        bots = _infer_intraday_bot_names(last_day)
        _ensure_intraday_ui_defaults(bots, store)
        return {"configs": (store.get("bots", {}) if isinstance(store, dict) else {})}
    except Exception as e:
        import traceback
        return {
            "configs": {},
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-2000:],
        }



@router.post("/configs")
async def update_intraday_config(payload: IntradayConfigUpdateRequest) -> Dict[str, Any]:
    store = _load_intraday_ui_configs()
    bots = store.get("bots", {})
    name = str(payload.bot_name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="bot_name is required")

    current = bots.get(name, {})
    if not isinstance(current, dict):
        current = {}

    merged = {**current, **(payload.config or {})}
    bots[name] = merged
    store["bots"] = bots
    _save_intraday_ui_configs(store)
    return {"bot_name": name, "config": merged}


@router.post("/configs/{bot_name}/reset-defaults")
async def reset_intraday_config(bot_name: str) -> Dict[str, Any]:
    store = _load_intraday_ui_configs()
    bots = store.get("bots", {})
    bots[str(bot_name)] = dict(_DEFAULT_INTRADAY_UI)
    store["bots"] = bots
    _save_intraday_ui_configs(store)
    return {"bot_name": bot_name, "config": bots[str(bot_name)]}


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.get("/logs/days")
async def list_log_days():
    """
    Returns list of trading days that have logs.
    Example:
      ["2025-01-09", "2025-01-10", "2025-01-11"]
    """
    days = _get_available_days()
    return {"count": len(days), "days": days}


@router.get("/logs/last-day")
async def get_last_day_logs():
    """
    Returns logs for the most recent trading day — all bots.
    """
    day = _get_latest_day()
    if not day:
        raise HTTPException(404, "No trading-day logs found.")
    return await get_logs_for_day(day)


@router.get("/logs/{day}")
async def get_logs_for_day(day: str):
    """
    Returns logs for all bots for a given day.
    Output:
        {
          "date": "2025-01-11",
          "bots": {
              "momentum_bot": {...},
              "hybrid_bot": {...},
              ...
          }
        }
    """
    files = list(SIM_LOG_DIR.glob(f"{day}_*.json"))
    if not files:
        raise HTTPException(404, f"No logs found for day '{day}'.")

    out: Dict[str, Any] = {"date": day, "bots": {}}
    for f in files:
        bot_name = f.stem.replace(f"{day}_", "")
        js = _read_json(f)
        if js is not None:
            out["bots"][bot_name] = js

    return out


@router.get("/logs/{day}/{bot_name}")
async def get_logs_for_bot(day: str, bot_name: str):
    """
    Returns logs for a single bot for a given day.
    """
    file = SIM_LOG_DIR / f"{day}_{bot_name}.json"
    if not file.exists():
        raise HTTPException(404, f"No logs for bot '{bot_name}' on day '{day}'.")
    js = _read_json(file)
    return {"date": day, "bot": bot_name, "log": js}


@router.get("/pnl/last-day")
async def get_last_day_pnl_summary():
    """
    Returns PnL summary for the latest trading day from sim_summary.json.

    Output format:
        {
          "date": "2025-01-11",
          "bots": {
              "momentum_bot": {"equity": 104.21, "positions": 2, "trades": 8},
              ...
          }
        }
    """
    summary = _read_json(SIM_SUMMARY_FILE)
    if not summary:
        raise HTTPException(404, "No simulation summary found.")

    days = summary.get("days", [])
    if not days:
        raise HTTPException(404, "Simulation summary contains no day entries.")

    last = days[-1]
    return last

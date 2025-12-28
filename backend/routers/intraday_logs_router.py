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
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.config import PATHS

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


def _ensure_intraday_defaults() -> None:
    """Create default UI configs for discovered intraday bots."""
    bots = _list_bot_names()
    if not bots:
        return

    store = _load_intraday_ui_configs()
    if not isinstance(store, dict):
        store = {}

    changed = False
    for b in bots:
        node = store.get(b)
        if not isinstance(node, dict):
            node = {}
        if "enabled" not in node:
            node["enabled"] = True
            changed = True
        if "aggression" not in node:
            node["aggression"] = 0.5
            changed = True
        if "max_alloc" not in node:
            node["max_alloc"] = 1000.0
            changed = True
        if "max_positions" not in node:
            node["max_positions"] = 10
            changed = True
        if "stop_loss" not in node:
            node["stop_loss"] = 0.02
            changed = True
        if "take_profit" not in node:
            node["take_profit"] = 0.03
            changed = True
        if "penny_only" not in node:
            node["penny_only"] = False
            changed = True
        store[b] = node

    if changed:
        _save_intraday_ui_configs(store)


class IntradayConfigUpdateRequest(BaseModel):
    bot_name: str
    config: Dict[str, Any]


@router.get("/status")
async def intraday_status() -> Dict[str, Any]:
    """Lightweight status for the intraday day-trading bot engine.

    This does NOT trigger any cycle; it only inspects the latest artifacts
    on disk.
    """
    last_day = _get_latest_day()
    bots = _infer_intraday_bot_names(last_day)

    # "Running" is a best-effort heuristic: do we have fresh-ish artifacts?
    # If you later add a real heartbeat file in dt_backend, wire it here.
    freshest_path = None
    freshest_mtime = 0.0
    try:
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
    except Exception:
        pass

    age_s = None
    running = False
    if freshest_mtime > 0:
        import time as _time
        age_s = float(_time.time() - freshest_mtime)
        running = age_s < 60 * 30  # 30 minutes "fresh" window

    return {
        "running": bool(running),
        "last_day": last_day,
        "bot_count": len(bots),
        "bots": bots,
        "freshest_artifact": str(freshest_path) if freshest_path else None,
        "freshness_age_s": age_s,
    }


@router.get("/configs")
async def intraday_configs() -> Dict[str, Any]:
    """Return UI-facing intraday bot configs.

    Stored as ML_DATA_DT/config/intraday_bots_ui.json.
    """
    store = _load_intraday_ui_configs()
    last_day = _get_latest_day()
    bots = _infer_intraday_bot_names(last_day)
    _ensure_intraday_ui_defaults(bots, store)
    return {"configs": store.get("bots", {})}


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

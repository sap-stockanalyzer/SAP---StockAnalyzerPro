# backend/routers/eod_bots_router.py
"""
EOD (swing) bots API — AION Analytics

Exposes:
    • GET /api/eod/status
        → Current PnL & positions per bot (using bot state + latest rolling prices)

    • GET /api/eod/logs/days
        → List of trading days with EOD bot logs

    • GET /api/eod/logs/last-day
        → Logs for latest trading day (all horizons, all bots)

    • GET /api/eod/logs/{day}
        → Logs for a specific day, separated by horizon and bot

    • GET /api/eod/logs/{day}/{horizon}/{bot_name}
        → Logs for a specific bot on a specific day & horizon

    • GET /api/eod/configs
        → Current bot configs (eod_1w / eod_2w / eod_4w)

    • GET /api/eod/configs/{bot_key}
        → Config for a single bot

    • PUT /api/eod/configs/{bot_key}
        → Update config for a single bot (partial payload allowed)
"""

from __future__ import annotations

import json
import gzip
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from backend.core.config import PATHS
except ImportError:
    from backend.config import PATHS  # type: ignore

# We import SwingBotConfig for typing only; config I/O is handled lazily
from backend.bots.base_swing_bot import SwingBotConfig  # type: ignore

router = APIRouter(prefix="/api/eod", tags=["eod-bots"])

ML_DATA = Path(PATHS.get("ml_data", "ml_data"))
STOCK_CACHE = Path(PATHS.get("stock_cache", "data_cache"))

# Where nightly bots keep state + logs
BOT_STATE_DIR = STOCK_CACHE / "master" / "bot"
BOT_LOG_ROOT = ML_DATA / "bot_logs"

# Horizons used by your nightly bots
HORIZONS = ["1w", "2w", "4w"]


# -------------------------------------------------------------------
# UI config overlays (enabled/aggression) + helpers
# -------------------------------------------------------------------

try:
    # Optional overlay store for UI-only fields (enabled/aggression, etc.)
    from backend.services.bot_ui_store import (
        ensure_swing_ui_defaults,
        load_swing_ui_overrides,
        update_swing_ui_overrides,
        reset_swing_ui_overrides,
    )
except Exception:  # pragma: no cover
    ensure_swing_ui_defaults = None  # type: ignore
    load_swing_ui_overrides = None  # type: ignore
    update_swing_ui_overrides = None  # type: ignore
    reset_swing_ui_overrides = None  # type: ignore


def _ui_enabled(bot_key: str, overrides: Dict[str, Any]) -> bool:
    try:
        node = overrides.get(bot_key) or {}
        return bool(node.get("enabled", True))
    except Exception:
        return True


def _ui_aggression(bot_key: str, overrides: Dict[str, Any]) -> float:
    try:
        node = overrides.get(bot_key) or {}
        val = float(node.get("aggression", 0.5))
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5


class _UIConfigUpdate(BaseModel):
    bot_key: str
    config: Dict[str, Any]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _load_gz_dict(path: Path) -> Optional[dict]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default  # NaN check
    except Exception:
        return default


def _load_prices_from_rolling() -> Dict[str, float]:
    """
    Load latest prices from core rolling.json.gz via backend.core.data_pipeline,
    falling back to direct file read if needed.

    Schema (core rolling):
        { "AAPL": {...}, "MSFT": {...}, ... }

    and picks price from: price / last / close / c
    """
    data: Dict[str, Any] = {}

    # Try to use core helper first
    try:
        from backend.core.data_pipeline import _read_rolling  # type: ignore

        data = _read_rolling() or {}
    except Exception:
        # Fallback: direct file read from ml_data/rolling.json.gz
        roll_path = ML_DATA / "rolling.json.gz"
        if roll_path.exists():
            maybe = _load_gz_dict(roll_path)
            if isinstance(maybe, dict):
                data = maybe

    if not isinstance(data, dict):
        return {}

    prices: Dict[str, float] = {}
    for sym, node in data.items():
        if isinstance(sym, str) and sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue

        price = (
            node.get("price")
            or node.get("last")
            or node.get("close")
            or node.get("c")
        )
        try:
            price_f = float(price)
        except Exception:
            continue
        if price_f > 0:
            prices[str(sym).upper()] = price_f
    return prices


def _load_bot_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    if path.suffix.endswith(".gz"):
        return _load_gz_dict(path)
    return _load_json(path)


def _list_log_days() -> List[str]:
    """
    Scan ml_data/bot_logs/<HORIZON>/bot_activity_YYYY-MM-DD.json
    and return a sorted list of unique days.
    """
    days = set()
    if not BOT_LOG_ROOT.exists():
        return []
    for horizon in HORIZONS:
        hdir = BOT_LOG_ROOT / horizon
        if not hdir.exists():
            continue
        for f in hdir.glob("bot_activity_*.json"):
            name = f.name
            if not name.startswith("bot_activity_"):
                continue
            day = name.replace("bot_activity_", "").replace(".json", "")
            days.add(day)
    return sorted(days)


def _latest_day() -> Optional[str]:
    days = _list_log_days()
    return days[-1] if days else None


# -------------------------------------------------------------------
# Status endpoint — current PnL / positions
# -------------------------------------------------------------------


@router.get("/status")
async def eod_status():
    """
    Current state snapshot for EOD (swing) bots.

    Uses:
      • bot state files: stock_cache/master/bot/rolling_<botkey>.json(.gz)
      • latest prices: via backend.core.data_pipeline._read_rolling() (primary)
        with a best-effort file fallback.

    Returns a UI-friendly shape:
      {
        "running": bool,
        "last_update": "ISO",
        "bots": {
           "eod_1w": {
              "enabled": true,
              "cash": 123.0,
              "invested": 456.0,
              "allocated": 600.0,
              "holdings_count": 3,
              "equity": 579.0,
              "last_update": "ISO",
              "positions": [...]
           },
           ...
        }
      }
    """
    try:
        prices = _load_prices_from_rolling()
        price_status = "ok" if prices else "no-prices"

        # Load core bot configs (best effort)
        core_cfgs: Dict[str, Any] = {}
        try:
            core_cfgs = _load_all_bot_configs()  # type: ignore
        except Exception:
            core_cfgs = {}

        # Determine bot keys: configs first, else state files, else defaults
        bot_keys: List[str] = list(core_cfgs.keys())

        BOT_STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_files = sorted(list(BOT_STATE_DIR.glob("rolling_*.json.gz")) + list(BOT_STATE_DIR.glob("rolling_*.json")))
        if not bot_keys and state_files:
            bot_keys = [sf.stem.replace("rolling_", "").replace(".json", "") for sf in state_files]

        if not bot_keys:
            bot_keys = [f"eod_{h}" for h in HORIZONS]

        ui_overrides = _load_bot_ui_overrides()

        bots_out: Dict[str, Any] = {}
        last_updates: List[str] = []

        for bot_key in sorted(set(bot_keys)):
            ui = ui_overrides.get(bot_key, {}) if isinstance(ui_overrides, dict) else {}
            enabled = bool(ui.get("enabled", True))

            # Try state file
            state_path_gz = BOT_STATE_DIR / f"rolling_{bot_key}.json.gz"
            state_path_js = BOT_STATE_DIR / f"rolling_{bot_key}.json"
            state = _load_bot_state(state_path_gz) or _load_bot_state(state_path_js) or {}
            if not isinstance(state, dict):
                state = {}

            # Pull cash; tolerate None / strings
            cash = _safe_float(state.get("cash") or 0.0, 0.0)

            # If we have a core config, use its initial_cash as a default when state is missing
            cfg = core_cfgs.get(bot_key)
            if cash <= 0 and cfg is not None:
                cash = _safe_float(getattr(cfg, "initial_cash", None), cash)

            raw_positions = state.get("positions") or {}
            positions = raw_positions if isinstance(raw_positions, dict) else {}

            invested = 0.0
            pos_list = []
            for sym, pos in positions.items():
                sym_u = str(sym).upper()
                px = prices.get(sym_u)

                if isinstance(pos, dict):
                    entry = _safe_float(pos.get("entry") or 0.0, 0.0)
                    qty = _safe_float(pos.get("qty") or 0.0, 0.0)
                    stop = _safe_float(pos.get("stop") or 0.0, 0.0)
                    target = _safe_float(pos.get("target") or 0.0, 0.0)
                else:
                    entry = qty = stop = target = 0.0

                mv = 0.0
                unreal = None
                if px is not None:
                    try:
                        px_f = float(px)
                        if px_f > 0 and qty:
                            mv = px_f * qty
                            invested += mv
                            unreal = (px_f - entry) * qty
                    except Exception:
                        pass

                pos_list.append(
                    {
                        "symbol": sym_u,
                        "qty": qty,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "last_price": px,
                        "market_value": mv,
                        "unrealized_pnl": unreal,
                    }
                )

            equity = cash + invested

            # Allocation limit is a UI concept; default 0.95 if missing
            max_alloc = ui.get("max_alloc", 0.95)
            max_alloc_f = _safe_float(max_alloc, 0.95)
            allocated = equity * max_alloc_f

            last_updated = str(state.get("last_updated") or state.get("last_update") or "")
            if not last_updated:
                # fall back to state file mtime if present
                try:
                    p = state_path_gz if state_path_gz.exists() else state_path_js
                    if p.exists():
                        last_updated = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
                except Exception:
                    last_updated = ""

            if last_updated:
                last_updates.append(last_updated)

            bots_out[bot_key] = {
                "enabled": enabled,
                "cash": cash,
                "invested": invested,
                "allocated": allocated,
                "holdings_count": len(pos_list),
                "equity": equity,
                "last_update": last_updated,
                "positions": pos_list,
            }

        running = any(v.get("enabled") for v in bots_out.values())
        last_update = max(last_updates) if last_updates else ""

        return {
            "running": running,
            "last_update": last_update,
            "price_status": price_status,
            "bots": bots_out,
        }

    except Exception as e:
        import traceback

        raise HTTPException(
            status_code=500,
            detail={
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-2000:],
            },
        )


# -------------------------------------------------------------------
# Logs — days / last-day / per-day / per-bot
# -------------------------------------------------------------------


@router.get("/logs/days")
async def eod_log_days():
    """
    List of trading days that have EOD bot logs (for any horizon).
    """
    days = _list_log_days()
    return {"count": len(days), "days": days}


@router.get("/logs/last-day")
async def eod_logs_last_day():
    """
    Logs for the most recent trading day, all horizons and bots.
    """
    day = _latest_day()
    if not day:
        raise HTTPException(status_code=404, detail="No EOD bot logs found.")
    return await eod_logs_for_day(day)


@router.get("/logs/{day}")
async def eod_logs_for_day(day: str):
    """
    Logs for a specific day, separated by horizon and bot.

    Output:
      {
        "date": "YYYY-MM-DD",
        "horizons": {
            "1w": { "bots": { "eod_1w": [...], ... } },
            "2w": { ... },
            "4w": { ... }
        }
      }
    """
    if not BOT_LOG_ROOT.exists():
        raise HTTPException(
            status_code=404, detail="No EOD bot logs root folder found."
        )

    horizons_out: Dict[str, Any] = {}
    found_any = False

    for horizon in HORIZONS:
        hdir = BOT_LOG_ROOT / horizon
        if not hdir.exists():
            continue

        f = hdir / f"bot_activity_{day}.json"
        if not f.exists():
            continue

        js = _load_json(f)
        if not isinstance(js, dict):
            continue

        horizons_out[horizon] = {"bots": js}
        found_any = True

    if not found_any:
        raise HTTPException(
            status_code=404,
            detail=f"No EOD bot logs found for day '{day}'.",
        )

    return {
        "date": day,
        "horizons": horizons_out,
    }


@router.get("/logs/{day}/{horizon}/{bot_name}")
async def eod_logs_for_bot(day: str, horizon: str, bot_name: str):
    """
    Logs for a single bot on a specific day & horizon.

    Example:
      /api/eod/logs/2025-01-11/1w/eod_1w
    """
    horizon = horizon.lower()
    if horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon '{horizon}'. Expected one of {HORIZONS}.",
        )

    f = BOT_LOG_ROOT / horizon / f"bot_activity_{day}.json"
    if not f.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No EOD bot log file for day '{day}' and horizon '{horizon}'.",
        )

    js = _load_json(f)
    if not isinstance(js, dict):
        raise HTTPException(status_code=500, detail="Corrupt EOD log file.")

    actions = js.get(bot_name)
    if actions is None:
        raise HTTPException(
            status_code=404,
            detail=f"No log entries for bot '{bot_name}' on '{day}' (horizon '{horizon}').",
        )

    return {
        "date": day,
        "horizon": horizon,
        "bot": bot_name,
        "actions": actions,
    }


# -------------------------------------------------------------------
# Config endpoints — view / update bot configs
# -------------------------------------------------------------------


def _load_all_bot_configs() -> Dict[str, SwingBotConfig]:
    """
    Lazy import of backend.bots.config_store so that an import error there
    doesn't prevent this router from being registered at all.
    """
    try:
        from backend.bots import config_store  # type: ignore

        return config_store.load_all_bot_configs()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load bot configs: {e}",
        )


def _save_all_bot_configs(configs: Dict[str, SwingBotConfig]) -> None:
    try:
        from backend.bots import config_store  # type: ignore

        config_store.save_all_bot_configs(configs)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save bot configs: {e}",
        )


def _get_bot_config(bot_key: str) -> SwingBotConfig:
    try:
        from backend.bots import config_store  # type: ignore

        return config_store.get_bot_config(bot_key)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load config for '{bot_key}': {e}",
        )


class EodBotUiConfig(BaseModel):
    """UI-friendly bot rules.

    We keep the bot engine's native SwingBotConfig untouched and layer
    "enabled" + "aggression" as UI-only overrides.
    """

    enabled: bool = True
    aggression: float = 0.50

    # Trading rules
    max_alloc: float = 0.0  # maps to SwingBotConfig.max_position_size
    max_positions: int = 0
    stop_loss: float = 0.0  # maps to SwingBotConfig.stop_pct
    take_profit: float = 0.0  # maps to SwingBotConfig.take_profit_pct
    min_confidence: float = 0.0
    initial_cash: float = 0.0


def _ui_config_from_core(bot_key: str, cfg: SwingBotConfig) -> Dict[str, Any]:
    ov = _load_bot_ui_overrides().get(bot_key, {})
    enabled = bool(ov.get("enabled", True))
    aggression = float(ov.get("aggression", 0.50) or 0.50)

    return EodBotUiConfig(
        enabled=enabled,
        aggression=aggression,
        max_alloc=float(getattr(cfg, "max_position_size", 0.0) or 0.0),
        max_positions=int(getattr(cfg, "max_positions", 0) or 0),
        stop_loss=float(getattr(cfg, "stop_pct", 0.0) or 0.0),
        take_profit=float(getattr(cfg, "take_profit_pct", 0.0) or 0.0),
        min_confidence=float(getattr(cfg, "min_confidence", 0.0) or 0.0),
        initial_cash=float(getattr(cfg, "initial_cash", 0.0) or 0.0),
    ).model_dump()


def _apply_ui_config(bot_key: str, cfg: SwingBotConfig, ui: Dict[str, Any]) -> SwingBotConfig:
    # Core-mapped fields
    if "max_alloc" in ui:
        cfg.max_position_size = float(ui.get("max_alloc") or 0.0)
    if "max_positions" in ui:
        cfg.max_positions = int(ui.get("max_positions") or 0)
    if "stop_loss" in ui:
        cfg.stop_pct = float(ui.get("stop_loss") or 0.0)
    if "take_profit" in ui:
        cfg.take_profit_pct = float(ui.get("take_profit") or 0.0)
    if "min_confidence" in ui:
        cfg.min_confidence = float(ui.get("min_confidence") or 0.0)
    if "initial_cash" in ui:
        cfg.initial_cash = float(ui.get("initial_cash") or 0.0)

    # UI-only overrides
    ov = _load_bot_ui_overrides()
    node = dict(ov.get(bot_key, {}) or {})
    if "enabled" in ui:
        node["enabled"] = bool(ui.get("enabled"))
    if "aggression" in ui:
        try:
            node["aggression"] = float(ui.get("aggression"))
        except Exception:
            pass
    if node:
        ov[bot_key] = node
        _save_bot_ui_overrides(ov)
    return cfg


@router.get("/configs")
async def list_eod_bot_configs() -> Dict[str, Any]:
    """Return UI-friendly configs for all swing bots.

    Shape matches the redesigned Bots page:
      { "configs": { "eod_1w": {...}, ... } }
    """
    configs = _load_all_bot_configs()
    _ensure_swing_ui_defaults(configs)
    out = {k: _ui_config_from_core(k, v) for k, v in configs.items()}
    return {"configs": out}


@router.get("/configs/{bot_key}")
async def get_eod_bot_config(bot_key: str) -> Dict[str, Any]:
    """
    Return config for a single bot (e.g. 'eod_1w').
    """
    cfg = _get_bot_config(bot_key)
    _ensure_swing_ui_defaults({bot_key: cfg})
    return _ui_config_from_core(bot_key, cfg)


class EodConfigUpdateRequest(BaseModel):
    bot_key: str
    config: Dict[str, Any]


@router.post("/configs")
async def update_eod_bot_config_ui(payload: EodConfigUpdateRequest) -> Dict[str, Any]:
    """Update swing bot config using the UI shape.

    Body:
      { "bot_key": "eod_1w", "config": { ... ui fields ... } }
    """
    bot_key = str(payload.bot_key).strip()
    if not bot_key:
        raise HTTPException(status_code=400, detail="Missing bot_key")

    configs = _load_all_bot_configs()
    if bot_key not in configs:
        raise HTTPException(status_code=404, detail="Unknown bot_key")

    cfg = configs[bot_key]
    cfg = _apply_ui_config(bot_key, cfg, dict(payload.config or {}))
    configs[bot_key] = cfg
    _save_all_bot_configs(configs)

    return {"bot_key": bot_key, "config": _ui_config_from_core(bot_key, cfg)}


@router.post("/configs/{bot_key}/reset-defaults")
async def reset_eod_bot_defaults(bot_key: str) -> Dict[str, Any]:
    """Reset a swing bot back to DEFAULT_BOT_CONFIGS and clear UI overrides."""
    try:
        from backend.bots import config_store  # type: ignore

        defaults = config_store.DEFAULT_BOT_CONFIGS
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed loading defaults: {e}")

    if bot_key not in defaults:
        raise HTTPException(status_code=404, detail="Unknown bot_key")

    configs = _load_all_bot_configs()
    configs[bot_key] = defaults[bot_key]
    _save_all_bot_configs(configs)

    # Clear UI overrides for this bot
    ov = _load_bot_ui_overrides()
    if bot_key in ov:
        ov.pop(bot_key, None)
        _save_bot_ui_overrides(ov)

    _ensure_swing_ui_defaults({bot_key: configs[bot_key]})
    return {"bot_key": bot_key, "config": _ui_config_from_core(bot_key, configs[bot_key])}


@router.put("/configs/{bot_key}")
async def update_eod_bot_config(bot_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update EOD bot config. Payload can be partial, e.g.:

        {
          "conf_threshold": 0.62,
          "max_positions": 30
        }
    """
    configs = _load_all_bot_configs()
    if bot_key not in configs:
        raise HTTPException(status_code=404, detail="Unknown bot_key")

    cfg = configs[bot_key]
    fields = SwingBotConfig.__dataclass_fields__.keys()

    for k, v in payload.items():
        if k in fields:
            setattr(cfg, k, v)

    configs[bot_key] = cfg
    _save_all_bot_configs(configs)
    return asdict(cfg)

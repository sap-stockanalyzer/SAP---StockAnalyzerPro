# backend/bots/config_store.py
"""
Bot Config Store — AION Analytics

Provides:
    • DEFAULT_BOT_CONFIGS — baked-in defaults (Option A for 1w/2w/4w).
    • load_all_bot_configs() — merge defaults + JSON overrides.
    • get_bot_config(bot_key) — used by strategy_1w/2w/4w.
    • save_all_bot_configs() — used by API to persist changes.

Config file:
    PATHS["ml_data"]/config/bots_config.json

Structure example:
    {
      "eod_1w": {
        "conf_threshold": 0.6,
        "max_positions": 25
      },
      "eod_2w": { ... }
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from backend.core.config import PATHS
from backend.bots.base_swing_bot import SwingBotConfig
from utils.logger import log

ML_DATA = Path(PATHS["ml_data"])
CONFIG_DIR = ML_DATA / "config"
CONFIG_FILE = CONFIG_DIR / "bots_config.json"

# ---------------------------------------------------------------------
# Defaults (Option A)
# ---------------------------------------------------------------------

DEFAULT_BOT_CONFIGS: Dict[str, SwingBotConfig] = {
    "eod_1w": SwingBotConfig(
        horizon="1w",
        bot_key="eod_1w",
        max_positions=20,
        base_risk_pct=0.20,
        conf_threshold=0.52,
        stop_loss_pct=-0.05,
        take_profit_pct=0.10,
        max_weight_per_name=0.15,
        initial_cash=1000.0,
    ),
    "eod_2w": SwingBotConfig(
        horizon="2w",
        bot_key="eod_2w",
        max_positions=25,
        base_risk_pct=0.20,
        conf_threshold=0.52,
        stop_loss_pct=-0.07,
        take_profit_pct=0.15,
        max_weight_per_name=0.15,
        initial_cash=1000.0,
    ),
    "eod_4w": SwingBotConfig(
        horizon="4w",
        bot_key="eod_4w",
        max_positions=30,
        base_risk_pct=0.20,
        conf_threshold=0.52,
        stop_loss_pct=-0.10,
        take_profit_pct=0.20,
        max_weight_per_name=0.15,
        initial_cash=1000.0,
    ),
}


# ---------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------

def _load_raw_overrides() -> Dict[str, dict]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as e:
        log(f"[bot_config] ⚠️ Failed to read {CONFIG_FILE}: {e}")
    return {}


def load_all_bot_configs() -> Dict[str, SwingBotConfig]:
    """
    Merge defaults with user overrides from bots_config.json.
    Unknown keys in the JSON are ignored.
    """
    raw = _load_raw_overrides()
    out: Dict[str, SwingBotConfig] = {}

    for key, default_cfg in DEFAULT_BOT_CONFIGS.items():
        base = asdict(default_cfg)
        override = raw.get(key) or {}
        if not isinstance(override, dict):
            override = {}
        merged = {**base, **override}
        out[key] = SwingBotConfig(**merged)

    return out


def get_bot_config(bot_key: str) -> SwingBotConfig:
    """
    Return config for a given bot_key.
    If no overrides + no default, falls back to a safe 1w-style config.
    """
    all_cfg = load_all_bot_configs()
    if bot_key in all_cfg:
        return all_cfg[bot_key]

    log(f"[bot_config] ⚠️ Unknown bot_key '{bot_key}', using fallback config.")
    return SwingBotConfig(
        horizon="1w",
        bot_key=bot_key,
        max_positions=20,
        base_risk_pct=0.20,
        conf_threshold=0.55,
        stop_loss_pct=-0.05,
        take_profit_pct=0.10,
        max_weight_per_name=0.15,
        initial_cash=100.0,
    )


def save_all_bot_configs(configs: Dict[str, SwingBotConfig]) -> None:
    """
    Persist the provided configs to bots_config.json.
    Intended to be used by the API (bots_router).
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: asdict(v) for k, v in configs.items()}
    try:
        CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"[bot_config] 💾 Saved bot configs → {CONFIG_FILE}")
    except Exception as e:
        log(f"[bot_config] ❌ Failed to save {CONFIG_FILE}: {e}")

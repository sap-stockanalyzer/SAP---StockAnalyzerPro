"""
config.py (ROOT)

Unified configuration + path registry for BOTH:
- backend (EOD/swing engine)
- dt_backend (intraday engine)

Phase A: foundation.
- Provides ROOT, PATHS, DT_PATHS and minimal helpers.
- Does NOT read secrets; those live in admin_keys.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from settings import TIMEZONE

ROOT = Path(__file__).resolve().parent

# ============================================================
#  TIMEZONE
# ============================================================


# ============================================================
#  DATA ROOTS
# ============================================================

DATA_ROOT = ROOT / "data"

RAW_ROOT = DATA_ROOT / "raw"
RAW_DAILY = RAW_ROOT / "daily_bars"
RAW_INTRADAY = RAW_ROOT / "intraday_bars"
RAW_NEWS = RAW_ROOT / "news"
RAW_SOCIAL = RAW_ROOT / "social"
RAW_FUNDAMENTALS = RAW_ROOT / "fundamentals"

UNIVERSE_ROOT = DATA_ROOT / "universe"
CACHE_ROOT = DATA_ROOT / "data_cache"

# ============================================================
#  BRAINS ROOT (NEW CANONICAL LOCATION)
# ============================================================

BRAINS_ROOT = ROOT / "da_brains"
BRAINS_ROOT.mkdir(parents=True, exist_ok=True)

CORE_BRAINS_DIR = BRAINS_ROOT / "core"
CORE_BRAINS_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_BRAIN = CORE_BRAINS_DIR / "rolling_brain.json.gz"
AION_BRAIN = CORE_BRAINS_DIR / "aion_brain.json.gz"

NEWS_BUZZ_BRAIN_DIR = BRAINS_ROOT / "news_n_buzz_brain"
NEWS_BUZZ_BRAIN_DIR.mkdir(parents=True, exist_ok=True)

NEWS_BRAIN_ROLLING = NEWS_BUZZ_BRAIN_DIR / "news_brain_rolling.json.gz"
NEWS_BRAIN_INTRADAY = NEWS_BUZZ_BRAIN_DIR / "news_brain_intraday.json.gz"
NEWS_BRAIN_META = NEWS_BUZZ_BRAIN_DIR / "meta.json"

# ============================================================
#  ROLLING + BACKUPS
# ============================================================

STOCK_CACHE_ROOT = DATA_ROOT / "stock_cache"
STOCK_CACHE_MASTER = STOCK_CACHE_ROOT / "master"

ROLLING_BODY_PATH = BRAINS_ROOT / "rolling_body.json.gz"
ROLLING_NERVOUS_PATH = BRAINS_ROOT / "rolling_nervous.json.gz"
# Backward-compat alias (old name)
ROLLING_PATH = ROLLING_BODY_PATH
ROLLING_BACKUPS = STOCK_CACHE_MASTER / "backups"
NIGHTLY_LOCK_FILE = BRAINS_ROOT / "nightly_job.lock"

DT_ROLLING_BRAIN = BRAINS_ROOT / "dt_rolling.json.gz"

# ============================================================
#  REPLAY PATHS (CANONICAL)
# ============================================================

SWING_REPLAY_ROOT = DATA_ROOT / "replay" / "swing"
SWING_REPLAY_STATE = SWING_REPLAY_ROOT / "replay_state.json"
SWING_REPLAY_LOCK = DATA_ROOT / "replay" / "locks" / "swing.lock"
MAINTENANCE_FLAG = DATA_ROOT / "replay" / "maintenance_mode.json"

# ============================================================
#  ML (NIGHTLY)
# ============================================================

ML_ROOT = ROOT / "ml_data"
ML_MODELS = ML_ROOT / "models"
ML_PREDICTIONS = ML_ROOT / "predictions"
ML_DATASETS = ML_ROOT / "datasets"
ML_TRAINING = ML_ROOT / "training"
ML_DATASET_DAILY = "ml_data/nightly/dataset/training_data_daily.parquet"

ML_LGBM_CACHE = ML_MODELS / "lgbm_cache"
MACRO_DIR = ML_ROOT / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

MACRO_STATE = ML_ROOT / "macro_state.json"
MARKET_STATE = ML_ROOT / "market_state.json"
REGIME_STATE = ML_ROOT / "regime_state.json"

# ============================================================
#  ML (INTRADAY / DT-BACKEND)
# ============================================================

MLDT_ROOT = ROOT / "ml_data_dt"
MLDT_INTRADAY = MLDT_ROOT / "intraday"
MLDT_INTRADAY_DATASETS = MLDT_INTRADAY / "dataset"
MLDT_INTRADAY_MODELS = MLDT_INTRADAY / "models"
MLDT_INTRADAY_PREDICTIONS = MLDT_INTRADAY / "predictions"
MLDT_INTRADAY_REPLAY = MLDT_INTRADAY / "replay"
MLDT_INTRADAY_RAW_DAYS = MLDT_INTRADAY_REPLAY / "raw_days"
MLDT_INTRADAY_REPLAY_RESULTS = MLDT_INTRADAY_REPLAY / "replay_results"

# ============================================================
#  INSIGHTS / REPORTS / ANALYTICS
# ============================================================

INSIGHTS_ROOT = ROOT / "insights"
DASHBOARD_ROOT = DATA_ROOT / "dashboard_cache"

ANALYTICS_ROOT = ROOT / "analytics"
ANALYTICS_PNL = ANALYTICS_ROOT / "pnl"
ANALYTICS_PERFORMANCE = ANALYTICS_ROOT / "performance"

# ============================================================
#  LOGS
# ============================================================

LOGS_ROOT = ROOT / "logs"
LOGS_BACKEND = LOGS_ROOT / "backend"
LOGS_NIGHTLY = LOGS_ROOT / "nightly"
LOGS_SCHEDULER = LOGS_ROOT / "scheduler"
LOGS_INTRADAY = LOGS_ROOT / "intraday"
NIGHTLY_PREDICTIONS_DIR = LOGS_NIGHTLY / "predictions"

# ============================================================
#  NEWS / SENTIMENT
# ============================================================

NEWS_CACHE = DATA_ROOT / "news_cache"
NEWS_DASHBOARD_JSON = NEWS_CACHE / "news_dashboard_latest.json"
SENTIMENT_MAP = NEWS_CACHE / "sentiment_map_latest.json"
SOCIAL_INTEL_FILE = LOGS_ROOT / "social_intel.json"

# ============================================================
#  CLOUD CACHE (SUPABASE)
# ============================================================

CLOUD_CACHE = DATA_ROOT / "cloud_cache"
UPDATES_DIR = DATA_ROOT / "updates"

# ============================================================
#  PATHS DICTIONARY (BACKEND)
# ============================================================

PATHS: Dict[str, Path] = {
    "root": ROOT,

    "raw_daily": RAW_DAILY,
    "raw_intraday": RAW_INTRADAY,
    "raw_news": RAW_NEWS,
    "raw_social": RAW_SOCIAL,
    "raw_fundamentals": RAW_FUNDAMENTALS,
    "fundamentals_raw": RAW_FUNDAMENTALS,

    "universe": UNIVERSE_ROOT,
    "universe_master_file": UNIVERSE_ROOT / "master_universe.json",
    "universe_swing_file": UNIVERSE_ROOT / "swing_universe.json",
    "universe_dt_file": UNIVERSE_ROOT / "dt_universe.json",

    "stock_cache": STOCK_CACHE_ROOT,
    "stock_cache_master": STOCK_CACHE_MASTER,
    "rolling_body": ROLLING_BODY_PATH,
    "rolling_nervous": ROLLING_NERVOUS_PATH,

    "rolling": ROLLING_BODY_PATH,
    "rolling_backups": ROLLING_BACKUPS,
    "nightly_lock": NIGHTLY_LOCK_FILE,

    "da_brains": BRAINS_ROOT,
    "brains_root": BRAINS_ROOT,
    "core_brains": CORE_BRAINS_DIR,

    "rolling_brain": ROLLING_BRAIN,
    "brain": AION_BRAIN,          # legacy
    "aion_brain": AION_BRAIN,    # canonical

    "swing_replay_root": SWING_REPLAY_ROOT,
    "swing_replay_state": SWING_REPLAY_STATE,
    "swing_replay_lock": SWING_REPLAY_LOCK,
    "maintenance_flag": MAINTENANCE_FLAG,

    "dt_rolling_brain": DT_ROLLING_BRAIN,

    "news_brain_dir": NEWS_BUZZ_BRAIN_DIR,
    "news_brain_rolling": NEWS_BRAIN_ROLLING,
    "news_brain_intraday": NEWS_BRAIN_INTRADAY,
    "news_brain_meta": NEWS_BRAIN_META,

    "ml_data": ML_ROOT,
    "ml_models": ML_MODELS,
    "ml_predictions": ML_PREDICTIONS,
    "ml_datasets": ML_DATASETS,
    "ml_training": ML_TRAINING,
    "ML_DATASET_DAILY": ML_DATASET_DAILY,

    "ml_lgbm_cache": ML_LGBM_CACHE,

    "macro": MACRO_DIR,
    "macro_state": MACRO_STATE,
    "market_state": MARKET_STATE,
    "regime_state": REGIME_STATE,

    "ml_data_dt": MLDT_ROOT,
    "ml_dt_intraday": MLDT_INTRADAY,
    "ml_dt_intraday_models": MLDT_INTRADAY_MODELS,
    "ml_dt_intraday_predictions": MLDT_INTRADAY_PREDICTIONS,
    "ml_dt_intraday_datasets": MLDT_INTRADAY_DATASETS,
    "ml_dt_intraday_replay": MLDT_INTRADAY_REPLAY,
    "ml_dt_intraday_raw_days": MLDT_INTRADAY_RAW_DAYS,
    "ml_dt_intraday_replay_results": MLDT_INTRADAY_REPLAY_RESULTS,

    "insights": INSIGHTS_ROOT,
    "dashboard_cache": DASHBOARD_ROOT,

    "analytics": ANALYTICS_ROOT,
    "analytics_pnl": ANALYTICS_PNL,
    "analytics_performance": ANALYTICS_PERFORMANCE,

    "logs": LOGS_ROOT,
    "backend_logs": LOGS_BACKEND,
    "nightly_logs": LOGS_NIGHTLY,
    "scheduler_logs": LOGS_SCHEDULER,
    "intraday_logs": LOGS_INTRADAY,
    "nightly_predictions": NIGHTLY_PREDICTIONS_DIR,

    "news_cache": NEWS_CACHE,
    "news_dashboard_json": NEWS_DASHBOARD_JSON,
    "sentiment_map": SENTIMENT_MAP,
    "social_intel": SOCIAL_INTEL_FILE,

    "cloud_cache": CLOUD_CACHE,
    "updates": UPDATES_DIR,

# ============================================================
#  MODEL / TRAINING SETTINGS
# ============================================================
  "max_train_rows": 800000,
  "train_batch_rows": 100000,
  "max_features": 180

}


# ============================================================
#  BOT / UI CONFIG FILES (CANONICAL)
# ============================================================

# These are UI + bot runtime configs consumed by the bots pages.
# We keep them under ml_data/config so they're easy to snapshot + sync.

_BOT_CONFIG_DIR = ML_ROOT / "config"
_BOT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PATHS.setdefault("bots_config", _BOT_CONFIG_DIR / "bots_config.json")
PATHS.setdefault("bots_ui_overrides", _BOT_CONFIG_DIR / "bots_ui_overrides.json")
PATHS.setdefault("nightly_summary", LOGS_NIGHTLY / "last_nightly_summary.json")

# ============================================================
#  AUTO-CREATE DIRECTORIES
# ============================================================

def _create_dirs():
    for _, path in PATHS.items():
        if isinstance(path, Path) and path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)

_create_dirs()

# Ensure LGBM cache exists even if it has suffixless path logic
try:
    ML_LGBM_CACHE.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# ============================================================
#  HELPER
# ============================================================

def get_path(key: str) -> Path:
    return PATHS.get(key)

# ============================================================
#  DT_BACKEND PATHS DICTIONARY (DT)
# ============================================================

# ============================================================
#  DATA ROOTS (DT)
# ============================================================
DT_BACKEND: Path = ROOT / "dt_backend"
ML_DATA_DT: Path = ROOT / "ml_data_dt"
LOGS_DT: Path = ROOT / "logs" / "dt_backend"
DA_BRAINS: Path = ROOT / "da_brains"
DATA_DT: Path = ROOT / "data_dt"
# ============================================================
#  PATHS DICTIONARY (DT)
# ============================================================

DT_PATHS: Dict[str, Path] = {
    "root": ROOT,
    "dt_backend": DT_BACKEND,
    "da_brains": DA_BRAINS,
    "data_dt": DATA_DT,
    "core": DT_BACKEND / "core",

    "universe_dir": UNIVERSE_ROOT,
    "universe_file": UNIVERSE_ROOT / "dt_universe.json",
    "legacy_universe_dir": DT_BACKEND / "universe",
    "legacy_universe_file": DT_BACKEND / "universe" / "symbol_universe.json",
    "exchanges_file": DT_BACKEND / "universe" / "exchanges.json",

    "bars_intraday_dir": ML_DATA_DT / "bars" / "intraday",
    "bars_daily_dir": ML_DATA_DT / "bars" / "daily",

    "rolling_intraday_dir": DA_BRAINS / "intraday",
    # DT engine rolling (context/features/predictions/policy/execution)
    "rolling": DA_BRAINS / "intraday" / "rolling_intraday.json.gz",
    "rolling_intraday_file": DA_BRAINS / "intraday" / "rolling_intraday.json.gz",
    # Live market bars rolling (written by live_market_data_loop only)
    "rolling_market_intraday_file": DA_BRAINS / "intraday" / "rolling_intraday_market.json.gz",
    # Lock file used when dt jobs write rolling_intraday_file on Windows
    "rolling_dt_lock_file": DA_BRAINS / "intraday" / ".rolling_intraday_dt.lock",
    "rolling_longterm_dir": DT_BACKEND / "rolling" / "longterm",

    # DT brain (durable learning / performance memory)
    "dt_brain_file": DA_BRAINS / "core" / "dt_brain.json.gz",

    "signals_intraday_dir": ML_DATA_DT / "signals" / "intraday",
    "signals_intraday_predictions_dir": DATA_DT / "signals" / "intraday" / "predictions",
    "signals_intraday_ranks_dir": ML_DATA_DT / "signals" / "intraday" / "ranks",
    "signals_intraday_boards_dir": ML_DATA_DT / "signals" / "intraday" / "boards",

    "signals_longterm_dir": ML_DATA_DT / "signals" / "longterm",
    "signals_longterm_predictions_dir": ML_DATA_DT / "signals" / "longterm" / "predictions",
    "signals_longterm_boards_dir": ML_DATA_DT / "signals" / "longterm" / "boards",

    "historical_replay_root": DATA_DT / "historical_replay",
    "historical_replay_raw": DATA_DT / "historical_replay" / "raw",
    "historical_replay_processed": DATA_DT / "historical_replay" / "processed",
    "historical_replay_meta": DATA_DT / "historical_replay" / "metadata.json",

    "ml_data_dt": ML_DATA_DT,
    "dtml_data": ML_DATA_DT,
    "dtml_intraday_dataset": ML_DATA_DT / "training_data_intraday.parquet",
    "dtmodels": ML_DATA_DT / "models",

    "models_root": DT_BACKEND / "models",
    "models_lgbm_intraday_dir": DT_BACKEND / "models" / "lightgbm_intraday",
    "models_lstm_intraday_dir": DT_BACKEND / "models" / "lstm_intraday",
    "models_transformer_intraday_dir": DT_BACKEND / "models" / "transformer_intraday",
    "models_ensemble_dir": DT_BACKEND / "models" / "ensemble",

    "logs_dt": LOGS_DT,
}


# Add canonical DT UI + summary paths (used by bots UI + diagnostics)
try:
    DT_PATHS.update({
        "intraday_ui_store": ML_DATA_DT / "config" / "intraday_bots_ui.json",
        "sim_summary": ML_DATA_DT / "sim_summary.json",
    })
except Exception:
    pass


def ensure_dt_dirs() -> None:
    """Best-effort directory creation. Never raises."""
    for _, path in DT_PATHS.items():
        try:
            target = path if path.suffix == "" else path.parent
            target.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue


ensure_dt_dirs()


def get_dt_path(key: str) -> Path:
    """Mirror backend.get_path()."""
    return DT_PATHS.get(key)


# ============================================================
#  PROJECT STRUCTURE BOOTSTRAP
# ============================================================

def ensure_project_structure() -> None:
    """Create all required directories and baseline files for bots + UI.

    Safe to call on every startup.
    """
    try:
        _create_dirs()
    except Exception:
        pass
    try:
        ensure_dt_dirs()
    except Exception:
        pass

    # Create empty JSON stores if missing (never overwrite)
    for k, default_obj in [
        ("bots_config", {}),
        ("bots_ui_overrides", {}),
    ]:
        try:
            p = Path(PATHS[k])
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text(json.dumps(default_obj, indent=2), encoding="utf-8")
        except Exception:
            continue

    # Intraday UI store
    try:
        p = Path(DT_PATHS.get("intraday_ui_store", ROOT / "ml_data_dt" / "config" / "intraday_bots_ui.json"))
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(json.dumps({"bots": {}}, indent=2), encoding="utf-8")
    except Exception:
        pass


def describe_paths() -> Dict[str, str]:
    """Resolved paths map for diagnostics/debug."""
    out: Dict[str, str] = {}
    for d in (PATHS, DT_PATHS):
        for k, v in d.items():
            try:
                out[k] = str(Path(v).resolve())
            except Exception:
                out[k] = str(v)
    return out


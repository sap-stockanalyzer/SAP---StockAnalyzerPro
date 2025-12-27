"""
AION Analytics — Central Configuration Module
----------------------------------------------

This file defines ALL backend paths and environment configuration used by:

    • backend/core (models, policy, context, regime, learning)
    • backend/services (fetchers, builders, logs, insights)
    • backend/jobs (nightly, intraday, system)
    • backend/routers (API responses)
    • dt_backend bridging modules

Rules:
    ✔ No hard-coded paths
    ✔ Everything derives from PROJECT ROOT
    ✔ Safe on Windows (no fcntl)
    ✔ Mirrors dt_backend path style (DT_PATHS)
    ✔ Includes all storage needed for nightly + intraday engines
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict
import pytz

# ============================================================
#  KEYS (env-driven, UI-editable)
# ============================================================

KEYS: Dict[str, str] = {
    # --- Supabase ---
    "SUPABASE_URL": os.getenv("SUPABASE_URL", ""),
    "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY", ""),
    "SUPABASE_BUCKET": os.getenv("SUPABASE_BUCKET", ""),

    # --- Trading / Market Data ---
    "ALPACA_API_KEY_ID": os.getenv("ALPACA_API_KEY_ID", ""),
    "ALPACA_API_SECRET_KEY": os.getenv("ALPACA_API_SECRET_KEY", ""),
    "ALPACA_PAPER_BASE_URL": os.getenv("ALPACA_PAPER_BASE_URL", ""),
    "FRED_API": os.getenv("FRED_API", ""),

    "PERIGON_KEY": os.getenv("PERIGON_KEY", ""),
    "MARKETAUX_API_KEY": os.getenv("MARKETAUX_API_KEY", ""),
    "FINNHUB_API_KEY": os.getenv("FINNHUB_API_KEY", ""),
    "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
    "RSS2JSON_KEY": os.getenv("RSS2JSON_KEY", ""),
    "TWITTER_BEARER": os.getenv("TWITTER_BEARER", ""),

    # --- Reddit ---
    "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID", ""),
    "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET", ""),
    "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT", ""),

    # --- Massive / Flatfiles ---
    "MASSIVE_API": os.getenv("MASSIVE_API", ""),
    "MASSIVE_S3API": os.getenv("MASSIVE_S3API", ""),
    "MASSIVE_S3API_SECRET": os.getenv("MASSIVE_S3API_SECRET", ""),
    "MASSIVE_S3_ENDPOINT": os.getenv("MASSIVE_S3_ENDPOINT", ""),
    "MASSIVE_S3_BUCKET": os.getenv("MASSIVE_S3_BUCKET", ""),
}

# ============================================================
#  Module-level aliases (contract adapters)
# ============================================================

SUPABASE_URL = KEYS["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = KEYS["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_ANON_KEY = KEYS["SUPABASE_ANON_KEY"]
SUPABASE_BUCKET = KEYS["SUPABASE_BUCKET"]

ALPACA_API_KEY_ID = KEYS["ALPACA_API_KEY_ID"]
ALPACA_KEY = KEYS["ALPACA_API_KEY_ID"]
ALPACA_API_SECRET_KEY = KEYS["ALPACA_API_SECRET_KEY"]
ALPACA_SECRET = KEYS["ALPACA_API_SECRET_KEY"]
ALPACA_PAPER_BASE_URL = KEYS["ALPACA_PAPER_BASE_URL"]
FRED_API = KEYS["FRED_API"]

PERIGON_KEY = KEYS["PERIGON_KEY"]
MARKETAUX_API_KEY = KEYS["MARKETAUX_API_KEY"]
FINNHUB_API_KEY = KEYS["FINNHUB_API_KEY"]
NEWSAPI_KEY = KEYS["NEWSAPI_KEY"]
RSS2JSON_KEY = KEYS["RSS2JSON_KEY"]
TWITTER_BEARER = KEYS["TWITTER_BEARER"]
MARKETAUX_KEY = KEYS["MARKETAUX_API_KEY"]

REDDIT_CLIENT_ID = KEYS["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = KEYS["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT = KEYS["REDDIT_USER_AGENT"]

MASSIVE_API = KEYS["MASSIVE_API"]
MASSIVE_S3API = KEYS["MASSIVE_S3API"]
MASSIVE_S3API_SECRET = KEYS["MASSIVE_S3API_SECRET"]
MASSIVE_S3_ENDPOINT = KEYS["MASSIVE_S3_ENDPOINT"]
MASSIVE_S3_BUCKET = KEYS["MASSIVE_S3_BUCKET"]

# ============================================================
#  PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# ============================================================
#  TIMEZONE
# ============================================================

TIMEZONE = pytz.timezone(os.getenv("AION_TZ", "America/Denver"))

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

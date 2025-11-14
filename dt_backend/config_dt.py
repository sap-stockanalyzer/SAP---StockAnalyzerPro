# dt_backend/config_dt.py — v1.1 (Intraday Config)
from pathlib import Path

# Root-level path reference (sap_env)
BASE_DIR = Path(__file__).resolve().parents[1]

# Intraday paths — fully separate from nightly backend
_DT_ROOT = BASE_DIR / "ml_data_dt"
_DT_MODEL_ROOT = _DT_ROOT / "models"
_DT_SIGNAL_ROOT = _DT_ROOT / "signals"
_DT_SIM_ROOT = _DT_ROOT / "sim_broker"
_DT_LOG_ROOT = _DT_ROOT / "logs"

DT_PATHS = {
    # === Intraday Data ===
    "dtnews": _DT_ROOT / "news",
    "dtmacro": _DT_ROOT / "macro",
    "dtml_data": _DT_ROOT,
    "dtsignals": _DT_SIGNAL_ROOT,
    "dtmodel_root": _DT_MODEL_ROOT,
    "dtmodels": _DT_MODEL_ROOT / "intraday",
    "dtlogs": _DT_LOG_ROOT,
    "dtbroker": _DT_SIM_ROOT,
    "stock_cache": BASE_DIR / "data_dt" / "stock_cache",
    "macro": BASE_DIR / "data_dt" / "macro_cache",
    # === Rolling, Brain, Metrics ===
    "dt_rolling": BASE_DIR / "data_dt" / "rolling_intraday.json.gz",
    "dtrolling": BASE_DIR / "data_dt" / "rolling_intraday.json.gz",
    "dtbrain": BASE_DIR / "data_dt" / "brain_intraday.json.gz",
    "dtmetrics": BASE_DIR / "data_dt" / "metrics_intraday.json",
    "dtdata": BASE_DIR / "data_dt",
}

# Ensure required directories exist (skip the flat files)
_REQUIRED_DIRS = {
    "dtnews",
    "dtmacro",
    "dtml_data",
    "dtsignals",
    "dtmodel_root",
    "dtmodels",
    "dtlogs",
    "dtbroker",
    "stock_cache",
    "macro",
}

for key in _REQUIRED_DIRS:
    path = DT_PATHS[key]
    path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Intraday Scheduler Settings
# ------------------------------------------------------------
FETCH_SPEED_FACTOR = 8  # ⚡ Lightning-fast polling rate (1=normal, higher=faster)


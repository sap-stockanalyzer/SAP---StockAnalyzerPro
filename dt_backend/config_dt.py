# dt_backend/config_dt.py — v1.1 (Intraday Config)
from pathlib import Path

# Root-level path reference (sap_env)
BASE_DIR = Path(__file__).resolve().parents[1]

# Intraday paths — fully separate from nightly backend
DT_PATHS = {
    # === Intraday Data ===
    "dtnews": BASE_DIR / "ml_data_dt" / "news",
    "dtmacro": BASE_DIR / "ml_data_dt" / "macro",
    "dtml_data": BASE_DIR / "ml_data_dt",
    "dtsignals": BASE_DIR / "ml_data_dt" / "signals",
    "dtmodels": BASE_DIR / "ml_data_dt" / "models" / "intraday",
    "dtlogs": BASE_DIR / "ml_data_dt" / "logs",
    "stock_cache": BASE_DIR / "data_dt" / "stock_cache",
    "macro": BASE_DIR / "data_dt" / "macro_cache",
    # === Rolling, Brain, Metrics ===
    "dt_rolling": BASE_DIR / "data_dt" / "rolling_intraday.json.gz",
    "dtrolling": BASE_DIR / "data_dt" / "rolling_intraday.json.gz",
    "dtbrain": BASE_DIR / "data_dt" / "brain_intraday.json.gz",
    "dtmetrics": BASE_DIR / "data_dt" / "metrics_intraday.json",
    "dtdata": BASE_DIR / "data_dt",
}

# Ensure required directories exist
for key, path in DT_PATHS.items():
    if "json" not in str(path):
        path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Intraday Scheduler Settings
# ------------------------------------------------------------
FETCH_SPEED_FACTOR = 8  # ⚡ Lightning-fast polling rate (1=normal, higher=faster)


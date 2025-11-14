"""
continuous_learning.py — v2.1 (Phase-1 Adaptive Trainer + Key Normalization)
AION Analytics / StockAnalyzerPro

Purpose:
Implements the self-learning feedback loop:
    Predictions → Outcomes → Incremental Training → Calibration → Brain Update

Major changes:
✅ Auto-normalizes all feature columns to snake_case.
✅ Uses ops_helpers.harvest_outcomes() + calibrate_confidence().
✅ Produces versioned incremental models (vYYYYMMDD_HHMMSS).
✅ Updates Rolling Brain with calibration metrics.
"""

from __future__ import annotations
import os, json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------
try:
    from backend.ops_helpers import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg)

# ---------------------------------------------------------------------
# Core utilities and imports
# ---------------------------------------------------------------------
from backend.ops_helpers import (
    harvest_outcomes,
    calibrate_confidence,
    _read_brain,
    save_brain,
)

# ✅ Unified config paths
try:
    from .config import PATHS  # type: ignore
except Exception:
    PATHS = {"ml_data": Path("ml_data")}  # safe fallback

# ---------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------
NORMALIZE_KEYS = {
    "peRatio": "pe_ratio", "pbRatio": "pb_ratio", "psRatio": "ps_ratio",
    "pegRatio": "peg_ratio", "debtEquity": "debt_equity", "debtEbitda": "debt_ebitda",
    "revenueGrowth": "revenue_growth", "epsGrowth": "eps_growth",
    "profitMargin": "profit_margin", "operatingMargin": "operating_margin",
    "grossMargin": "gross_margin", "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio", "marketCap": "marketCap"
}

def normalize_keys(df_or_dict):
    """Normalizes keys/columns from camelCase → snake_case (in-place)."""
    if df_or_dict is None:
        return df_or_dict

    # If dict (Rolling node, metadata, etc.)
    if isinstance(df_or_dict, dict):
        for old, new in NORMALIZE_KEYS.items():
            if old in df_or_dict and new not in df_or_dict:
                df_or_dict[new] = df_or_dict.pop(old)
        return df_or_dict

    # If DataFrame
    try:
        import pandas as pd
        if isinstance(df_or_dict, pd.DataFrame):
            rename_map = {old: new for old, new in NORMALIZE_KEYS.items() if old in df_or_dict.columns}
            return df_or_dict.rename(columns=rename_map)
    except Exception:
        pass

    return df_or_dict


# ---------------------------------------------------------------------
# Phase-1 API functions
# ---------------------------------------------------------------------
def record_predictions(predictions: List[Dict[str, Any]], out_path: str = "metrics_history/predictions.jsonl") -> str:
    """Legacy fallback recorder (still available if needed)."""
    import json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for p in predictions or []:
            p = dict(p)
            p["_ts"] = datetime.utcnow().isoformat()
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"✅ Recorded {len(predictions or [])} predictions → {out_path}")
    return out_path


def update_with_outcomes(window_days: int = 20, pred_log: str = "metrics_history/predictions.jsonl") -> Optional[str]:
    """Stub placeholder (kept for backward compatibility)."""
    import os, json
    if not os.path.exists(pred_log):
        log("ℹ️ No predictions log found. Skipping outcomes update.")
        return None
    resolved = 0
    tmp_out = pred_log.replace(".jsonl", f"_resolved_{datetime.utcnow().strftime('%Y%m%d')}.jsonl")
    with open(pred_log, "r", encoding="utf-8") as fin, open(tmp_out, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj["resolved"] = True
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            resolved += 1
    log(f"🧮 Outcomes updated for {resolved} predictions → {tmp_out}")
    return tmp_out


# ---------------------------------------------------------------------
# ✅ Phase-1 Incremental Trainer
# ---------------------------------------------------------------------
def train_incremental(feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Performs online (incremental) model updates using latest outcomes.

    Steps:
      1. Harvest any new outcomes (ops_helpers.harvest_outcomes)
      2. Load newest prediction/outcome pairs
      3. Retrain LightGBM incrementally (per available features)
      4. Calibrate confidence → update Rolling Brain
    Returns rich status dict for nightly_job.
    """
    import pandas as pd
    from backend.train_lightgbm import train_lightgbm_models
    from backend.ops_helpers import DEFAULT_LOG_DIR, OUTCOMES_DIR

    # ✅ Use unified config paths
    base_dir = PATHS.get("ml_data", Path("ml_data"))
    models_dir = (PATHS.get("models") if isinstance(PATHS, dict) else None) or (base_dir / "models")
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # 1️⃣ Ensure new outcomes exist
    outcome_result = harvest_outcomes()
    if outcome_result.get("status") not in ("ok",):
        log(f"ℹ️ No new outcomes to train from ({outcome_result.get('status')}).")
        return {"status": "no_outcomes"}

    out_file = outcome_result.get("file")
    if not out_file or not os.path.exists(out_file):
        log("⚠️ Outcome file missing after harvest.")
        return {"status": "error", "step": "harvest"}

    # 2️⃣ Load most recent prediction log and outcomes
    try:
        latest_pred = sorted(Path(DEFAULT_LOG_DIR).glob("preds_*.parquet"))[-1]
        preds = pd.read_parquet(latest_pred)
        outs = pd.read_parquet(out_file)
        preds = normalize_keys(preds)
        outs = normalize_keys(outs)
    except Exception as e:
        log(f"⚠️ Failed to read incremental data: {e}")
        return {"status": "read_error", "error": str(e)}

    if preds.empty or outs.empty:
        return {"status": "empty_data"}

    if "symbol" not in preds.columns or "symbol" not in outs.columns:
        return {"status": "invalid_schema"}

    # 3️⃣ Merge predictions ↔ outcomes
    df = preds.merge(outs, on="symbol", how="inner", suffixes=("_pred", "_real"))
    if df.empty:
        return {"status": "no_overlap"}

    # Normalize columns once more after merge
    df = normalize_keys(df)

    # Target selection
    y = df.get("realized_return_pct")
    if y is None or y.isna().all():
        log("⚠️ No valid realized returns found.")
        return {"status": "no_targets"}

    # Feature selection
    feature_cols = feature_cols or [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        # fallback: use numeric columns that aren’t meta
        feature_cols = [c for c in df.select_dtypes("number").columns if c not in ("y_pred", "proba", "realized_return_pct")]

    if not feature_cols:
        log("⚠️ No suitable feature columns detected.")
        return {"status": "no_features"}

    # 4️⃣ Train LightGBM incremental model
    try:
        import lightgbm as lgb
        X = df[feature_cols]
        y = y.astype(float) / 100.0  # scale to fractional
        reg = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        reg.fit(X, y)
        model_version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
        out_path = Path(models_dir) / f"incremental_{model_version}.txt"
        reg.booster_.save_model(str(out_path))
        log(f"✅ Incremental model retrained → {out_path}")
    except Exception as e:
        log(f"⚠️ Incremental LightGBM training failed: {e}")
        return {"status": "train_error", "error": str(e)}

    # 5️⃣ Calibrate confidence and update Rolling Brain
    brain = _read_brain() or {}
    brain.setdefault("model_versions", []).append(model_version)
    brain = calibrate_confidence(brain)
    save_brain(brain)

    log(
        f"🧠 Brain updated → version={model_version}, "
        f"mult={brain.get('calibration',{}).get('confidence_multiplier')}, "
        f"hit={brain.get('calibration',{}).get('hit_rate')}, "
        f"brier={brain.get('calibration',{}).get('brier')}"
    )

    return {
        "status": "ok",
        "model_version": model_version,
        "trained_rows": len(X),
        "features": len(feature_cols),
        "calibration": brain.get("calibration"),
        "outcome_file": out_file,
    }

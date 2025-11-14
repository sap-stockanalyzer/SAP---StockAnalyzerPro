# dt_backend/ai_model_intraday.py — v1.2 (AION Intraday Model Scoring)
"""
Loads trained intraday LightGBM model + feature list,
scores the latest snapshot per symbol, and writes predictions to:
    ml_data_dt/signals/intraday_predictions.json
"""

from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------
# Import shim + logger
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from dt_backend.dt_logger import dt_log as log
except Exception:
    def log(msg: str): print(msg, flush=True)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
from dt_backend.config_dt import DT_PATHS

DATA_PATH = DT_PATHS["dtml_data"] / "training_data_intraday.parquet"
MODEL_DIR = DT_PATHS["dtmodels"]
SIGNALS_DIR = DT_PATHS["dtsignals"]
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = SIGNALS_DIR / "intraday_predictions.json"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_artifacts():
    """Load model, feature list, and label map from DT model directory."""
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:
        log(f"[ai_intraday] ⚠️ LightGBM not available: {e}")
        return None, None, None

    model_path = MODEL_DIR / "model.txt"
    feats_path = MODEL_DIR / "feature_list.json"
    label_map_path = MODEL_DIR / "label_map.json"

    if not model_path.exists() or not feats_path.exists() or not label_map_path.exists():
        log("[ai_intraday] ⚠️ model/feature_map/label_map missing — train first.")
        return None, None, None

    booster = lgb.Booster(model_file=str(model_path))
    with open(feats_path, "r", encoding="utf-8") as f:
        feature_list = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return booster, feature_list, label_map


def _load_dataset() -> pd.DataFrame | None:
    """Load the most recent intraday parquet dataset."""
    if not DATA_PATH.exists():
        log(f"[ai_intraday] ⚠️ dataset missing: {DATA_PATH}")
        return None
    try:
        return pd.read_parquet(DATA_PATH)
    except Exception as e:
        log(f"[ai_intraday] ⚠️ failed to read parquet: {e}")
        return None


def _latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the last available bar per symbol."""
    if "symbol" not in df.columns:
        return pd.DataFrame()
    idx = df.groupby("symbol").tail(1).index
    return df.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------
# Main Scoring Function
# ---------------------------------------------------------------------
def score_intraday_tickers() -> Dict[str, Any]:
    """Score all available tickers and write compact prediction file."""
    booster, feature_list, label_map = _load_artifacts()
    if booster is None:
        return {}

    df = _load_dataset()
    if df is None or df.empty:
        return {}

    snap = _latest_snapshot(df)
    if snap.empty:
        log("[ai_intraday] ⚠️ no latest snapshots found")
        return {}

    feature_list = feature_list or [
        c for c in snap.columns if c not in {"symbol", "timestamp", "split", "target_label_15m", "target_ret_15m"}
    ]
    # Prepare features in the same order
    X = snap.reindex(columns=feature_list).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    pred_proba = booster.predict(X.values)
    classes = label_map.get("LABEL_ORDER", ["SELL", "HOLD", "BUY"])
    id2label = {i: c for i, c in enumerate(classes)}

    results: Dict[str, dict] = {}
    for i, row in snap.iterrows():
        sym = str(row.get("symbol"))
        probs = pred_proba[i]
        score_val = None
        if isinstance(probs, (list, tuple, np.ndarray)):
            probs = np.asarray(probs, dtype=float)
            j = int(np.argmax(probs))
            conf = float(np.max(probs))
            label = id2label.get(j, "HOLD")
            buy_idx = label_map.get("LABEL2ID", {}).get("BUY")
            sell_idx = label_map.get("LABEL2ID", {}).get("SELL")
            if buy_idx is not None and buy_idx < probs.size:
                buy_prob = float(probs[buy_idx])
            else:
                buy_prob = 0.0
            if sell_idx is not None and sell_idx < probs.size:
                sell_prob = float(probs[sell_idx])
            else:
                sell_prob = 0.0
            score_val = buy_prob - sell_prob
        else:
            label, conf = "HOLD", 0.34

        node = {
            "symbol": sym,
            "timestamp": str(row.get("timestamp")),
            "label": label,
            "confidence": conf,
            "predicted": score_val if score_val is not None else conf,
            "score": score_val if score_val is not None else conf,
            "currentPrice": float(row.get("close", np.nan)) if pd.notna(row.get("close")) else None,
            "momentum_score": float(row.get("momentum_score", np.nan)) if "momentum_score" in row else None,
            "orderflow_score": float(row.get("orderflow_score", np.nan)) if "orderflow_score" in row else None,
            "probabilities": {
                id2label.get(idx, str(idx)): float(probs[idx]) if isinstance(probs, np.ndarray) and idx < probs.size else None
                for idx in range(len(classes))
            } if isinstance(probs, np.ndarray) else None,
        }
        results[sym] = node

    # -----------------------------------------------------------------
    # Save compact signal map
    # -----------------------------------------------------------------
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"generated_at": datetime.utcnow().isoformat(), "rows": list(results.values())},
                f,
                indent=2,
            )
        log(f"[ai_intraday] ✅ signals written → {OUT_PATH} ({len(results)} symbols)")
    except Exception as e:
        log(f"[ai_intraday] ⚠️ failed to write signals: {e}")

    return results


# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    out = score_intraday_tickers()
    print(f"scored: {len(out)}")

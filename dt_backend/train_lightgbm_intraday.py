# dt_backend/train_lightgbm_intraday.py — v1.0
# Trains a fast 3-class classifier (SELL/HOLD/BUY) on intraday features.
# Saves model + feature list under: ml_data_dt/models/intraday/

from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Allow dt_backend to import backend helpers when run standalone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from dt_backend.dt_logger import dt_log as log
except Exception:
    def log(msg: str): print(msg, flush=True)

from dt_backend.config_dt import DT_PATHS

DATA_PATH = DT_PATHS["dtml_data"] / "training_data_intraday.parquet"
MODEL_DIR = DT_PATHS["dtmodels"]
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Label mapping for 3-class classification
LABEL_ORDER = ["SELL", "HOLD", "BUY"]
LABEL2ID = {c: i for i, c in enumerate(LABEL_ORDER)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

def _load_dataset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log(f"[train_intraday] ⚠️ dataset missing: {path}")
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            log("[train_intraday] ⚠️ dataset empty")
            return None
        return df
    except Exception as e:
        log(f"[train_intraday] ⚠️ failed to read parquet: {e}")
        return None

def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    # drop obvious non-features & targets
    drop_cols = {"symbol", "timestamp", "target_ret_15m", "target_label_15m", "split"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].select_dtypes(include=["float64", "float32", "int64", "int32"]).copy()

    # align y
    y_raw = (df["target_label_15m"].astype(str).str.upper()
             if "target_label_15m" in df.columns else pd.Series("HOLD", index=df.index))
    y = y_raw.map(lambda s: LABEL2ID.get(s, 1))  # default HOLD=1
    mask = y.notna() & np.isfinite(X.fillna(0)).all(axis=1)
    X = X.loc[mask].fillna(0)
    y = y.loc[mask].astype(int)

    if "split" in df.columns:
        train_mask = df["split"].str.lower() == "train"
        valid_mask = ~train_mask
        if valid_mask.sum() == 0:
            valid_mask.iloc[-int(len(df) * 0.1):] = True
        X_train, X_valid = X[train_mask], X[valid_mask]
        y_train, y_valid = y[train_mask], y[valid_mask]
    else:
        # chronological fallback
        n = len(y)
        split = max(50, int(n * 0.8))
        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]

    return (X_train, y_train, X_valid, y_valid, feature_cols)

def train_intraday_models() -> Dict[str, Any]:
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:
        log(f"[train_intraday] ⚠️ LightGBM not available: {e}")
        return {"status": "skipped_no_lgbm"}

    df = _load_dataset(DATA_PATH)
    if df is None:
        return {"status": "skipped_no_data"}

    X_tr, y_tr, X_va, y_va, feat_cols = _prepare(df)
    if X_tr.empty or X_va.empty:
        log("[train_intraday] ⚠️ insufficient data after filtering")
        return {"status": "skipped_insufficient"}

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=63,
        objective="multiclass",
        class_weight=None,
    )
    clf.fit(X_tr, y_tr)

    # basic metrics
    acc = float((clf.predict(X_va) == y_va).mean())
    try:
        from sklearn.metrics import f1_score
        f1_macro = float(f1_score(y_va, clf.predict(X_va), average="macro"))
    except Exception:
        f1_macro = None

    # persist model + features + label map
    try:
        clf.booster_.save_model(str(MODEL_DIR / "model.txt"))
        with open(MODEL_DIR / "feature_list.json", "w", encoding="utf-8") as f:
            json.dump(feat_cols, f, indent=2)
        with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
            json.dump({"LABEL_ORDER": LABEL_ORDER, "LABEL2ID": LABEL2ID, "ID2LABEL": ID2LABEL}, f, indent=2)
        log(f"[train_intraday] 💾 saved model → {MODEL_DIR / 'model.txt'}")
    except Exception as e:
        log(f"[train_intraday] ⚠️ failed to save model: {e}")

    metrics = {
        "status": "ok",
        "n_train": int(len(y_tr)),
        "n_valid": int(len(y_va)),
        "acc": acc,
        "f1_macro": f1_macro,
        "n_features": int(len(feat_cols)),
        "model_dir": str(MODEL_DIR),
    }

    try:
        from backend.ml_helpers import register_model  # type: ignore

        register_model(
            "intraday_lightgbm",
            {k: v for k, v in metrics.items() if k in {"acc", "f1_macro", "n_train", "n_valid"}},
            feat_cols,
        )
    except Exception:
        pass

    return metrics

if __name__ == "__main__":
    out = train_intraday_models()
    print(out)

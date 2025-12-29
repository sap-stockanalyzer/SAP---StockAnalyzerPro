"""Train a fast 3-class LightGBM intraday model.

Expects a parquet built by ml_data_builder_intraday.py with:
  - feature columns
  - label column: 'label' (SELL/HOLD/BUY) or 'label_id' (0/1/2)

Saves artifacts under: DT_PATHS["dtmodels"] / "lightgbm_intraday"
  - model.txt
  - feature_map.json
  - label_map.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS: Dict[str, Path] = {
        "dtml_data": Path("ml_data_dt"),
        "dtmodels": Path("dt_backend") / "models",
    }

from dt_backend.models import LABEL_ORDER, LABEL2ID, ID2LABEL, get_model_dir

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


def _resolve_training_data() -> Path:
    # Accept multiple keys to match refactors
    for key in ("dtml_data", "ml_data_dt", "ml_data"):
        base = DT_PATHS.get(key)
        if base:
            return Path(base) / "training_data_intraday.parquet"
    return Path("ml_data_dt") / "training_data_intraday.parquet"


def _encode_non_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Convert object/category/datetime columns to numeric codes.

    This keeps training resilient if any feature is categorical (e.g. vol_bucket).
    """
    X2 = X.copy()
    min_i64 = np.iinfo("int64").min

    for c in list(X2.columns):
        s = X2[c]

        # Datetimes -> int64 (nan-safe). Avoid Series.view() deprecation warnings.
        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                vals = s.astype("int64")
                # Pandas encodes NaT as int64 min; map that to 0.
                vals = vals.where(vals != min_i64, 0)
                X2[c] = vals.astype(np.int64)
            except Exception:
                X2[c] = 0
            continue

        # Object/categorical -> category codes (>=0)
        is_cat = isinstance(s.dtype, pd.CategoricalDtype)
        if pd.api.types.is_object_dtype(s) or is_cat:
            codes = pd.Categorical(s).codes.astype(np.int32)
            codes = np.where(codes < 0, 0, codes)
            X2[c] = codes
            continue

    # Ensure everything is numeric
    for c in list(X2.columns):
        if not pd.api.types.is_numeric_dtype(X2[c]) and not pd.api.types.is_bool_dtype(X2[c]):
            X2 = X2.drop(columns=[c])

    X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X2


def _load_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    path = _resolve_training_data()
    if not path.exists():
        raise FileNotFoundError(f"Intraday training data not found at {path}")

    log(f"[train_lightgbm_intraday] 📦 Loading training data from {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Training dataframe at {path} is empty.")

    if "label_id" in df.columns:
        y = df["label_id"].astype(int)
    elif "label" in df.columns:
        y = df["label"].map(LABEL2ID)
        if y.isna().any():
            bad = df["label"][y.isna()].unique().tolist()
            raise ValueError(f"Unknown labels in training data: {bad}")
        y = y.astype(int)
    else:
        raise ValueError("Training data must contain 'label' or 'label_id' column.")

    # Drop non-feature columns
    drop_cols = [c for c in ("label", "label_id", "symbol") if c in df.columns]
    X = df.drop(columns=drop_cols)

    # ts can be useful, but encode to int. If missing it's fine.
    X = _encode_non_numeric(X)

    # Ensure we have enough label variety for multiclass
    uniq = sorted(set(int(v) for v in y.unique().tolist()))
    if len(uniq) < 2:
        raise ValueError(
            f"Not enough label variety for training (unique labels={uniq}). "
            "This usually means your dataset is too small or returns are flat. "
            "Try building with more symbols or during active market hours."
        )

    return X, y


def _train_lgb(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any] | None = None,
) -> lgb.Booster:
    if params is None:
        params = {
            "objective": "multiclass",
            "num_class": len(LABEL_ORDER),
            "metric": ["multi_logloss"],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "seed": 42,
            "verbosity": -1,
        }

    # class weights can help if quantiles aren't perfectly balanced
    try:
        counts = y.value_counts().to_dict()
        total = float(len(y))
        weights = {int(k): total / (len(counts) * float(v)) for k, v in counts.items() if v}
        w = y.map(lambda k: weights.get(int(k), 1.0)).astype(float).values
    except Exception:
        w = None

    dtrain = lgb.Dataset(X, label=y.values, weight=w)
    log(f"[train_lightgbm_intraday] 🚀 Training on {len(X):,} rows, {X.shape[1]} features...")
    # LightGBM version compatibility:
    #   - Some installs (notably newer ones) removed/changed `verbose_eval`.
    #   - Logging is controlled via callbacks instead.
    callbacks = []
    try:
        callbacks.append(lgb.log_evaluation(period=50))
    except Exception:
        callbacks = []

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=callbacks,
    )
    log("[train_lightgbm_intraday] ✅ Training complete.")
    return booster


def _save_artifacts(booster: lgb.Booster, feature_names: List[str]) -> None:
    model_dir = get_model_dir("lightgbm")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.txt"
    fmap_path = model_dir / "feature_map.json"
    label_map_path = model_dir / "label_map.json"

    booster.save_model(str(model_path))
    with fmap_path.open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "label_order": LABEL_ORDER,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log(f"[train_lightgbm_intraday] 💾 Saved model → {model_path}")
    log(f"[train_lightgbm_intraday] 💾 Saved feature_map → {fmap_path}")
    log(f"[train_lightgbm_intraday] 💾 Saved label_map → {label_map_path}")


def train_lightgbm_intraday() -> Dict[str, Any]:
    """High-level entrypoint to train & persist the intraday LightGBM model."""
    try:
        log(f"[train_lightgbm_intraday] ℹ️ Using LightGBM v{getattr(lgb, '__version__', '?')}")
    except Exception:
        pass
    X, y = _load_training_data()
    booster = _train_lgb(X, y)
    _save_artifacts(booster, list(X.columns))
    summary = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "label_order": LABEL_ORDER,
        "labels": sorted(set(int(v) for v in y.unique().tolist())),
    }
    log(f"[train_lightgbm_intraday] 📊 Summary: {summary}")
    return summary


if __name__ == "__main__":
    train_lightgbm_intraday()

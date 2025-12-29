"""Train a fast 3-class LightGBM intraday model (DT backend).

Reads:  ml_data_dt/training_data_intraday.parquet
Writes: dt_backend/models/lightgbm_intraday/
  - model.txt
  - feature_map.json
  - label_map.json

This fixes the common mismatch where training writes to dt_backend/models/lightgbm/
while the runtime loader reads dt_backend/models/lightgbm_intraday/.

Compatible with LightGBM >= 4.x (no verbose_eval kwarg; uses callbacks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import lightgbm as lgb
import pandas as pd

try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS: Dict[str, Path] = {
        "dtml_data": Path("ml_data_dt"),
        "dtmodels": Path("dt_backend") / "models",
    }

from dt_backend.models import LABEL_ORDER, LABEL2ID, ID2LABEL

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:  # pragma: no cover

    def log(msg: str) -> None:
        print(msg, flush=True)


def _resolve_training_data() -> Path:
    base = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    return base / "training_data_intraday.parquet"


def _model_dir_intraday() -> Path:
    base = Path(DT_PATHS.get("dtmodels", Path("dt_backend") / "models"))
    return base / "lightgbm_intraday"


def _load_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    path = _resolve_training_data()
    if not path.exists():
        raise FileNotFoundError(f"Intraday training data not found at {path}")

    log(f"[train_lightgbm_intraday] \U0001F4E6 Loading training data from {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Training dataframe at {path} is empty.")

    # Labels
    if "label_id" in df.columns:
        y = df["label_id"].astype(int)
    elif "label" in df.columns:
        y = df["label"].map(LABEL2ID)
        if y.isna().any():
            bad = df.loc[y.isna(), "label"].unique().tolist()
            raise ValueError(f"Unknown labels in training data: {bad}")
        y = y.astype(int)
    else:
        raise ValueError("Training data must contain 'label' or 'label_id' column.")

    # Features
    drop_cols = [c for c in ("label", "label_id") if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Never train on symbol identifier
    if "symbol" in X.columns:
        X = X.drop(columns=["symbol"])

    # Timestamps: if datetime-like, convert to epoch seconds; if string, try parse.
    if "ts" in X.columns:
        s = X["ts"]
        if pd.api.types.is_datetime64_any_dtype(s):
            X["ts"] = (s.astype("int64") // 1_000_000_000).fillna(0).astype("int64")
        else:
            try:
                dt = pd.to_datetime(s, errors="coerce", utc=True)
                X["ts"] = (dt.astype("int64") // 1_000_000_000).fillna(0).astype("int64")
            except Exception:
                X = X.drop(columns=["ts"])

    # Convert any remaining object/categorical columns to category codes
    for c in list(X.columns):
        col = X[c]
        if pd.api.types.is_object_dtype(col) or isinstance(col.dtype, pd.CategoricalDtype):
            cat = pd.Categorical(col)
            X[c] = pd.Series(cat.codes, index=X.index).astype("int32")

    # Ensure numeric
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        log(
            f"[train_lightgbm_intraday] \u26A0\uFE0F Dropping non-numeric columns: "
            f"{non_numeric[:10]}{'...' if len(non_numeric) > 10 else ''}"
        )
        X = X.drop(columns=non_numeric)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found after preprocessing.")

    return X, y


def _train_lgb(X: pd.DataFrame, y: pd.Series, params: Dict[str, Any] | None = None) -> lgb.Booster:
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
            "min_data_in_leaf": 50,
            "seed": 42,
            "verbosity": -1,
        }

    dtrain = lgb.Dataset(X, label=y.values)
    log(f"[train_lightgbm_intraday] \u2139\uFE0F Using LightGBM v{getattr(lgb, '__version__', '?')}")
    log(f"[train_lightgbm_intraday] \U0001F680 Training on {len(X):,} rows, {X.shape[1]} features...")

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=50)],
    )

    log("[train_lightgbm_intraday] \u2705 Training complete.")
    return booster


def _atomic_write_bytes(path: Path, b: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(b)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _save_artifacts(booster: lgb.Booster, feature_names: list[str]) -> None:
    model_dir = _model_dir_intraday()
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.txt"
    fmap_path = model_dir / "feature_map.json"
    label_map_path = model_dir / "label_map.json"

    # Atomic model save
    tmp_model = model_path.with_suffix(".txt.tmp")
    booster.save_model(str(tmp_model))
    tmp_model.replace(model_path)

    _atomic_write_text(fmap_path, json.dumps(feature_names, ensure_ascii=False, indent=2))
    _atomic_write_text(
        label_map_path,
        json.dumps(
            {"label_order": LABEL_ORDER, "label2id": LABEL2ID, "id2label": ID2LABEL},
            ensure_ascii=False,
            indent=2,
        ),
    )

    log(f"[train_lightgbm_intraday] \U0001F4BE Saved model \u2192 {model_path}")


def train_lightgbm_intraday() -> Dict[str, Any]:
    X, y = _load_training_data()
    booster = _train_lgb(X, y)
    _save_artifacts(booster, list(X.columns))

    summary = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "label_order": LABEL_ORDER,
        "model_dir": str(_model_dir_intraday()),
    }
    log(f"[train_lightgbm_intraday] \U0001F4CA Summary: {summary}")
    return summary


# Back-compat: some parts of dt_backend import train_intraday_models
train_intraday_models = train_lightgbm_intraday


if __name__ == "__main__":
    train_lightgbm_intraday()

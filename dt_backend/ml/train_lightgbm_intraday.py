# dt_backend/ml/train_lightgbm_intraday.py — v1.1 (ATOMIC SAVE + BACKUP + LOCK)
"""Train a fast 3-class LightGBM intraday model.

Writes:
  dt_backend/models/lightgbm_intraday/model.txt
  dt_backend/models/lightgbm_intraday/model.txt.bak

Run:
  python -m dt_backend.ml.train_lightgbm_intraday
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS: Dict[str, Path] = {
        "ml_data_dt": Path("ml_data_dt"),
        "dtmodels": Path("dt_backend") / "models",
    }

from dt_backend.models import LABEL_ORDER, LABEL2ID, ID2LABEL

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


def _model_lock_path(model_dir: Path) -> Path:
    return model_dir / ".model_write.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def _read_lock_pid(lock: Path) -> int:
    try:
        raw = lock.read_text(encoding="utf-8", errors="ignore").strip()
        parts = raw.split()
        return int(parts[0]) if parts else -1
    except Exception:
        return -1


def _acquire_model_lock(lock: Path, timeout_s: float = 180.0) -> bool:
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(0.0, float(timeout_s))

    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                payload = f"{os.getpid()} {time.time():.3f}"
                os.write(fd, payload.encode("utf-8", errors="ignore"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            pid = _read_lock_pid(lock)
            if pid > 0 and not _pid_alive(pid):
                try:
                    lock.unlink(missing_ok=True)  # type: ignore[arg-type]
                    continue
                except Exception:
                    pass
            if time.time() >= deadline:
                return False
            time.sleep(0.2)
        except Exception:
            return False


def _release_model_lock(lock: Path) -> None:
    try:
        lock.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def _resolve_training_data() -> Path:
    root = (
        DT_PATHS.get("ml_data_dt")
        or DT_PATHS.get("dtml_data")
        or DT_PATHS.get("ml_data")
        or Path("ml_data_dt")
    )
    return Path(root) / "training_data_intraday.parquet"


def _coerce_features(X: pd.DataFrame) -> pd.DataFrame:
    X2 = pd.DataFrame(index=X.index)

    for c in X.columns:
        s = X[c]

        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_datetime64tz_dtype(s):
            try:
                if pd.api.types.is_datetime64tz_dtype(s):
                    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
                X2[c] = s.astype("int64").fillna(0).astype(np.int64)
            except Exception:
                X2[c] = 0
            continue

        if pd.api.types.is_bool_dtype(s):
            X2[c] = s.fillna(False).astype(np.int8)
            continue

        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            codes, _ = pd.factorize(s.astype("string"), sort=True)
            X2[c] = pd.Series(codes, index=X.index).astype(np.int32)
            continue

        if pd.api.types.is_numeric_dtype(s):
            X2[c] = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32)
            continue

        X2[c] = pd.to_numeric(s.astype("string"), errors="coerce").fillna(0.0).astype(np.float32)

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

    drop_cols = [c for c in ("label", "label_id") if c in df.columns]
    X = df.drop(columns=drop_cols)

    if "symbol" in X.columns:
        X = X.drop(columns=["symbol"])

    return _coerce_features(X), y


def _train_lgb(X: pd.DataFrame, y: pd.Series, params: Optional[Dict[str, Any]] = None) -> lgb.Booster:
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
    log(f"[train_lightgbm_intraday] ℹ️ Using LightGBM v{getattr(lgb, '__version__', '?')}")
    log(f"[train_lightgbm_intraday] 🚀 Training on {len(X):,} rows, {X.shape[1]} features...")

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=50)],
    )
    log("[train_lightgbm_intraday] ✅ Training complete.")
    return booster


def _resolve_model_dir() -> Path:
    base = DT_PATHS.get("dtmodels") or DT_PATHS.get("dt_models") or (Path("dt_backend") / "models")
    return Path(base) / "lightgbm_intraday"


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _save_model_atomic(booster: lgb.Booster, model_path: Path) -> None:
    bak = model_path.with_name(model_path.name + ".bak")
    tmp = model_path.with_name(model_path.name + ".tmp")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        try:
            bak.write_bytes(model_path.read_bytes())
        except Exception:
            pass

    booster.save_model(str(tmp))
    os.replace(tmp, model_path)

    # Verify it loads (catches truncated garbage)
    _ = lgb.Booster(model_file=str(model_path))


def train_lightgbm_intraday() -> Dict[str, Any]:
    X, y = _load_training_data()
    booster = _train_lgb(X, y)

    model_dir = _resolve_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    lock = _model_lock_path(model_dir)
    if not _acquire_model_lock(lock, timeout_s=float(os.getenv("DT_MODEL_LOCK_TIMEOUT", "180"))):
        raise TimeoutError(f"Timed out acquiring model lock: {lock}")

    try:
        _save_model_atomic(booster, model_dir / "model.txt")
        _atomic_write_json(model_dir / "feature_map.json", list(X.columns))
        _atomic_write_json(
            model_dir / "label_map.json",
            {"label_order": LABEL_ORDER, "label2id": LABEL2ID, "id2label": ID2LABEL},
        )
    finally:
        _release_model_lock(lock)

    summary = {"n_rows": int(len(X)), "n_features": int(X.shape[1]), "model_dir": str(model_dir)}
    log(f"[train_lightgbm_intraday] 📊 Summary: {summary}")
    return summary


if __name__ == "__main__":
    train_lightgbm_intraday()

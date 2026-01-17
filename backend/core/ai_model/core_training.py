# backend/core/ai_model/core_training.py — v1.7.0
"""
AION Analytics — Regression Engine (Multi-Horizon Expected Returns)

What this module does
- Trains one regressor per horizon on target_ret_<h> (1d…52w)
- Produces, per symbol+per horizon:
    predicted_return (decimal)
    label (-1/0/+1)
    rating (STRONG_SELL … STRONG_BUY) + rating_score (-2..+2)
    confidence ∈ [0.50, 0.98] (intended to become calibrated P(hit))
    score ∈ [-1, 1] (ranking signal)

Key upgrades in v1.7.0 (fixes the “±80% everywhere” clown-show)
- Dynamic, per-horizon clipping derived from the *observed* target distribution
  (p01/p99 + std) instead of a single hard +/-80% clamp.
- Quantile-based target clipping during training (same clip limits) to reduce
  outlier-driven model instability.
- Prediction diagnostics written nightly:
    raw_pred histogram, clipped_pred histogram, % clipped, and clip_limit used.
- Stronger post-train sanity gate uses the *dynamic* clip limit, not the hard cap.

Important note on “confidence”
- Today, confidence is still mostly “P(direction correct)” via accuracy_engine buckets
  unless your hit-definition becomes magnitude-aware and you track calibration error
  (Brier score) per horizon. This module is written so it will not fight that future.

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import log, _read_rolling, _read_aion_brain
from backend.core.sector_training.sector_inference import SectorModelStore

store = SectorModelStore()

from backend.core.memmap_trainer import train_lgbm_memmap_reservoir
from backend.core.confidence_calibrator import (
    load_calibration_map,
    load_accuracy_latest,
    calibrated_confidence,
    recent_horizon_accuracy_conf,
    combine_confidence,
    soft_performance_cap,
)

# ==========================================================
# LightGBM / RF
# ==========================================================

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load

# ==========================================================
# Optional Optuna
# ==========================================================

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


# === Refactor imports (mechanical) ===
from backend.core.ai_model.target_builder import (
    # public constants used throughout this module
    HORIZONS,
    MODEL_ROOT,
    TMP_MEMMAP_ROOT,
    ML_DATA_ROOT,
    LATEST_FEATURES_FILE,
    LATEST_FEATURES_CSV,
    PRED_DIAG_FILE,
    MIN_USABLE_ROWS,
    MIN_TARGET_STD,
    MAX_TARGET_ZERO_FRAC,
    MAX_STATS_SAMPLES,
    MAX_VAL_SAMPLES,
    HARD_MAX_ABS_RET,
    MIN_PRED_STD,
    MIN_CONF,
    MAX_CONF,
    DIAG_BINS,
    MAX_CLIP_SAT_FRAC,

    # internal helpers (leading underscore) referenced by core_training
    _preflight_dataset_or_die,
    _stream_target_stats,
    _clip_limit_for_horizon,
    _iter_parquet_batches,
    _stream_validation_sample,
    _feature_map_path,
    _load_horizon_feature_map,
    _model_path,
    _booster_path,
    _save_return_stats,
    _load_return_stats,
    _last_close_asof,
    _try_import_pyarrow,
)
from backend.core.ai_model.trainer import _make_regressor, _tune_lightgbm_regressor
from backend.core.ai_model.sanity_gates import _post_train_sanity
from backend.core.ai_model.feature_pipeline import _load_feature_list

# Exported for legacy callers (e.g., sector_training) that import from
# core_training directly.
from backend.core.ai_model.constants import FEATURE_LIST_FILE


def _aion_meta_snapshot() -> Dict[str, Any]:
    try:
        ab = _read_aion_brain() or {}
        meta = ab.get("_meta", {}) if isinstance(ab, dict) else {}
        if not isinstance(meta, dict):
            meta = {}

        cb = float(meta.get("confidence_bias", 1.0) or 1.0)
        rb = float(meta.get("risk_bias", 1.0) or 1.0)
        ag = float(meta.get("aggressiveness", 1.0) or 1.0)

        cb = float(max(0.70, min(1.30, cb)))
        rb = float(max(0.60, min(1.40, rb)))
        ag = float(max(0.60, min(1.50, ag)))

        return {
            "updated_at": meta.get("updated_at"),
            "confidence_bias": cb,
            "risk_bias": rb,
            "aggressiveness": ag,
        }
    except Exception:
        return {"updated_at": None, "confidence_bias": 1.0, "risk_bias": 1.0, "aggressiveness": 1.0}


# ==========================================================
# PATH HELPERS
# ==========================================================
def _resolve_dataset_path(dataset_name: str) -> Path:
    p = Path(dataset_name)
    if p.exists():
        return p

    # named shortcuts
    if dataset_name in ("daily", "training_data_daily.parquet"):
        return Path(PATHS["ML_DATASET_DAILY"])
    if dataset_name in ("intraday", "training_data_intraday.parquet"):
        return Path(PATHS["ML_DATASET_INTRADAY"])

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


# ==========================================================
# Underscore helpers
# (Imported from target_builder; do not redefine here — keep single source of truth.)
# ==========================================================

def train_model(
    dataset_name: str = "training_data_daily.parquet",
    use_optuna: bool = True,
    n_trials: int = 20,
    batch_size: int = 150_000,
    symbol_whitelist: Optional[set[str]] = None,
    model_root: Path | None = None,
) -> Dict[str, Any]:
    log(f"[ai_model] 🧠 Training regression models v1.7.0 (optuna={use_optuna}, batch_rows={batch_size})")

    model_root = model_root or MODEL_ROOT

    feat_info = _load_feature_list()
    feature_cols: List[str] = feat_info.get("feature_columns", [])
    target_cols: List[str] = feat_info.get("target_columns", [])

    if not feature_cols:
        log("[ai_model] ❌ No feature columns found in feature_list.")
        return {"status": "error", "error": "no_features"}

    df_path = _resolve_dataset_path(dataset_name)
    _preflight_dataset_or_die(df_path)

    summaries: Dict[str, Any] = {}
    return_stats: Dict[str, Any] = {}

    for horizon in HORIZONS:
        tgt_ret = f"target_ret_{horizon}"
        if tgt_ret not in target_cols:
            continue

        # Persist horizon feature map (currently identical to global list)
        try:
            _feature_map_path(horizon, model_root=model_root).write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
        except Exception:
            pass

        tstats = _stream_target_stats(
            df_path,
            tgt_ret,
            batch_size=max(50_000, int(batch_size)),
            max_samples=MAX_STATS_SAMPLES,
            symbol_whitelist=symbol_whitelist,
        )
        if tstats.get("status") != "ok":
            summaries[horizon] = {"status": "error", "error": f"target_stats_failed: {tstats.get('error')}"}
            continue

        usable_est = int(tstats.get("usable_rows_est", 0) or 0)
        y_std = float(tstats.get("std", 0.0) or 0.0)
        y_zero_frac = float(tstats.get("zero_frac", 1.0) or 1.0)

        clip_lim = _clip_limit_for_horizon(horizon, tstats)
        y_clip_low = -clip_lim
        y_clip_high = clip_lim

        log(
            f"[ai_model] 📌 Horizon={horizon} target stats: "
            f"usable_est={usable_est}, std={y_std:.6g}, zero_frac={y_zero_frac:.4f}, "
            f"p01={tstats.get('p01'):.6g}, p50={tstats.get('p50'):.6g}, p99={tstats.get('p99'):.6g}, "
            f"clip_limit={clip_lim:.4f}"
        )

        if usable_est < MIN_USABLE_ROWS:
            summaries[horizon] = {"status": "skipped", "reason": f"too_few_usable_rows({usable_est}<{MIN_USABLE_ROWS})", "target_stats": tstats}
            continue

        if y_std < MIN_TARGET_STD:
            summaries[horizon] = {"status": "skipped", "reason": f"low_target_variance(std<{MIN_TARGET_STD})", "target_stats": tstats}
            continue

        if y_zero_frac >= MAX_TARGET_ZERO_FRAC:
            summaries[horizon] = {"status": "skipped", "reason": f"targets_mostly_zero(zero_frac>={MAX_TARGET_ZERO_FRAC})", "target_stats": tstats}
            continue

        tstats["clip_limit"] = float(clip_lim)
        return_stats[horizon] = dict(tstats)

        # --------------------------
        # LightGBM path
        # --------------------------
        if HAS_LGBM:
            try:
                base = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "learning_rate": 0.05,
                    "num_leaves": 64,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "min_data_in_leaf": 50,
                    "lambda_l2": 1.0,
                    "num_boost_round": 800,
                }

                tuned: Dict[str, Any] = {}
                if use_optuna and n_trials and n_trials > 0:
                    log(f"[ai_model] ℹ️ Optuna requested for {horizon}. Building arrays for tuning (bounded).")

                    needed_cols = feature_cols + [tgt_ret]
                    X_parts: List[np.ndarray] = []
                    y_parts: List[np.ndarray] = []
                    total_used = 0

                    for df_batch in _iter_parquet_batches(df_path, needed_cols, batch_size=batch_size, symbol_whitelist=symbol_whitelist):
                        if df_batch.empty or tgt_ret not in df_batch.columns:
                            continue

                        y_raw = pd.to_numeric(df_batch[tgt_ret], errors="coerce").replace([np.inf, -np.inf], np.nan)
                        mask = y_raw.notna()
                        if not mask.any():
                            continue

                        X_df = df_batch.loc[mask, feature_cols]
                        X_df = X_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

                        y = y_raw.loc[mask].clip(lower=y_clip_low, upper=y_clip_high).to_numpy(dtype=np.float32, copy=False)
                        X = X_df.to_numpy(dtype=np.float32, copy=False)

                        if X.size == 0 or y.size == 0:
                            continue

                        X_parts.append(X)
                        y_parts.append(y)
                        total_used += int(len(y))
                        if total_used >= 250_000:
                            break

                    if total_used >= 5000:
                        X_all = np.concatenate(X_parts, axis=0)
                        y_all = np.concatenate(y_parts, axis=0)
                        tuned = _tune_lightgbm_regressor(X_all, y_all, horizon, n_trials=int(n_trials))
                    else:
                        log(f"[ai_model] ⚠️ Not enough rows for Optuna sample on {horizon}. Skipping tuning.")

                base.update(tuned or {})

                mm = train_lgbm_memmap_reservoir(
                    parquet_path=str(df_path),
                    feature_cols=feature_cols,
                    target_col=tgt_ret,
                    lgb_params=base,
                    symbol_whitelist=symbol_whitelist,
                    tmp_root=str(TMP_MEMMAP_ROOT),
                    max_rows=800_000,
                    batch_rows=int(batch_size),
                    min_rows=MIN_USABLE_ROWS,
                    seed=42,
                    cleanup=True,
                    y_clip_low=float(y_clip_low),
                    y_clip_high=float(y_clip_high),
                )

                booster = mm.model

                Xv, yv = _stream_validation_sample(
                    df_path,
                    feature_cols=feature_cols,
                    target_col=tgt_ret,
                    batch_size=max(50_000, int(batch_size)),
                    max_rows=MAX_VAL_SAMPLES,
                    seed=42,
                    y_clip_low=float(y_clip_low),
                    y_clip_high=float(y_clip_high),
                    symbol_whitelist=symbol_whitelist,
                )

                vdiag = _post_train_sanity(booster, Xv, yv, horizon=horizon, clip_limit=float(clip_lim))

                if vdiag.get("status") == "reject":
                    summaries[horizon] = {
                        "status": "rejected",
                        "reason": vdiag.get("reject_reason"),
                        "target_stats": tstats,
                        "validation": vdiag,
                        "rows_seen": int(mm.rows_seen),
                        "rows_used": int(mm.rows_used),
                        "seconds_ingest": float(mm.seconds_ingest),
                        "seconds_train": float(mm.seconds_train),
                    }
                    log(f"[ai_model] 🧨 Rejecting horizon={horizon}: {vdiag.get('reject_reason')}")
                    try:
                        rs = return_stats.get(horizon) if isinstance(return_stats, dict) else None
                        if not isinstance(rs, dict):
                            rs = dict(tstats) if isinstance(tstats, dict) else {}
                            return_stats[horizon] = rs
                        rs["valid_global"] = False
                        rs["invalid_reason"] = str(vdiag.get("reject_reason") or "rejected")
                        if isinstance(vdiag, dict):
                            rs["validation"] = dict(vdiag)
                    except Exception:
                        pass
                    continue

                bp = _booster_path(horizon, model_root=model_root)
                booster.save_model(str(bp))
                dump(booster, _model_path(horizon, model_root=model_root))

                summaries[horizon] = {
                    "status": "ok",
                    "model_path": str(_model_path(horizon, model_root=model_root)),
                    "booster_path": str(bp),
                    "target_stats": tstats,
                    "validation": vdiag,
                    "clip_limit": float(clip_lim),
                    "rows_seen": int(mm.rows_seen),
                    "rows_used": int(mm.rows_used),
                    "seconds_ingest": float(mm.seconds_ingest),
                    "seconds_train": float(mm.seconds_train),
                }
                
                # Track feature importance (adaptive ML pipeline)
                try:
                    from backend.core.ai_model.feature_importance import FeatureImportanceTracker
                    importance_tracker = FeatureImportanceTracker()
                    top_features = importance_tracker.compute_importance(
                        booster,
                        feature_cols,
                        horizon,
                        top_n=20
                    )
                    summaries[horizon]["top_features"] = list(top_features.keys())[:5]
                except Exception as e:
                    log(f"[ai_model] ⚠️ Feature importance tracking failed for {horizon}: {e}")

                try:
                    rs = return_stats.get(horizon) if isinstance(return_stats, dict) else None
                    if not isinstance(rs, dict):
                        rs = dict(tstats) if isinstance(tstats, dict) else {}
                        return_stats[horizon] = rs
                    rs["valid_global"] = True
                    rs["invalid_reason"] = None
                    if isinstance(vdiag, dict):
                        rs["validation"] = dict(vdiag)
                except Exception:
                    pass
            except Exception as e:
                log(f"[ai_model] ❌ Memmap training failed for {horizon}: {e}")
                summaries[horizon] = {"status": "error", "error": str(e), "target_stats": tstats}

            continue

        # --------------------------
        # RandomForest fallback
        # --------------------------
        try:
            needed_cols = feature_cols + [tgt_ret]
            X_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []
            total_used = 0

            for df_batch in _iter_parquet_batches(df_path, needed_cols, batch_size=batch_size, symbol_whitelist=symbol_whitelist):
                if df_batch.empty or tgt_ret not in df_batch.columns:
                    continue

                y_raw = pd.to_numeric(df_batch[tgt_ret], errors="coerce").replace([np.inf, -np.inf], np.nan)
                mask = y_raw.notna()
                if not mask.any():
                    continue

                X_df = (
                    df_batch.loc[mask, feature_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

                y = y_raw.loc[mask].clip(lower=y_clip_low, upper=y_clip_high).to_numpy(dtype=np.float32, copy=False)
                X = X_df.to_numpy(dtype=np.float32, copy=False)

                if X.size == 0 or y.size == 0:
                    continue

                X_parts.append(X)
                y_parts.append(y)
                total_used += int(len(y))

                if total_used >= 500_000:
                    break

            if total_used < 5000:
                summaries[horizon] = {"status": "skipped", "reason": f"too_few_samples({total_used})", "target_stats": tstats}
                continue

            X_all = np.concatenate(X_parts, axis=0)
            y_all = np.concatenate(y_parts, axis=0)

            X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

            model = _make_regressor({})
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)

            mae = float(mean_absolute_error(y_val, y_pred_val))
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            r2 = float(r2_score(y_val, y_pred_val))

            vdiag = _post_train_sanity(
                model,
                X_val.astype(np.float32, copy=False),
                y_val.astype(np.float32, copy=False),
                horizon=horizon,
                clip_limit=float(clip_lim),
            )
            if vdiag.get("status") == "reject":
                summaries[horizon] = {"status": "rejected", "reason": vdiag.get("reject_reason"), "target_stats": tstats, "validation": vdiag}
                log(f"[ai_model] 🧨 Rejecting RF horizon={horizon}: {vdiag.get('reject_reason')}")
                continue

            mp = _model_path(horizon, model_root=model_root)
            dump(model, mp)

            summaries[horizon] = {
                "status": "ok",
                "model_path": str(mp),
                "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "n_val": int(len(y_val))},
                "target_stats": tstats,
                "validation": vdiag,
                "clip_limit": float(clip_lim),
            }
            
            # Track feature importance (adaptive ML pipeline)
            try:
                from backend.core.ai_model.feature_importance import FeatureImportanceTracker
                importance_tracker = FeatureImportanceTracker()
                top_features = importance_tracker.compute_importance(
                    model,
                    feature_cols,
                    horizon,
                    top_n=20
                )
                summaries[horizon]["top_features"] = list(top_features.keys())[:5]
            except Exception as e:
                log(f"[ai_model] ⚠️ Feature importance tracking failed for {horizon}: {e}")

        except Exception as e:
            log(f"[ai_model] ❌ RF training failed for {horizon}: {e}")
            summaries[horizon] = {"status": "error", "error": str(e), "target_stats": tstats}

    if return_stats:
        _save_return_stats(return_stats)

    return {"status": "ok", "horizons": summaries}


def train_all_models(
    dataset_name: str = "training_data_daily.parquet",
    use_optuna: bool = True,
    n_trials: int = 20,
    batch_size: int = 150_000,
    symbol_whitelist: Optional[set[str]] = None,
    model_root: Path | None = None,
    as_of_date: Optional[str] = None,  # NEW: for replay mode point-in-time filtering
    **_: Any,
) -> Dict[str, Any]:
    # NOTE: Orchestration layers (nightly/replay) may pass extra keywords like
    # mode/as_of_date/force. We intentionally ignore them here so imports don't
    # break during refactors.
    # 
    # IMPORTANT: as_of_date filtering should happen at dataset build time
    # (build_daily_dataset already supports as_of_date parameter).
    # This parameter is kept here for API compatibility and documentation.
    
    if as_of_date:
        log(f"[ai_model] 📌 Training with as_of_date={as_of_date} (point-in-time mode)")
        log(f"[ai_model] ⚠️ Ensure dataset was built with same as_of_date to avoid look-ahead bias")
    
    return train_model(
        dataset_name=dataset_name,
        use_optuna=use_optuna,
        n_trials=n_trials,
        batch_size=batch_size,
        symbol_whitelist=symbol_whitelist,
        model_root=model_root,
    )


# ==========================================================
# LOAD REGRESSION MODELS
# ==========================================================

def _load_regressors(model_root: Path | None = None) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for horizon in HORIZONS:
        pkl = _model_path(horizon, model_root=model_root)
        txt = _booster_path(horizon, model_root=model_root)

        if pkl.exists():
            try:
                models[horizon] = load(pkl)
                continue
            except Exception as e:
                log(f"[ai_model] ⚠️ Failed to load regressor pkl for {horizon}: {e}")

        if HAS_LGBM and txt.exists():
            try:
                models[horizon] = lgb.Booster(model_file=str(txt))
            except Exception as e:
                log(f"[ai_model] ⚠️ Failed to load booster txt for {horizon}: {e}")

    return models


# ==========================================================
# RATING / LABEL / CONFIDENCE HELPERS
# ==========================================================

def _rating_from_return(pred_ret: float, stats: Dict[str, Any], base_conf: float) -> Tuple[str, int, int]:
    std = float(stats.get("std", 0.05)) or 0.05
    t_hold = 0.25 * std
    t_buy = 0.75 * std
    t_strong = 1.5 * std

    if pred_ret >= t_strong and base_conf >= 0.6:
        return "STRONG_BUY", 2, 1
    if pred_ret >= t_buy:
        return "BUY", 1, 1
    if pred_ret <= -t_strong and base_conf >= 0.6:
        return "STRONG_SELL", -2, -1
    if pred_ret <= -t_buy:
        return "SELL", -1, -1
    if abs(pred_ret) <= t_hold:
        return "HOLD", 0, 0

    if pred_ret > 0:
        return "BUY", 1, 1
    if pred_ret < 0:
        return "SELL", -1, -1
    return "HOLD", 0, 0


def _confidence_from_signal(pred_ret: np.ndarray, stats: Dict[str, Any], sector_momo: np.ndarray | None = None) -> np.ndarray:
    std = float(stats.get("std", 0.05)) or 0.05
    eps = 1e-8

    z = np.abs(pred_ret) / (std + eps)
    base_conf = 0.5 + 0.5 * (1.0 - np.exp(-z))

    if sector_momo is not None:
        sec = np.clip(sector_momo, -0.20, 0.20)
        tilt = 1.0 + 0.5 * sec
        base_conf = base_conf * tilt

    return np.clip(base_conf, MIN_CONF, MAX_CONF)


# ==========================================================
# Prediction features loader (parquet + csv snapshot)
# ==========================================================

def _read_latest_snapshot_any() -> Optional[pd.DataFrame]:
    if LATEST_FEATURES_FILE.exists():
        try:
            return pd.read_parquet(LATEST_FEATURES_FILE)
        except Exception as e:
            log(f"[ai_model] ⚠️ Failed reading latest_features parquet: {e}")

    if LATEST_FEATURES_CSV.exists():
        try:
            return pd.read_csv(LATEST_FEATURES_CSV)
        except Exception as e:
            log(f"[ai_model] ⚠️ Failed reading latest_features csv: {e}")

    return None


from typing import Optional, Set

def _load_latest_features_df(
    required_feature_cols: List[str],
    symbol_whitelist: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Load latest feature snapshot for prediction.

    1) Prefer latest_features snapshot (parquet/csv).
    2) Fallback: scan dataset parquet to find latest row per symbol by asof_date.

    symbol_whitelist:
      Optional set of symbols to restrict outputs (uppercase recommended).
    """
    # Normalize whitelist (if any)
    if symbol_whitelist:
        symbol_whitelist = {str(s).upper() for s in symbol_whitelist}

    snap = _read_latest_snapshot_any()
    if snap is not None and not snap.empty:
        try:
            df = snap
            if "symbol" not in df.columns:
                raise ValueError("latest_features missing symbol")

            df["symbol"] = df["symbol"].astype(str).str.upper()

            # Filter snapshot by whitelist if provided
            if symbol_whitelist:
                df = df[df["symbol"].isin(symbol_whitelist)]

            df = df.set_index("symbol")

            for c in required_feature_cols:
                if c not in df.columns:
                    df[c] = 0.0

            out = df[required_feature_cols].copy()
            out = (
                out.apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            return out.astype(np.float32, copy=False)

        except Exception as e:
            log(f"[ai_model] ⚠️ Failed preparing latest_features snapshot df: {e}")

    pa, ds = _try_import_pyarrow()
    if pa is None or ds is None:
        raise RuntimeError(
            "No latest_features snapshot and pyarrow unavailable for fallback prediction load."
        )

    df_path = Path(PATHS["ML_DATASET_DAILY"])
    cols = ["symbol", "asof_date"] + required_feature_cols

    latest_map: Dict[str, Tuple[str, np.ndarray]] = {}

    for df_batch in _iter_parquet_batches(
        df_path,
        cols,
        batch_size=200_000,
        symbol_whitelist=symbol_whitelist,
    ):
        if df_batch.empty:
            continue

        df_batch["symbol"] = df_batch["symbol"].astype(str).str.upper()
        df_batch["asof_date"] = df_batch["asof_date"].astype(str)

        feats = (
            df_batch[required_feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

        syms = df_batch["symbol"].values
        dates = df_batch["asof_date"].values

        for i in range(len(df_batch)):
            s = str(syms[i])
            d = str(dates[i])
            prev = latest_map.get(s)
            if prev is None or d > prev[0]:
                latest_map[s] = (d, feats[i].copy())

    if not latest_map:
        raise RuntimeError("Prediction feature fallback produced no rows.")

    symbols = sorted(latest_map.keys())
    mat = np.vstack([latest_map[s][1] for s in symbols]).astype(np.float32, copy=False)
    out = pd.DataFrame(mat, index=symbols, columns=required_feature_cols)
    return out


# ==========================================================
# Prediction diagnostics writer
# ==========================================================

def _hist_counts(values: np.ndarray, bins: np.ndarray) -> List[int]:
    try:
        counts, _ = np.histogram(values.astype(float, copy=False), bins=bins)
        return [int(x) for x in counts]
    except Exception:
        return []


def _write_pred_diagnostics(diag: Dict[str, Any]) -> None:
    try:
        PRED_DIAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        PRED_DIAG_FILE.write_text(json.dumps(diag, indent=2), encoding="utf-8")
        log(f"[ai_model] 📈 Prediction diagnostics written → {PRED_DIAG_FILE}")
    except Exception as e:
        log(f"[ai_model] ⚠️ Failed writing prediction diagnostics: {e}")


# ==========================================================
# PREDICTION — REGRESSION OVER ROLLING (EARNED CONFIDENCE)
# ==========================================================

def predict_all(
    rolling: Optional[Dict[str, Any]] = None,
    *,
    write_diagnostics: Optional[bool] = None,
    **_: Any,
) -> Dict[str, Any]:
    if rolling is None:
        rolling = _read_rolling() or {}

    if not rolling:
        log("[ai_model] ⚠️ predict_all: rolling is empty.")
        return {}

    # default: write diagnostics unless explicitly disabled
    if write_diagnostics is None:
        write_diagnostics = (os.getenv("AION_PRED_DIAGNOSTICS", "1") == "1")

    aion_meta = _aion_meta_snapshot()
    aion_cb = float(aion_meta.get("confidence_bias", 1.0) or 1.0)

    feat_info = _load_feature_list()
    feature_cols: List[str] = feat_info.get("feature_columns", [])
    if not feature_cols:
        log("[ai_model] ❌ predict_all: feature_columns list is empty.")
        return {}

    # Build rolling key map first (ALWAYS initialize)
    rolling_key_by_upper: Dict[str, str] = {}
    for k in rolling.keys():
        if str(k).startswith("_"):
            continue
        rolling_key_by_upper[str(k).upper()] = str(k)

    try:
        symbol_whitelist = set(rolling_key_by_upper.keys())
        X_df = _load_latest_features_df(feature_cols, symbol_whitelist=symbol_whitelist)
    except Exception as e:
        log(f"[ai_model] ❌ predict_all: failed to load latest features: {e}")
        return {}

    symbols: List[str] = [s for s in rolling_key_by_upper.keys() if s in X_df.index]
    if not symbols:
        log("[ai_model] ⚠️ predict_all: no overlap between rolling symbols and latest feature snapshot.")
        return {}

    X_df = X_df.loc[symbols, feature_cols]
    n_samples = int(X_df.shape[0])

    sec1 = X_df["sector_ret_1w"].to_numpy(dtype=float, copy=False) if "sector_ret_1w" in X_df.columns else np.zeros(n_samples, dtype=float)
    sec4 = X_df["sector_ret_4w"].to_numpy(dtype=float, copy=False) if "sector_ret_4w" in X_df.columns else np.zeros(n_samples, dtype=float)
    sector_momo = 0.5 * sec1 + 0.5 * sec4

    # derive sector label per symbol (from stable one-hot schema produced by ml_data_builder)
    sector_cols = [c for c in X_df.columns if isinstance(c, str) and c.startswith("sector_")]
    if sector_cols:
        sec_mat = X_df[sector_cols].to_numpy(dtype=float, copy=False)
        sec_idx = np.argmax(sec_mat, axis=1)
        sector_labels = np.array([sector_cols[int(i)][len("sector_"):] for i in sec_idx], dtype=object)
        sector_labels = np.array([str(s).upper().strip() if s else "UNKNOWN" for s in sector_labels], dtype=object)
    else:
        sector_labels = np.array(["UNKNOWN"] * len(X_df), dtype=object)

    regressors = _load_regressors()
    stats_map = _load_return_stats()

    cal_map = load_calibration_map()
    acc_latest = load_accuracy_latest()

    # Sector-level accuracy for lightweight confidence calibration (optional)
    sector_accuracy: Dict[str, Any] = {}
    try:
        p = (ML_DATA_ROOT / "metrics" / "accuracy_by_sector.json")
        if p.exists():
            sector_accuracy = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(sector_accuracy, dict):
                sector_accuracy = {}
    except Exception:
        sector_accuracy = {}

    # Global context/regime snapshot (optional)
    context_state: Dict[str, Any] = {}
    try:
        mp = PATHS.get("market_state", ML_DATA_ROOT / "market_state.json")
        mp_path = Path(mp) if not isinstance(mp, Path) else mp
        if mp_path.exists():
            context_state = json.loads(mp_path.read_text(encoding="utf-8"))
            if not isinstance(context_state, dict):
                context_state = {}
    except Exception:
        context_state = {}

    try:
        macro_blk = context_state.get("macro", {}) if isinstance(context_state, dict) else {}
        risk_off = bool((macro_blk or {}).get("risk_off") or context_state.get("risk_off"))
    except Exception:
        risk_off = False

    preds: Dict[str, Dict[str, Any]] = {}
    horizon_raw_preds: Dict[str, np.ndarray] = {}
    horizon_clip_limits: Dict[str, float] = {}

    # batch predict per horizon
    for h in HORIZONS:
        model = regressors.get(h)
        if model is None:
            continue

        stats = stats_map.get(h, {}) if isinstance(stats_map, dict) else {}
        if not isinstance(stats, dict):
            stats = {}
        clip_lim = float(stats.get("clip_limit") or _clip_limit_for_horizon(h, stats or {"std": 0.05}))
        horizon_clip_limits[h] = float(clip_lim)

        h_feats = _load_horizon_feature_map(h, fallback=feature_cols)
        Xh = X_df.reindex(columns=h_feats, fill_value=0.0)
        X_np = Xh.to_numpy(dtype=np.float32, copy=False)

        # Sector-aware prediction:
        # - prefer sector model if present + horizon valid
        # - fallback to global model
        # - if no model resolves for any rows, skip horizon
        from backend.core.sector_training.sector_validator import horizon_valid

        pred = np.full((X_np.shape[0],), np.nan, dtype=float)
        unique_secs = set(map(str, np.unique(sector_labels)))
        global_model = regressors.get(h)

        for sec in unique_secs:
            sec_model = None
            sec_stats = None

            bundle = store._load_sector_bundle(sec)  # cached
            if bundle is not None:
                ok, _reason = horizon_valid(bundle.stats, h)
                if ok:
                    sec_model = (bundle.models or {}).get(h)
                    # sector stats per-horizon block (optional)
                    try:
                        sec_stats = ((bundle.stats.get("horizons") or {}).get(h) or {})
                    except Exception:
                        sec_stats = None

            model = sec_model or global_model
            if model is None:
                continue

            mask = (sector_labels == sec)
            if not mask.any():
                continue

            try:
                pred[mask] = np.asarray(model.predict(X_np[mask]), dtype=float)
            except Exception as e:
                log(f"[ai_model] ⚠️ Prediction failed for sector={sec} horizon={h}: {e}")
                continue

        if np.all(np.isnan(pred)):
            continue

        horizon_raw_preds[h] = pred

    if not horizon_raw_preds:
        log("[ai_model] ⚠️ predict_all: no horizons produced predictions (no models loaded or all failed).")
        return {}

    # build diagnostics
    diagnostics: Dict[str, Any] = {
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "symbols": int(len(symbols)),
        "bins": [float(x) for x in DIAG_BINS.tolist()],
        "horizons": {},
    }

    # precompute clipped arrays for diagnostics (cheap + keeps logic consistent)
    horizon_clipped_preds: Dict[str, np.ndarray] = {}
    for h, raw_arr in horizon_raw_preds.items():
        lim = float(horizon_clip_limits.get(h, HARD_MAX_ABS_RET))
        clipped = np.clip(raw_arr.astype(float, copy=False), -lim, lim)
        horizon_clipped_preds[h] = clipped
        clip_frac = float(np.mean(np.isclose(clipped, -lim, atol=1e-12) | np.isclose(clipped, lim, atol=1e-12)))
        diagnostics["horizons"][h] = {
            "clip_limit": float(lim),
            "n": int(clipped.size),
            "raw_mean": float(np.mean(raw_arr)),
            "raw_std": float(np.std(raw_arr)),
            "clipped_mean": float(np.mean(clipped)),
            "clipped_std": float(np.std(clipped)),
            "clip_frac": float(clip_frac),
            "raw_hist": _hist_counts(raw_arr, DIAG_BINS),
            "clipped_hist": _hist_counts(clipped, DIAG_BINS),
        }

    # ----------------------------------------------------------
    # Coverage: sector-level viability per horizon (proxy)
    # ----------------------------------------------------------
    MIN_SECTOR_NAMES = int(os.getenv("AION_MIN_SECTOR_NAMES", "30"))
    sector_validity: Dict[str, Dict[str, Any]] = {}
    try:
        for h, clipped in horizon_clipped_preds.items():
            sec_map: Dict[str, Any] = {}
            for sec in np.unique(sector_labels):
                mask = (sector_labels == sec)
                n = int(np.sum(mask))
                if n <= 0:
                    continue
                std = float(np.std(clipped[mask]))
                valid = bool((n >= MIN_SECTOR_NAMES) and (std >= 0.5 * float(MIN_PRED_STD)))
                sec_map[str(sec)] = {"valid": valid, "n": n, "pred_std": std}
            sector_validity[h] = sec_map
    except Exception:
        sector_validity = {}

    # warn if clipping is still heavy
    for h, info in diagnostics["horizons"].items():
        try:
            if float(info.get("clip_frac", 0.0)) >= 0.20:
                log(f"[ai_model] ⚠️ Horizon={h} heavy clipping: clip_frac={info.get('clip_frac'):.3f} (instability likely)")
        except Exception:
            pass

    if write_diagnostics:
        _write_pred_diagnostics(diagnostics)

    # materialize per symbol predictions

    def _ticker_perf_ok(node: Dict[str, Any], horizon: str) -> bool:
        """Allow a ticker to speak even when the horizon/sector is weak,
        if we have evidence this ticker+horizon has been predictable historically."""
        try:
            b = node.get("brain") or node.get("horizon_perf") or {}
            if isinstance(b, dict) and "horizon_perf" in b and isinstance(b.get("horizon_perf"), dict):
                perf = b.get("horizon_perf") or {}
            elif isinstance(b, dict):
                perf = b
            else:
                perf = {}
            hnode = perf.get(horizon) or perf.get(horizon.lower()) or {}
            if not isinstance(hnode, dict):
                return False
            n = float(hnode.get("n", hnode.get("samples", 0)) or 0)
            hit = float(hnode.get("hit_ratio", hnode.get("hit_rate", 0)) or 0)
            mae = float(hnode.get("mae", 1e9) or 1e9)
            # conservative defaults: require decent history + edge
            return bool((n >= 60) and (hit >= 0.55) and (mae < 0.08))
        except Exception:
            return False

    for idx, sym_u in enumerate(symbols):
        rolling_key = rolling_key_by_upper.get(sym_u)
        node = rolling.get(rolling_key, {}) if rolling_key is not None else {}

        as_of_date = node.get("asof_date") or node.get("date")
        last_close = _last_close_asof(node, as_of_date)

        sym_res: Dict[str, Any] = {}
        sec_momo_val = float(sector_momo[idx]) if idx < len(sector_momo) else 0.0

        for h in HORIZONS:
            if h not in horizon_raw_preds:
                continue

            LONG_HORIZONS = {"13w", "26w", "52w"}
            if h in LONG_HORIZONS and bool(risk_off):
                sym_res[h] = {
                    "valid": False,
                    "status": "regime_blocked",
                    "reason": "risk_off_long_horizon",
                    "coverage": {"level": "regime"},
                }
                continue

            stats = stats_map.get(h, {}) if isinstance(stats_map, dict) else {}
            if not isinstance(stats, dict) or not stats:
                stats = {"std": 0.05}

            lim = float(horizon_clip_limits.get(h, HARD_MAX_ABS_RET))

            try:
                raw_pred = float(horizon_raw_preds[h][idx])
            except Exception as e:
                log(f"[ai_model] ⚠️ Regression prediction read failed for {sym_u}, horizon={h}: {e}")
                continue

            if not np.isfinite(raw_pred):
                # no valid model resolved for this symbol/horizon
                continue

            pred_ret = float(np.clip(raw_pred, -lim, lim))
            sec_label = str(sector_labels[idx]) if idx < len(sector_labels) else "UNKNOWN"
            global_ok = bool(stats.get("valid_global", True))
            # also require universe-wide prediction spread to avoid flat rankings
            try:
                hdiag = diagnostics.get("horizons", {}).get(h, {})
                if isinstance(hdiag, dict) and float(hdiag.get("clipped_std", 0.0) or 0.0) < float(MIN_PRED_STD):
                    global_ok = False
            except Exception:
                pass
            sector_ok = bool(sector_validity.get(h, {}).get(sec_label, {}).get("valid", False))
            ticker_ok = _ticker_perf_ok(node, h)
            if not (global_ok or sector_ok or ticker_ok):
                sym_res[h] = {
                    "valid": False,
                    "status": "insufficient_variance",
                    "reason": str(stats.get("invalid_reason") or "no_usable_signal"),
                    "coverage": {"level": "none", "sector": sec_label},
                }
                continue
            coverage_level = "global" if global_ok else ("sector" if sector_ok else "ticker")
            try:
                if coverage_level == "sector":
                    b = store._load_sector_bundle(sec_label)
                    if b is not None:
                        sstats = ((b.stats.get("horizons") or {}).get(h) or {})
                        if isinstance(sstats, dict) and float(sstats.get("std", 0.0) or 0.0) > 0:
                            stats = {**stats, **sstats}
            except Exception:
                pass
            was_clipped = bool(abs(raw_pred) > lim + 1e-12)

            conf_raw_arr = _confidence_from_signal(
                np.array([pred_ret], dtype=float),
                stats,
                np.array([sec_momo_val], dtype=float),
            )
            conf_raw = float(conf_raw_arr[0])

            conf_cal = calibrated_confidence(conf_raw, h, cal_map, min_conf=MIN_CONF, max_conf=MAX_CONF)
            conf_perf = recent_horizon_accuracy_conf(h, acc_latest, window_days=30, min_conf=MIN_CONF, max_conf=MAX_CONF)
            conf = combine_confidence(conf_raw, conf_cal, conf_perf, min_conf=MIN_CONF, max_conf=MAX_CONF)

            cb = float(np.clip(aion_cb, 0.70, 1.30))
            conf = float(np.clip(conf * (0.85 + 0.15 * cb), MIN_CONF, MAX_CONF))

            conf = soft_performance_cap(conf, conf_perf, max_overhang=0.12, min_conf=MIN_CONF, max_conf=MAX_CONF)

            # Sector-specific confidence calibration (lightweight, no retrain)
            sec_tilt = 1.0
            try:
                sec_perf = (sector_accuracy.get(sec_label, {}) or {}).get(h, {})
                if isinstance(sec_perf, dict) and int(sec_perf.get("n", 0) or 0) >= 50:
                    hit = float(sec_perf.get("hit_rate", 0.5) or 0.5)
                    sec_tilt = 0.9 + 0.4 * (hit - 0.5)
                    conf = float(np.clip(conf * sec_tilt, MIN_CONF, MAX_CONF))
            except Exception:
                sec_tilt = 1.0

            rating, rating_score, label = _rating_from_return(pred_ret, stats, conf)

            std = float(stats.get("std", 0.05)) or 0.05
            score = float(np.tanh(pred_ret / (2.0 * std)) * conf)

            target_price = float(last_close * (1.0 + pred_ret)) if last_close is not None else None

            components: Dict[str, Any] = {
                "model": {
                    "raw_prediction": float(raw_pred),
                    "predicted_return": float(pred_ret),
                    "clip_limit": float(lim),
                    "was_clipped": bool(was_clipped),
                },
                "confidence": {
                    "raw": float(conf_raw),
                    "calibrated": float(conf_cal),
                    "recent_perf": float(conf_perf),
                    "p_hit": float(conf),
                    "sector_tilt": float(sec_tilt),
                },
                "aion_brain": {
                    "confidence_bias": float(aion_cb),
                    "updated_at": aion_meta.get("updated_at"),
                },
            }

            sym_res[h] = {
                "valid": True,
                "coverage": {"level": str(coverage_level), "sector": str(sec_label)},
                "label": int(label),
                "confidence": float(conf),
                "score": float(score),
                "predicted_return": float(pred_ret),
                "target_price": target_price,
                "rating": rating,
                "rating_score": int(rating_score),
                "components": components,
            }

        if sym_res:
            preds[sym_u] = sym_res

    return preds


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AION Analytics Regression Model Engine")
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Mode: train or predict")
    parser.add_argument("--trials", type=int, default=10, help="Optuna trials")
    parser.add_argument("--dataset", type=str, default="training_data_daily.parquet", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=150_000, help="Parquet scan batch size")
    parser.add_argument("--no-optuna", action="store_true", help="Disable Optuna tuning")
    parser.add_argument("--no-diag", action="store_true", help="Disable prediction diagnostics write (predict mode)")

    args = parser.parse_args()

    if args.mode == "train":
        summary = train_model(
            dataset_name=args.dataset,
            use_optuna=(not bool(args.no_optuna)),
            n_trials=int(args.trials),
            batch_size=int(args.batch_size),
        )
        print(json.dumps(summary, indent=2))
    else:
        log("[ai_model] 🔍 Running batch regression prediction (--mode predict)…")
        rolling = _read_rolling() or {}
        if not rolling:
            print(json.dumps({"error": "rolling_empty"}, indent=2))
        else:
            preds = predict_all(rolling, write_diagnostics=(not bool(args.no_diag)))
            out = {"symbols_predicted": len(preds), "sample_preview": {}}
            for s in sorted(preds.keys())[:5]:
                out["sample_preview"][s] = preds[s]
            print(json.dumps(out, indent=2))

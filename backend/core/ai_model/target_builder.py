# backend/core/ai_model/target_builder.py
# Auto-refactor from backend/core/ai_model.py v1.7.0 (mechanical split)
#
# This module keeps *underscore helpers* that core_training / predictor depend on.
# Important: keep this module import-safe (no star-import assumptions).

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from backend.core.config import TIMEZONE
from backend.core.data_pipeline import log

# Single source of truth for paths/constants
from .constants import (
    ML_DATA_ROOT,
    DATASET_DIR,
    DATASET_FILE,
    FEATURE_LIST_FILE,
    LATEST_FEATURES_FILE,
    LATEST_FEATURES_CSV,
    MODEL_ROOT,
    TMP_MEMMAP_ROOT,
    METRICS_ROOT,
    RETURN_STATS_FILE,
    PRED_DIAG_FILE,
    HORIZONS,
    HARD_MAX_ABS_RET,
    MIN_CONF,
    MAX_CONF,
    DIAG_BINS,
    MIN_USABLE_ROWS,
    MIN_TARGET_STD,
    MIN_PRED_STD,
    MAX_TARGET_ZERO_FRAC,
    MAX_CLIP_SAT_FRAC,
    MAX_STATS_SAMPLES,
    MAX_VAL_SAMPLES,
    CLIP_FACTOR,
    MIN_CLIP_SHORT,
    MIN_CLIP_LONG,
    MAX_CLIP_LONG,
    MAX_CLIP_SHORT,
)

# Re-export common names for backward compatibility with older import sites.
__all__ = [
    "HORIZONS",
    "MODEL_ROOT",
    "TMP_MEMMAP_ROOT",
    "ML_DATA_ROOT",
    "LATEST_FEATURES_FILE",
    "LATEST_FEATURES_CSV",
    "PRED_DIAG_FILE",
    "MIN_USABLE_ROWS",
    "MIN_TARGET_STD",
    "MAX_TARGET_ZERO_FRAC",
    "MAX_STATS_SAMPLES",
    "MAX_VAL_SAMPLES",
    "HARD_MAX_ABS_RET",
    "MIN_PRED_STD",
    "MIN_CONF",
    "MAX_CONF",
    "DIAG_BINS",
    "MAX_CLIP_SAT_FRAC",
    "_try_import_pyarrow",
    "_preflight_dataset_or_die",
    "_stream_target_stats",
    "_clip_limit_for_horizon",
    "_iter_parquet_batches",
    "_stream_validation_sample",
    "_feature_map_path",
    "_load_horizon_feature_map",
    "_model_path",
    "_booster_path",
    "_save_return_stats",
    "_load_return_stats",
    "_last_close_asof",
]


def _try_import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.dataset as ds  # type: ignore
        return pa, ds
    except Exception as e:
        log(f"[ai_model] ⚠️ pyarrow not available: {e}")
        return None, None


# ==========================================================
# PATH HELPERS
# ==========================================================
def _model_path(horizon: str, model_root: Path | None = None) -> Path:
    root = Path(model_root) if model_root is not None else Path(MODEL_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return root / f"regressor_{str(horizon).strip()}.pkl"


def _booster_path(horizon: str, model_root: Path | None = None) -> Path:
    root = Path(model_root) if model_root is not None else Path(MODEL_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return root / f"regressor_{str(horizon).strip()}.txt"


def _feature_map_path(horizon: str, model_root: Path | None = None) -> Path:
    root = Path(model_root) if model_root is not None else Path(MODEL_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return root / f"feature_map_{str(horizon).strip()}.json"


def _load_horizon_feature_map(horizon: str, fallback: List[str], model_root: Path | None = None) -> List[str]:
    p = _feature_map_path(horizon, model_root=model_root)
    if not p.exists():
        return list(fallback)
    try:
        arr = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(arr, list) and arr:
            return [str(x) for x in arr]
    except Exception:
        pass
    return list(fallback)


# ==========================================================
# RETURN STATS
# ==========================================================
def _save_return_stats(stats: Dict[str, Any], stats_path: Path | None = None) -> None:
    payload = {"generated_at": datetime.now(TIMEZONE).isoformat(), "horizons": stats}
    try:
        METRICS_ROOT.mkdir(parents=True, exist_ok=True)
        (stats_path or RETURN_STATS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"[ai_model] 📊 Return stats written → {(stats_path or RETURN_STATS_FILE)}")
    except Exception as e:
        log(f"[ai_model] ⚠️ Failed to write return stats: {e}")


def _load_return_stats(stats_path: Path | None = None) -> Dict[str, Any]:
    path = stats_path or RETURN_STATS_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw.get("horizons", {}) or {}
    except Exception:
        pass
    return {}


def _clip_limit_for_horizon(horizon: str, stats: Dict[str, Any]) -> float:
    """
    Compute a sane, per-horizon clip limit (absolute return).
    """
    std = float(stats.get("std", 0.05) or 0.05)
    p01 = float(stats.get("p01", 0.0) or 0.0)
    p99 = float(stats.get("p99", 0.0) or 0.0)

    base = max(abs(p01), abs(p99))
    if base <= 0:
        base = 6.0 * std  # fallback if percentiles missing

    base = float(base) * float(CLIP_FACTOR)

    is_long = horizon in ("13w", "26w", "52w")
    min_lim = float(MIN_CLIP_LONG if is_long else MIN_CLIP_SHORT)
    max_lim = float(MAX_CLIP_LONG if is_long else MAX_CLIP_SHORT)

    lim = max(base, 3.0 * std, min_lim)
    lim = min(lim, max_lim, HARD_MAX_ABS_RET)
    return float(lim)


# ==========================================================
# Batch readers (pyarrow streaming)
# ==========================================================
def _iter_parquet_batches(
    parquet_path: Path,
    columns: Optional[List[str]] = None,
    batch_size: int = 100_000,
    symbol_whitelist: Optional[Set[str]] = None,
):
    pa, ds = _try_import_pyarrow()
    if pa is None or ds is None:
        raise RuntimeError("pyarrow is required for streaming parquet batches")

    # Normalize whitelist once (upper-case, de-duped)
    wl: Optional[Set[str]] = None
    if symbol_whitelist:
        wl = {str(s).upper().strip() for s in symbol_whitelist if str(s).strip()}
        if not wl:
            wl = None

    # If we're filtering by symbol, make sure "symbol" is actually available
    # for filtering (either via scanner predicate or pandas fallback).
    cols = list(columns) if columns else None
    if wl is not None and cols is not None and "symbol" not in cols:
        cols = ["symbol"] + cols

    dataset = ds.dataset(str(parquet_path), format="parquet")

    # Prefer Arrow-side filtering (way faster than pandas filtering)
    scanner_kwargs: Dict[str, Any] = {
        "columns": cols,
        "batch_size": int(batch_size),
    }
    if wl is not None:
        try:
            scanner_kwargs["filter"] = ds.field("symbol").isin(sorted(wl))
        except Exception:
            # Some pyarrow versions can be picky; fall back to pandas filtering.
            pass

    scanner = dataset.scanner(**scanner_kwargs)
    for batch in scanner.to_batches():
        if batch.num_rows <= 0:
            continue
        df = batch.to_pandas()
        if wl is not None and "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper().isin(wl)]
        if df is None or len(df) <= 0:
            continue
        yield df


def _preflight_dataset_or_die(parquet_path: Path) -> None:
    """Hard gate: ensure parquet exists, is non-empty, and is scan-readable."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"[DATA PREFLIGHT] Parquet file missing: {parquet_path}")
    try:
        if parquet_path.stat().st_size <= 0:
            raise RuntimeError(f"[DATA PREFLIGHT] Parquet file is empty: {parquet_path}")
    except Exception as e:
        raise RuntimeError(f"[DATA PREFLIGHT] Failed to stat parquet: {e}")

    # Minimal scan to ensure dataset is readable and yields at least one batch.
    try:
        for _df in _iter_parquet_batches(parquet_path, columns=None, batch_size=10_000):
            return
        raise RuntimeError("[DATA PREFLIGHT] Parquet readable but yielded no rows.")
    except Exception as e:
        raise RuntimeError(f"[DATA PREFLIGHT] Failed to scan parquet: {e}")


# ==========================================================
# Streamed diagnostics + sanity gates
# ==========================================================
def _stream_target_stats(
    parquet_path: Path,
    target_col: str,
    *,
    batch_size: int = 200_000,
    max_samples: int = MAX_STATS_SAMPLES,
    symbol_whitelist: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    usable = 0
    samples: List[float] = []

    # NOTE: if a symbol whitelist is provided, we MUST include "symbol" in the
    # scan columns, otherwise the whitelist filter can't be applied.
    cols = [target_col]
    if symbol_whitelist:
        cols = ["symbol", target_col]

    try:
        for df in _iter_parquet_batches(parquet_path, cols, batch_size=batch_size, symbol_whitelist=symbol_whitelist):
            if df is None or df.empty or target_col not in df.columns:
                continue
            s = pd.to_numeric(df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            v = s.dropna()
            if v.empty:
                continue

            arr = v.to_numpy(dtype=float, copy=False)
            usable += int(arr.size)

            if len(samples) < int(max_samples):
                take = min(int(max_samples) - len(samples), int(arr.size))
                if take > 0:
                    # Keep tails bounded while sampling without distorting horizons.
                    sl = np.clip(arr[:take], -float(HARD_MAX_ABS_RET), float(HARD_MAX_ABS_RET))
                    samples.extend([float(x) for x in sl])

    except Exception as e:
        return {"status": "error", "error": str(e)}

    if not samples:
        return {
            "status": "ok",
            "usable_rows_est": int(usable),
            "n_samples": 0,
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "zero_frac": 1.0,
        }

    y = np.array(samples, dtype=float)
    zero_frac = float(np.mean(np.isclose(y, 0.0, atol=1e-12)))

    return {
        "status": "ok",
        "usable_rows_est": int(usable),
        "n_samples": int(len(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "p01": float(np.percentile(y, 1)),
        "p05": float(np.percentile(y, 5)),
        "p25": float(np.percentile(y, 25)),
        "p50": float(np.percentile(y, 50)),
        "p75": float(np.percentile(y, 75)),
        "p95": float(np.percentile(y, 95)),
        "p99": float(np.percentile(y, 99)),
        "zero_frac": float(zero_frac),
    }


def _stream_validation_sample(
    parquet_path: Path,
    feature_cols: List[str],
    target_col: str,
    *,
    batch_size: int = 200_000,
    max_rows: int = MAX_VAL_SAMPLES,
    seed: int = 42,
    y_clip_low: Optional[float] = None,
    y_clip_high: Optional[float] = None,
    symbol_whitelist: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_features = int(len(feature_cols))
    if n_features <= 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_res = np.empty((int(max_rows), n_features), dtype=np.float32)
    y_res = np.empty((int(max_rows),), dtype=np.float32)
    rows_seen = 0
    rows_used = 0

    cols = list(feature_cols) + [target_col]
    # Ensure "symbol" is present when whitelisting, otherwise filtering can't run
    # at the scanner level (and may not run at all).
    if symbol_whitelist and "symbol" not in cols:
        cols = ["symbol"] + cols

    for df in _iter_parquet_batches(parquet_path, cols, batch_size=batch_size, symbol_whitelist=symbol_whitelist):
        if df is None or df.empty or target_col not in df.columns:
            continue

        y = pd.to_numeric(df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = y.notna()
        if not mask.any():
            continue

        df_use = df.loc[mask, feature_cols]
        y_use = y.loc[mask]

        Xn = df_use.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        yn = pd.to_numeric(y_use, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if yn.empty:
            continue

        if y_clip_low is not None and y_clip_high is not None:
            yn = yn.clip(lower=float(y_clip_low), upper=float(y_clip_high))

        idx = yn.index
        X_arr = Xn.loc[idx].to_numpy(dtype=np.float32, copy=False)
        y_arr = yn.to_numpy(dtype=np.float32, copy=False)
        if X_arr.size == 0 or y_arr.size == 0:
            continue

        for i in range(X_arr.shape[0]):
            rows_seen += 1
            if rows_used < int(max_rows):
                X_res[rows_used] = X_arr[i]
                y_res[rows_used] = y_arr[i]
                rows_used += 1
            else:
                j = int(rng.integers(0, rows_seen))
                if j < int(max_rows):
                    X_res[j] = X_arr[i]
                    y_res[j] = y_arr[i]

    return X_res[:rows_used], y_res[:rows_used]


# ==========================================================
# Rolling helper
# ==========================================================
def _last_close_asof(node: Dict[str, Any], as_of_date: Any) -> Optional[float]:
    """
    Best-effort: extract a "last close" value for target-price computation.

    This is intentionally defensive because rolling nodes evolved over time.
    """
    try:
        # Direct keys
        for k in ("last_close", "close", "adj_close", "price", "last_price"):
            v = node.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass

        # History arrays (common pattern)
        hist = node.get("history")
        if isinstance(hist, list) and hist:
            # try exact date match first
            if as_of_date is not None:
                sdate = str(as_of_date)
                for row in reversed(hist):
                    if not isinstance(row, dict):
                        continue
                    d = row.get("date") or row.get("asof_date") or row.get("timestamp")
                    if d is not None and str(d) == sdate:
                        v = row.get("close") or row.get("adj_close") or row.get("price")
                        if v is not None:
                            return float(v)
            # fallback: last entry
            for row in reversed(hist):
                if not isinstance(row, dict):
                    continue
                v = row.get("close") or row.get("adj_close") or row.get("price")
                if v is not None:
                    return float(v)

        return None
    except Exception:
        return None

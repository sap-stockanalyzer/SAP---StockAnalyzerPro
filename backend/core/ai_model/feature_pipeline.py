"""Feature pipeline — AION Analytics

Includes:
- pyarrow/parquet batch scanning helpers
- latest_features snapshot loader
- fallback loader that scans the training parquet for the latest row per symbol
- feature_list loader: _load_feature_list()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from backend.core.data_pipeline import log

# Constants live in a dedicated module to avoid circular imports.
from .constants import (
    DATASET_FILE,
    FEATURE_LIST_FILE,
    LATEST_FEATURES_CSV,
    LATEST_FEATURES_FILE,
)


def _try_import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.dataset as ds  # type: ignore
        return pa, ds
    except Exception:
        return None, None


# ==========================================================
# Feature list loader
# ==========================================================
def _load_feature_list() -> Dict[str, Any]:
    """
    Load the feature_list JSON that the ML data builder writes.
    """
    if not FEATURE_LIST_FILE.exists():
        raise FileNotFoundError(f"Feature list missing at {FEATURE_LIST_FILE}")
    return json.loads(FEATURE_LIST_FILE.read_text(encoding="utf-8"))


# ==========================================================
# Dataset helpers
# ==========================================================
def _resolve_dataset_path(dataset_name: str | None = None) -> Path:
    """
    Return the best-available training dataset parquet path.

    dataset_name is accepted to preserve older call-sites, but today we
    always use the single daily parquet.
    """
    return Path(DATASET_FILE)


def _iter_parquet_batches(
    path: Path,
    columns: Optional[List[str]] = None,
    *,
    batch_size: int = 200_000,
    symbol_whitelist: Optional[Set[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Yield dataframes from a parquet file in chunks when possible.
    """
    if symbol_whitelist:
        symbol_whitelist = {str(s).upper() for s in symbol_whitelist}

    pa, ds = _try_import_pyarrow()
    if pa is None or ds is None:
        # Fallback: one-shot read
        df = pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)
        if symbol_whitelist and "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper().isin(symbol_whitelist)]
        yield df
        return

    dataset = ds.dataset(str(path), format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=int(batch_size))
    for rb in scanner.to_batches():
        if rb.num_rows <= 0:
            continue
        df = rb.to_pandas()
        if symbol_whitelist and "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper().isin(symbol_whitelist)]
        if df is None or df.empty:
            continue
        yield df


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


def _load_latest_features_df(
    required_feature_cols: List[str],
    *,
    symbol_whitelist: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Load latest feature snapshot for prediction.

    1) Prefer latest_features snapshot (parquet/csv).
    2) Fallback: scan dataset parquet to find latest row per symbol by asof_date.
    """
    if symbol_whitelist:
        symbol_whitelist = {str(s).upper() for s in symbol_whitelist}

    snap = _read_latest_snapshot_any()
    if snap is not None and not snap.empty:
        try:
            df = snap
            if "symbol" not in df.columns:
                raise ValueError("latest_features missing symbol")

            df["symbol"] = df["symbol"].astype(str).str.upper()

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

    # Fallback: parquet scan for latest row per symbol
    pa, ds = _try_import_pyarrow()
    if pa is None or ds is None:
        raise RuntimeError("No latest_features snapshot and pyarrow unavailable for fallback prediction load.")

    df_path = _resolve_dataset_path(DATASET_FILE.name)
    cols = ["symbol", "asof_date"] + list(required_feature_cols)

    latest_map: Dict[str, Tuple[str, np.ndarray]] = {}

    for df_batch in _iter_parquet_batches(
        df_path,
        columns=cols,
        batch_size=200_000,
        symbol_whitelist=symbol_whitelist,
    ):
        if df_batch is None or df_batch.empty:
            continue

        if "symbol" not in df_batch.columns or "asof_date" not in df_batch.columns:
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

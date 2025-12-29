# backend/core/memmap_trainer.py
"""
Memmap + Reservoir trainer for LightGBM
- Streams parquet in batches
- Writes into disk-backed np.memmap (float32)
- Uses reservoir sampling to cap max_rows without loading everything
- Trains LightGBM directly from memmap

OOM hardening upgrades (2025-12-28):
- Chunked Arrow -> NumPy conversion to avoid building full batch X in RAM.
- Optional fast symbol whitelist filtering (pyarrow.compute) with safe fallback.
- More efficient memmap writes (bulk fill while reservoir not full).
- Debug option to keep memmap temp dir: set AION_KEEP_MEMMAP=1.
- Safety cap for huge batch_rows: uses internal chunk_rows (default 4096) for conversion/writes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import lightgbm as lgb

# -------------------------
# PyArrow imports (SAFE)
# -------------------------
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    try:
        import pyarrow.compute as pc  # type: ignore
    except Exception:
        pc = None  # type: ignore
except Exception:
    pa = None  # type: ignore
    pq = None  # type: ignore
    pc = None  # type: ignore


@dataclass
class MemmapTrainResult:
    model: object
    rows_seen: int
    rows_used: int
    seconds_ingest: float
    seconds_train: float
    tmp_dir: str


def _ensure_pyarrow() -> None:
    if pa is None or pq is None:
        raise RuntimeError(
            "pyarrow is required for memmap trainer. Install with: pip install pyarrow"
        )


def _safe_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32, copy=False)


def _finite_row_mask(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.isfinite(X).all(axis=1)


def _norm_whitelist(symbol_whitelist: Optional[Set[str]]) -> Optional[Set[str]]:
    if symbol_whitelist is None:
        return None
    return {str(s).upper() for s in symbol_whitelist}


def _arrow_mask_for_symbols(tbl: "pa.Table", symbol_whitelist_u: Set[str]) -> Optional[np.ndarray]:
    """Best-effort boolean mask for rows whose 'symbol' is in whitelist."""
    try:
        if "symbol" not in tbl.column_names:
            return None
        col = tbl.column("symbol")

        if pc is not None:
            wl = pa.array(list(symbol_whitelist_u), type=pa.string())
            try:
                col_u = pc.utf8_upper(col)  # type: ignore[attr-defined]
            except Exception:
                col_u = col
            keep = pc.is_in(col_u, value_set=wl)
            return keep.to_numpy(zero_copy_only=False).astype(bool, copy=False)

        sym = col.to_numpy(zero_copy_only=False)
        sym_u = np.array([str(x).upper() for x in sym], dtype=object)
        return np.array([s in symbol_whitelist_u for s in sym_u], dtype=bool)
    except Exception:
        return None


def train_lgbm_memmap_reservoir(
    parquet_path: str,
    feature_cols: List[str],
    target_col: str,
    lgb_params: Dict,
    symbol_whitelist: Optional[Set[str]] = None,
    *,
    tmp_root: str,
    max_rows: int = 800_000,
    batch_rows: int = 100_000,
    min_rows: int = 20_000,
    seed: int = 42,
    y_clip_low: float | None = None,
    y_clip_high: float | None = None,
    cleanup: bool = True,
) -> MemmapTrainResult:
    """Stream parquet -> reservoir sample into disk-backed memmap -> train LGBM."""
    _ensure_pyarrow()

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    os.makedirs(tmp_root, exist_ok=True)
    run_id = f"mm_{int(time.time())}_{os.getpid()}_{np.random.default_rng(seed).integers(0, 1_000_000)}"
    tmp_dir = os.path.join(tmp_root, run_id)
    os.makedirs(tmp_dir, exist_ok=True)

    X_path = os.path.join(tmp_dir, "X.float32.mmap")
    y_path = os.path.join(tmp_dir, "y.float32.mmap")

    n_features = len(feature_cols)
    if n_features == 0:
        raise ValueError("feature_cols is empty")

    rng = np.random.default_rng(seed)

    X_mm = np.memmap(X_path, mode="w+", dtype=np.float32, shape=(max_rows, n_features))
    y_mm = np.memmap(y_path, mode="w+", dtype=np.float32, shape=(max_rows,))

    rows_seen = 0
    rows_used = 0

    ingest_start = time.time()

    symbol_whitelist_u = _norm_whitelist(symbol_whitelist)

    pf = pq.ParquetFile(parquet_path)
    cols = list(feature_cols) + [target_col]
    if symbol_whitelist_u and "symbol" not in cols:
        cols.append("symbol")

    chunk_rows = int(os.getenv("AION_MEMMAP_CHUNK_ROWS", "4096"))
    chunk_rows = max(256, min(50_000, chunk_rows))

    for batch in pf.iter_batches(batch_size=int(batch_rows), columns=cols):
        tbl = pa.Table.from_batches([batch])
        n_batch = int(tbl.num_rows)
        if n_batch <= 0:
            continue

        keep_mask = None
        if symbol_whitelist_u is not None:
            keep_mask = _arrow_mask_for_symbols(tbl, symbol_whitelist_u)

        for start in range(0, n_batch, chunk_rows):
            length = min(chunk_rows, n_batch - start)
            tsub = tbl.slice(start, length)

            if keep_mask is not None:
                km = keep_mask[start : start + length]
                if km is None or km.size == 0 or (not km.any()):
                    continue
            else:
                km = None

            col_arrays: List[np.ndarray] = []
            for c in feature_cols:
                try:
                    a = tsub.column(c).to_numpy(zero_copy_only=False)
                except Exception:
                    a = np.zeros((length,), dtype=np.float32)
                col_arrays.append(_safe_float32(np.asarray(a)))

            X_chunk = np.stack(col_arrays, axis=1)
            y_arr = tsub.column(target_col).to_numpy(zero_copy_only=False)
            y_chunk = _safe_float32(np.asarray(y_arr))

            if km is not None:
                X_chunk = X_chunk[km]
                y_chunk = y_chunk[km]

            if y_clip_low is not None and y_clip_high is not None:
                y_chunk = np.clip(y_chunk, float(y_clip_low), float(y_clip_high))

            mask = _finite_row_mask(X_chunk, y_chunk)
            if not mask.all():
                X_chunk = X_chunk[mask]
                y_chunk = y_chunk[mask]

            if X_chunk.shape[0] == 0:
                continue

            m = int(X_chunk.shape[0])

            if rows_used < max_rows:
                take = min(m, max_rows - rows_used)
                X_mm[rows_used : rows_used + take] = X_chunk[:take]
                y_mm[rows_used : rows_used + take] = y_chunk[:take]
                rows_used += take
                rows_seen += take

                if take < m:
                    X_rem = X_chunk[take:]
                    y_rem = y_chunk[take:]
                else:
                    continue
            else:
                X_rem = X_chunk
                y_rem = y_chunk

            for i in range(X_rem.shape[0]):
                rows_seen += 1
                j = int(rng.integers(0, rows_seen))
                if j < max_rows:
                    X_mm[j] = X_rem[i]
                    y_mm[j] = y_rem[i]

    seconds_ingest = time.time() - ingest_start

    if rows_used < min_rows:
        if cleanup and os.getenv("AION_KEEP_MEMMAP", "0") != "1":
            _cleanup_dir(tmp_dir)
        raise RuntimeError(
            f"Not enough training rows for {target_col}: rows_used={rows_used}, min_rows={min_rows}"
        )

    X_train = X_mm[:rows_used]
    y_train = y_mm[:rows_used]

    train_start = time.time()

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    num_boost_round = int(lgb_params.get("num_boost_round", 800))

    model = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=num_boost_round)

    seconds_train = time.time() - train_start

    if cleanup and os.getenv("AION_KEEP_MEMMAP", "0") != "1":
        _cleanup_dir(tmp_dir)

    return MemmapTrainResult(
        model=model,
        rows_seen=int(rows_seen),
        rows_used=int(rows_used),
        seconds_ingest=float(seconds_ingest),
        seconds_train=float(seconds_train),
        tmp_dir=str(tmp_dir),
    )


def _cleanup_dir(path: str) -> None:
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except Exception:
                    pass
        try:
            os.rmdir(path)
        except Exception:
            pass
    except Exception:
        pass

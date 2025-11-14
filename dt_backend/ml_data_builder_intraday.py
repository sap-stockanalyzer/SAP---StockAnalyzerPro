# dt_backend/ml_data_builder_intraday.py — v2.0
"""
Intraday ML dataset builder for AION day-trading engine.

Reads:
    data_dt/rolling_intraday.json.gz
         Either of:
           {"bars": {SYM: [bar, ...], ...}, ...}
        or {SYM: {"bars": [bar, ...], ...}, ...}

Writes:
    ml_data_dt/training_data_intraday.parquet

Each row:
    ts          (timestamp, UTC)
    symbol      (str)
    label       (str: SELL/HOLD/BUY)
    fwd_ret     (float)
    split       ("train"/"valid")
    + numeric feature columns from feature_engineering / indicators
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

import pandas as pd

try:
    # Prefer dedicated DT logger if available
    from dt_backend.dt_logger import dt_log as log
except Exception:  # pragma: no cover - simple fallback
    def log(msg: str) -> None:
        print(msg, flush=True)

from .config_dt import DT_PATHS
from .feature_engineering import build_symbol_features


ROLLING_PATH: Path = DT_PATHS["data_dt"] / "rolling_intraday.json.gz"
DATASET_PATH: Path = DT_PATHS["dtml_data"] / "training_data_intraday.parquet"
BUILD_LOG_PATH: Path = DT_PATHS["dtlogs"] / "dataset_builds.jsonl"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _read_raw_rolling() -> Dict[str, Any]:
    """
    Raw JSON loader for intraday rolling cache.

    Returns an empty dict if file is missing or invalid.
    """
    if not ROLLING_PATH.exists():
        log(f"[DT] [ml_data_builder_intraday] ⚠️ rolling file missing at {ROLLING_PATH}")
        return {}

    try:
        with gzip.open(ROLLING_PATH, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log("[DT] [ml_data_builder_intraday] ⚠️ rolling file not a dict; resetting to empty.")
            return {}
        return data
    except Exception as e:
        log(f"[DT] [ml_data_builder_intraday] ⚠️ failed to read rolling: {e}")
        return {}


def _extract_bar_map(raw: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalize various rolling layouts into a simple:

        { "SYM": [ {bar}, ... ], ... }

    Supported layouts:

      1) Top-level 'bars' dict:
            {"bars": {"AAPL": [...], "MSFT": [...]}, ...}

      2) Per-symbol nodes:
            {"AAPL": {"bars": [...]}, "MSFT": {"bars": [...]}, ...}
    """
    if not raw:
        return {}

    # Case 1: bars at top level
    if "bars" in raw and isinstance(raw["bars"], dict):
        out = {}
        for sym, val in raw["bars"].items():
            if isinstance(val, list):
                out[sym] = val
        return out

    # Case 2: symbol -> {"bars": [...]}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sym, node in raw.items():
        if not isinstance(node, dict):
            continue
        bars = node.get("bars")
        if isinstance(bars, list):
            out[sym] = bars

    return out


def _select_symbols(
    bar_map: Dict[str, List[Dict[str, Any]]],
    max_symbols: int,
    min_bars: int,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Choose up to max_symbols with at least min_bars bars, sorted by length desc.
    """
    candidates = [
        (sym, bars)
        for sym, bars in bar_map.items()
        if isinstance(bars, list) and len(bars) >= min_bars
    ]

    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    return candidates[:max_symbols]


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------


def build_intraday_dataset(
    dt_rolling: Optional[Dict[str, Any]] = None,
    max_symbols: int = 300,
    min_bars: int = 120,
    horizon: int = 15,
) -> Optional[pd.DataFrame]:
    """
    Main dataset builder.

    Parameters
    ----------
    dt_rolling : dict or None
        If provided, uses this object directly. If None, loads from ROLLING_PATH.
    max_symbols : int
        Max number of symbols to include in the dataset.
    min_bars : int
        Minimum 1m bars required per symbol.
    horizon : int
        Forward-return horizon (in bars) for label construction.

    Returns
    -------
    DataFrame or None
        The full dataset (also written to DATASET_PATH). None if nothing built.
    """
    log(f"[DT] [ml_data_builder_intraday] 🚀 Building intraday dataset for "
        f"{max_symbols} symbols (1m, lookback>={min_bars}m).")

    # Load raw rolling if not provided
    if dt_rolling is None:
        raw = _read_raw_rolling()
    else:
        raw = dt_rolling

    bar_map = _extract_bar_map(raw)
    if not bar_map:
        log("[DT] [ml_data_builder_intraday] ⚠️ No intraday bar data found in rolling.")
        return None

    selected = _select_symbols(bar_map, max_symbols=max_symbols, min_bars=min_bars)
    if not selected:
        log("[DT] [ml_data_builder_intraday] ⚠️ No symbols meet min_bars threshold; dataset empty.")
        return None

    frames: List[pd.DataFrame] = []
    for sym, bars in selected:
        try:
            df_sym = build_symbol_features(sym, bars, horizon=horizon)
            if df_sym is not None and not df_sym.empty:
                frames.append(df_sym)
        except Exception as e:
            log(f"[DT] [ml_data_builder_intraday] ⚠️ feature build failed for {sym}: {e}")

    if not frames:
        log("[DT] [ml_data_builder_intraday] ⚠️ No symbol produced usable feature rows.")
        return None

    dataset = pd.concat(frames, ignore_index=True)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce")
    dataset = dataset.dropna(subset=["timestamp"]).reset_index(drop=True)
    dataset["target_label_15m"] = dataset["target_label_15m"].astype(str)
    dataset = dataset.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Ensure output directory
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset.to_parquet(DATASET_PATH, index=False)
        log(f"[DT] [ml_data_builder_intraday] ✅ Wrote dataset → {DATASET_PATH} "
            f"({len(dataset):,} rows, {dataset.shape[1]} cols).")
        _write_build_log(len(dataset), len(selected), list(dataset.columns))
    except Exception as e:
        log(f"[DT] [ml_data_builder_intraday] ⚠️ Failed to write parquet: {e}")

    return dataset


def _write_build_log(rows: int, symbols: int, columns: List[str]) -> None:
    try:
        BUILD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rows": rows,
            "symbols": symbols,
            "columns": columns,
            "dataset": str(DATASET_PATH),
        }
        with open(BUILD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as err:
        log(f"[DT] [ml_data_builder_intraday] ⚠️ Could not append build log: {err}")


if __name__ == "__main__":
    # Manual test entrypoint
    res = build_intraday_dataset()
    n = 0 if res is None else len(res)
    log(f"[DT] [ml_data_builder_intraday] Finished CLI build with {n:,} rows.")

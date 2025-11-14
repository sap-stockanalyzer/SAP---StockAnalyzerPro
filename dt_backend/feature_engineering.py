# dt_backend/feature_engineering.py — v1.0
"""
Higher-level intraday feature builder for AION day-trading engine.

Responsibilities:
  • Take raw 1m OHLCV bars per symbol
  • Add indicators (via dt_backend.indicators)
  • Build forward returns & 3-class labels (SELL/HOLD/BUY)
  • Add time-based features
  • Mark train/valid splits

Outputs are model-ready rows consumed by train_lightgbm_intraday.py.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from .indicators import compute_indicators

LABEL_ORDER = ["SELL", "HOLD", "BUY"]


def _time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Simple time-of-day features (normalized).
    """
    df = pd.DataFrame(index=idx)
    df["minute_of_day"] = idx.hour * 60 + idx.minute
    df["minute_of_day"] = df["minute_of_day"] / (24 * 60.0)  # 0–1
    df["hour"] = idx.hour / 23.0
    df["day_of_week"] = idx.dayofweek / 6.0
    return df


def _label_forward_returns(
    df: pd.DataFrame,
    horizon: int = 15,
    buy_th: float = 0.002,
    sell_th: float = -0.002,
) -> pd.DataFrame:
    """
    Build forward return labels:

        fwd_ret = close[t+horizon] / close[t] - 1

        BUY  if fwd_ret >= buy_th
        SELL if fwd_ret <= sell_th
        HOLD otherwise
    """
    df = df.copy()
    fwd_price = df["close"].shift(-horizon)
    df["fwd_ret"] = fwd_price / df["close"] - 1.0

    df["label"] = "HOLD"
    df.loc[df["fwd_ret"] >= buy_th, "label"] = "BUY"
    df.loc[df["fwd_ret"] <= sell_th, "label"] = "SELL"

    # Drop rows where forward return is NaN (end-of-series)
    df = df[~df["fwd_ret"].isna()].copy()
    return df


def build_symbol_features(
    sym: str,
    bars: list[Dict[str, Any]],
    horizon: int = 15,
) -> pd.DataFrame:
    """
    Convert raw intraday bars for a single symbol into model-ready features.

    bars: list of dicts like:
        {"t": "...Z", "o": float, "h": float, "l": float, "c": float, "v": int}
    """
    if not bars:
        return pd.DataFrame()

    # Base OHLCV frame
    df = pd.DataFrame(bars)
    # Normalize keys
    rename_map = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    df = df.rename(columns=rename_map)
    # Parse timestamps & sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    # Basic cleaning
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    if df.empty:
        return pd.DataFrame()

    # Indicators
    df = compute_indicators(df)

    # Time features
    tf = _time_features(df.index)
    df = pd.concat([df, tf], axis=1)

    # Labeling
    df = _label_forward_returns(df, horizon=horizon)

    if df.empty:
        return df

    # Symbol + split
    df["symbol"] = sym

    # Simple chronological split: last 10% as validation
    n = len(df)
    split_idx = int(n * 0.9)
    df["split"] = "train"
    if split_idx < n:
        df.iloc[split_idx:, df.columns.get_loc("split")] = "valid"

    df = df.rename(columns={
        "fwd_ret": "target_ret_15m",
        "label": "target_label_15m",
    })

    df = df.reset_index().rename(columns={"index": "timestamp"})
    df["split"] = df["split"].astype(str)

    return df

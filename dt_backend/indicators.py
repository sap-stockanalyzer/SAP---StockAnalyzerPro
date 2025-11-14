# dt_backend/indicators.py — v1.0
"""Low-level intraday indicators for the DT feature pipeline.

Inputs:
    DataFrame with at least: ["open", "high", "low", "close", "volume"]
    index: DatetimeIndex (1m bars)

Outputs:
    DataFrame with added numeric feature columns:
        ret_1, ret_5, ret_15
        vol_20, vol_60
        ema_5, ema_10, ema_20
        rsi_14
        vwap_20
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def _safe_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float)
    return pd.Series(x, dtype=float)


def ema(series: pd.Series, span: int) -> pd.Series:
    s = _safe_series(series)
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    s = _safe_series(series)
    delta = s.diff()

    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, window: int = 20) -> pd.Series:
    h = _safe_series(high)
    l = _safe_series(low)
    c = _safe_series(close)
    v = _safe_series(volume)

    typical = (h + l + c) / 3.0
    cum_pv = (typical * v).rolling(window=window, min_periods=1).sum()
    cum_vol = v.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return cum_pv / cum_vol


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches df with intraday indicators. Does NOT drop any rows.

    Expects columns: open, high, low, close, volume
    """
    if df.empty:
        return df

    df = df.copy()

    # Short horizon returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)

    # Rolling volatility of 1-bar returns
    df["vol_20"] = df["ret_1"].rolling(20, min_periods=5).std()
    df["vol_60"] = df["ret_1"].rolling(60, min_periods=10).std()

    # EMAs
    df["ema_5"] = ema(df["close"], 5)
    df["ema_10"] = ema(df["close"], 10)
    df["ema_20"] = ema(df["close"], 20)

    # RSI
    df["rsi_14"] = rsi(df["close"], 14)

    # VWAP-like rolling window
    df["vwap_20"] = rolling_vwap(df["high"], df["low"], df["close"], df["volume"], 20)

    return df

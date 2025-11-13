# dt_backend/momentum_detector.py — v1.0
# Intraday momentum + VWAP + breakout detectors (feature builders)

from __future__ import annotations
import os, sys, math
import numpy as np
import pandas as pd
from datetime import datetime

# Allow importing backend utilities when run standalone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from backend.data_pipeline import log  # type: ignore
except Exception:
    def log(msg: str): print(msg, flush=True)

from dt_backend.config_dt import DT_PATHS

# ------------------------------- helpers -------------------------------

def _rolling_vwap(df: pd.DataFrame, price_col="close", vol_col="volume", window=30) -> pd.Series:
    """Simple rolling VWAP using price*volume / volume over a short window."""
    pv = (df[price_col].astype(float) * df[vol_col].astype(float)).rolling(window, min_periods=1).sum()
    vv = df[vol_col].astype(float).rolling(window, min_periods=1).sum()
    return pv / vv.replace(0, np.nan)

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def _bollinger(series: pd.Series, window=20, num_std=2.0):
    ma = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, width

# ------------------------------- features -------------------------------

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns (intraday): ['timestamp','open','high','low','close','volume']
    Returns df with added momentums (RSI/MACD/Bollinger/VWAP/Volatility).
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    c = out["close"].astype(float)
    v = out.get("volume", pd.Series(index=out.index, dtype=float)).astype(float)

    # RSI family
    out["rsi_5"] = _rsi(c, 5).clip(0, 100)
    out["rsi_14"] = _rsi(c, 14).clip(0, 100)

    # MACD
    macd, macd_sig, macd_hist = _macd(c, 12, 26, 9)
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    # VWAP (rolling intraday proxy)
    out["vwap_30"] = _rolling_vwap(out, "close", "volume", window=30)
    out["vwap_dev_pct"] = (c - out["vwap_30"]) / out["vwap_30"] * 100.0

    # Bollinger
    ma20, bb_up, bb_lo, bb_w = _bollinger(c, 20, 2.0)
    out["ma20"] = ma20
    out["bb_upper"] = bb_up
    out["bb_lower"] = bb_lo
    out["bb_width"] = bb_w

    # Returns & volatility
    out["ret_1"] = c.pct_change(1) * 100.0
    out["ret_5"] = c.pct_change(5) * 100.0
    out["ret_15"] = c.pct_change(15) * 100.0
    out["volatility_30"] = out["ret_1"].rolling(30, min_periods=5).std(ddof=0)

    # Breakout flags
    out["breakout_high_20"] = (c > out["high"].rolling(20, min_periods=1).max()).astype(int)
    out["breakdown_low_20"] = (c < out["low"].rolling(20, min_periods=1).min()).astype(int)

    # VWAP bounce: cross from below to above (and reverse)
    below = (c < out["vwap_30"]).astype(int)
    cross_up = (below.shift(1) == 1) & (below == 0)
    cross_dn = (below.shift(1) == 0) & (below == 1)
    out["vwap_cross_up"] = cross_up.astype(int)
    out["vwap_cross_dn"] = cross_dn.astype(int)

    # Momentum score (bounded [-1,1])
    z_rsi = (out["rsi_14"] - 50) / 50.0
    z_macd = np.tanh(out["macd_hist"].fillna(0) / (out["volatility_30"].fillna(1e-6) + 1e-6))
    z_vwap = np.tanh(out["vwap_dev_pct"].fillna(0) / 2.5)
    out["momentum_score"] = (0.5*z_rsi + 0.3*z_macd + 0.2*z_vwap).clip(-1, 1)

    return out

# --------------------------- signal snapshots ---------------------------

def latest_snapshot_signals(fe_df: pd.DataFrame) -> dict:
    """
    Consume the last row of a feature-enriched DF to build simple signals.
    Returns a dict (e.g., for ranking or for model features).
    """
    if fe_df is None or fe_df.empty:
        return {}

    row = fe_df.iloc[-1]
    sig = {
        "rsi_5": float(row.get("rsi_5", np.nan)),
        "rsi_14": float(row.get("rsi_14", np.nan)),
        "macd_hist": float(row.get("macd_hist", np.nan)),
        "vwap_dev_pct": float(row.get("vwap_dev_pct", np.nan)),
        "bb_width": float(row.get("bb_width", np.nan)),
        "momentum_score": float(row.get("momentum_score", 0.0)),
        "vwap_cross_up": int(row.get("vwap_cross_up", 0) or 0),
        "vwap_cross_dn": int(row.get("vwap_cross_dn", 0) or 0),
        "breakout_high_20": int(row.get("breakout_high_20", 0) or 0),
        "breakdown_low_20": int(row.get("breakdown_low_20", 0) or 0),
    }
    return sig

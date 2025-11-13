# dt_backend/orderflow_analyzer.py — v1.0
# Approximate order-flow / pressure features from intraday bars

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from backend.data_pipeline import log  # type: ignore
except Exception:
    def log(msg: str): print(msg, flush=True)

from dt_backend.config_dt import DT_PATHS

def _signed_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    VERY simple sign proxy: volume * sign(price change). If you have bid/ask,
    replace with true aggressor side classification.
    """
    ret = close.pct_change().fillna(0.0)
    sign = np.sign(ret).replace(0, np.nan).fillna(0.0)
    return volume.fillna(0.0) * sign

def build_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: ['timestamp','open','high','low','close','volume'].
    Outputs order flow features: volume_pressure, tick_imbalance, slippage, etc.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    c = out["close"].astype(float)
    v = out.get("volume", pd.Series(index=out.index, dtype=float)).astype(float)

    # Signed volume + rolling pressure
    out["signed_vol"] = _signed_volume(c, v)
    out["volume_pressure_10"] = out["signed_vol"].rolling(10, min_periods=3).sum()
    out["volume_pressure_30"] = out["signed_vol"].rolling(30, min_periods=5).sum()

    # Tick imbalance: up moves vs down moves
    uptick = (c.diff() > 0).astype(int)
    downtick = (c.diff() < 0).astype(int)
    up10 = uptick.rolling(10, min_periods=3).sum()
    dn10 = downtick.rolling(10, min_periods=3).sum()
    out["tick_imbalance_10"] = (up10 - dn10) / (up10 + dn10 + 1e-6)

    # VWAP slippage proxy: close vs rolling avg price
    mean_px = c.rolling(20, min_periods=5).mean()
    out["slippage_20bps"] = (c - mean_px) / mean_px * 10000.0  # in basis points

    # Liquidity proxy: volume percentile (short window)
    vol_rank = v.rolling(60, min_periods=10).apply(lambda x: (x.rank().iloc[-1] / max(len(x), 1)), raw=False)
    out["volume_percentile_60"] = vol_rank.fillna(0.5)

    # Final pressure score (bounded)
    z1 = np.tanh(out["volume_pressure_10"].fillna(0) / (v.rolling(10, min_periods=3).mean().fillna(1e-6)))
    z2 = out["tick_imbalance_10"].fillna(0)
    z3 = np.tanh(-out["slippage_20bps"].fillna(0) / 25.0)
    out["orderflow_score"] = (0.5*z1 + 0.3*z2 + 0.2*z3).clip(-1, 1)

    return out

# dt_backend/risk_manager.py — v1.0
# Day-trading risk utilities: position sizing, SL/TP suggestions

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

def atr_like(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """ATR proxy from OHLC series."""
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=3).mean()

def position_size(equity: float, risk_pct: float, stop_distance: float, price: float) -> int:
    """
    Fixed fractional position sizing.
    - equity: total capital
    - risk_pct: fraction at risk (e.g., 0.005 = 0.5%)
    - stop_distance: absolute price distance to stop (e.g., 0.50 dollars)
    """
    if equity <= 0 or risk_pct <= 0 or stop_distance <= 0 or price <= 0:
        return 0
    risk_cap = equity * risk_pct
    shares = int(max(0, np.floor(risk_cap / stop_distance)))
    # Ensure shares correspond to >0 notional
    return int(min(shares, np.floor(equity / price))) if price > 0 else 0

def propose_stops_tps(last_price: float, atr_val: float, side: str, atr_mult_sl=1.5, atr_mult_tp=2.0):
    """
    Suggest stop-loss / take-profit around last price based on ATR.
    side: 'BUY' or 'SELL'
    """
    if any(x is None or not np.isfinite(x) or x <= 0 for x in [last_price, atr_val, atr_mult_sl, atr_mult_tp]):
        return None, None
    if side == "BUY":
        sl = last_price - atr_mult_sl * atr_val
        tp = last_price + atr_mult_tp * atr_val
    else:
        sl = last_price + atr_mult_sl * atr_val
        tp = last_price - atr_mult_tp * atr_val
    return float(sl), float(tp)

def add_risk_columns(df: pd.DataFrame, side_col="pred_label", price_col="close") -> pd.DataFrame:
    """
    Adds 'atr_14', 'sl', 'tp' columns to a feature DF for downstream use.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if {"high","low","close"}.issubset(out.columns):
        out["atr_14"] = atr_like(out["close"], out["high"], out["low"], 14)
    else:
        out["atr_14"] = out.get("ret_1", pd.Series(index=out.index, dtype=float)).rolling(14, min_periods=3).std()
    last_price = out[price_col].astype(float)
    side = out.get(side_col, pd.Series(index=out.index, dtype=object)).astype(str).str.upper().fillna("HOLD")
    sl_list, tp_list = [], []
    for i in out.index:
        p = last_price.iloc[i]
        a = out["atr_14"].iloc[i]
        s = "BUY" if side.iloc[i] not in ("SELL","SHORT") else "SELL"
        sl, tp = propose_stops_tps(p, a if np.isfinite(a) else 0.0, s)
        sl_list.append(sl)
        tp_list.append(tp)
    out["sl"] = sl_list
    out["tp"] = tp_list
    return out

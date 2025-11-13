# dt_backend/trade_simulator.py — v1.0
# Simple same-day simulator to evaluate intraday signals with risk rules.

from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from backend.data_pipeline import log  # type: ignore
except Exception:
    def log(msg: str): print(msg, flush=True)

from dt_backend.config_dt import DT_PATHS
from dt_backend.risk_manager import position_size, propose_stops_tps

LOG_DIR = DT_PATHS["dtml_data"] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _pnl_for_trade(side: str, entry: float, exit_: float, shares: int) -> float:
    if any(x is None for x in [entry, exit_, shares]) or shares <= 0:
        return 0.0
    if side == "BUY":
        return (exit_ - entry) * shares
    else:
        return (entry - exit_) * shares

def simulate_intraday(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    equity: float = 100_000.0,
    risk_pct: float = 0.005,
    max_hold_minutes: int = 60,
) -> dict:
    """
    bars: intraday OHLCV with 'timestamp','open','high','low','close','volume' for a single symbol
    signals: dataframe indexed to same timestamps with columns ['pred_label','confidence']
             pred_label in {'BUY','SELL','HOLD'}
    """
    if bars is None or bars.empty or signals is None or signals.empty:
        return {"trades": [], "summary": {"n": 0, "pnl": 0.0, "win_rate": None}}

    df = bars.copy().reset_index(drop=True)
    sig = signals.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.to_datetime(df.index)

    df["idx"] = range(len(df))
    sig = sig.copy()
    if "timestamp" in sig.columns:
        sig["timestamp"] = pd.to_datetime(sig["timestamp"])
    else:
        sig["timestamp"] = pd.to_datetime(sig.index)

    # align by nearest minute
    merged = pd.merge_asof(sig.sort_values("timestamp"), df[["timestamp","open","high","low","close","volume","idx"]].sort_values("timestamp"), on="timestamp", direction="nearest")
    trades = []
    for _, row in merged.iterrows():
        label = str(row.get("pred_label","HOLD")).upper()
        conf = float(row.get("confidence", 0.0))
        if label not in ("BUY","SELL"):
            continue

        i = int(row["idx"])
        if i >= len(df):
            continue

        entry_price = float(df.loc[i, "close"])
        # quick ATR proxy from last 14 bars
        atr = (df["high"] - df["low"]).rolling(14, min_periods=3).mean().iloc[i]
        side = label
        sl, tp = propose_stops_tps(entry_price, float(atr) if np.isfinite(atr) else 0.2 * entry_price * 0.01, side)
        stop_dist = abs(entry_price - sl) if sl is not None else max(0.01, 0.002*entry_price)
        shares = position_size(equity, risk_pct, stop_dist, entry_price)
        if shares <= 0:
            continue

        # walk forward up to max_hold_minutes or until SL/TP
        exit_price = float(df.loc[min(i + max_hold_minutes, len(df)-1), "close"])
        hit_reason = "timeout"
        for j in range(i+1, min(i + max_hold_minutes + 1, len(df))):
            hi = float(df.loc[j, "high"])
            lo = float(df.loc[j, "low"])
            if side == "BUY":
                if tp is not None and hi >= tp:
                    exit_price = tp; hit_reason = "take_profit"; break
                if sl is not None and lo <= sl:
                    exit_price = sl; hit_reason = "stop_loss"; break
            else:
                if tp is not None and lo <= tp:
                    exit_price = tp; hit_reason = "take_profit"; break
                if sl is not None and hi >= sl:
                    exit_price = sl; hit_reason = "stop_loss"; break

        pnl = _pnl_for_trade(side, entry_price, exit_price, shares)
        trades.append({
            "time": str(df.loc[i, "timestamp"]),
            "side": side,
            "conf": conf,
            "entry": entry_price,
            "exit": exit_price,
            "shares": int(shares),
            "reason": hit_reason,
            "pnl": float(pnl),
        })

    pnl_total = float(np.sum([t["pnl"] for t in trades])) if trades else 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    summary = {
        "n": len(trades),
        "pnl": pnl_total,
        "win_rate": (wins / len(trades)) if trades else None,
        "avg_pnl": (pnl_total / len(trades)) if trades else None,
    }

    # write run log
    try:
        out_path = LOG_DIR / f"sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"trades": trades, "summary": summary}, f, indent=2)
        log(f"🧪 DT backtest log → {out_path}")
    except Exception:
        pass

    return {"trades": trades, "summary": summary}

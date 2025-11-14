# dt_backend/continuous_learning_intraday.py — v1.0
# Incremental reinforcement for intraday model.
# Learns from same-day realized returns or trade simulator outcomes.
# Output → updated metrics in data_dt/brain_intraday.json.gz

from __future__ import annotations
import os, sys, json, gzip, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# import safety shim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from dt_backend.dt_logger import dt_log as log
except Exception:
    def log(msg: str): print(msg, flush=True)

from dt_backend.config_dt import DT_PATHS
try:
    from dt_backend.trade_simulator import simulate_intraday
except Exception:
    simulate_intraday = None

BRAIN_PATH = DT_PATHS["dtbrain"]
DATA_PATH = DT_PATHS["dtml_data"] / "training_data_intraday.parquet"
SIGNALS_PATH = DT_PATHS["dtsignals"] / "intraday_predictions.json"

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def _read_brain() -> dict:
    if not BRAIN_PATH.exists():
        return {}
    try:
        with gzip.open(BRAIN_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"[dt_learn] ⚠️ failed to read brain: {e}")
        return {}

def _save_brain(obj: dict):
    try:
        BRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(BRAIN_PATH) + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, BRAIN_PATH)
        log(f"[dt_learn] 💾 saved brain → {BRAIN_PATH}")
    except Exception as e:
        log(f"[dt_learn] ⚠️ failed to save brain: {e}")

# ---------------------------------------------------------------------
# Core incremental logic
# ---------------------------------------------------------------------
def _calc_outcome_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0.0, "win_rate": None, "avg_pnl": 0.0}
    pnl = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)
    wins = (pnl > 0).sum()
    return {
        "n": int(len(pnl)),
        "pnl": float(pnl.sum()),
        "win_rate": float(wins / len(pnl)) if len(pnl) else None,
        "avg_pnl": float(np.mean(pnl)) if len(pnl) else 0.0,
    }

def train_incremental_intraday(
    *,
    equity: float = 100_000.0,
    risk_pct: float = 0.005,
    max_hold_minutes: int = 60,
) -> dict:
    """
    Uses trade_simulator to evaluate latest signals on recent intraday bars.
    Updates brain_intraday.json.gz with running P&L stats and calibration weight.
    """
    t0 = time.time()
    summary = {"status": "ok", "updated_at": datetime.utcnow().isoformat()}

    # 1️⃣ Load data + predictions
    if simulate_intraday is None:
        log("[dt_learn] ⚠️ trade_simulator unavailable; skipping incremental learning")
        summary["status"] = "skipped_no_sim"
        return summary

    if not DATA_PATH.exists() or not SIGNALS_PATH.exists():
        log("[dt_learn] ⚠️ missing dataset or signals → skip incremental")
        summary["status"] = "skipped_no_data"
        return summary

    try:
        df = pd.read_parquet(DATA_PATH)
        js = json.load(open(SIGNALS_PATH, "r", encoding="utf-8"))
        preds = js.get("rows", []) if isinstance(js, dict) else js
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    except Exception as e:
        log(f"[dt_learn] ⚠️ failed to load artifacts: {e}")
        summary["status"] = "error_load"
        return summary

    # 2️⃣ Simulate each symbol individually (sample subset for speed)
    symbols = sorted({p.get("symbol") for p in preds if p.get("symbol")})
    if len(symbols) > 50:
        symbols = symbols[:50]

    all_trades = []
    for sym in symbols:
        bars = df[df["symbol"] == sym].copy()
        if bars.empty:
            continue
        sig_rows = [p for p in preds if p.get("symbol") == sym]
        if not sig_rows:
            continue
        sig_df = pd.DataFrame(sig_rows)
        res = simulate_intraday(bars, sig_df, equity=equity, risk_pct=risk_pct, max_hold_minutes=max_hold_minutes)
        trades = res.get("trades", [])
        all_trades.extend(trades)

    # 3️⃣ Aggregate results
    stats = _calc_outcome_stats(all_trades)
    log(f"[dt_learn] 🧪 Sim summary → {stats}")

    # 4️⃣ Update brain state
    brain = _read_brain()
    history = brain.get("history", [])
    history.append({"ts": datetime.utcnow().isoformat(), **stats})
    brain["history"] = history[-50:]  # keep last 50 entries

    # running aggregates
    pnl_hist = np.array([x.get("pnl", 0.0) for x in history])
    avg_pnl = pnl_hist.mean() if len(pnl_hist) else 0.0
    win_rates = [x.get("win_rate") for x in history if x.get("win_rate") is not None]
    mean_win_rate = float(np.mean(win_rates)) if win_rates else 0.0

    # dynamic calibration multiplier (affects model weighting or learning rate)
    calib = {
        "confidence_multiplier": round(1.0 + (mean_win_rate - 0.5) * 0.5, 3),
        "hit_rate": round(mean_win_rate, 3),
        "avg_pnl": round(avg_pnl, 4),
        "samples": int(stats["n"]),
    }
    brain["calibration"] = calib
    brain["updated_at"] = datetime.utcnow().isoformat()
    _save_brain(brain)

    duration = time.time() - t0
    summary.update({"duration": round(duration, 2), "calibration": calib})
    log(f"[dt_learn] 🔁 calibration updated → {calib}")
    return summary

if __name__ == "__main__":
    out = train_incremental_intraday()
    print(out)

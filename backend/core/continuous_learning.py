# backend/core/continuous_learning.py — v1.3 (Regression Edition, Horizon-Aware, AION Brain Learning)
"""
AION Analytics — Continuous Learning (Regression Version, Horizon-Aware)

This engine evaluates how well AION’s regression predictions are performing
and continuously adapts the system by:

1) Computing realized returns per horizon using price history:
       realized_ret(h) = (close_now - close_then) / close_then
   where "then" is approximately horizon trading days ago.

2) Comparing realized_ret(h) vs predicted_return (per-symbol per-horizon).

3) Maintaining error windows per horizon:
       • short_window (30 samples)
       • long_window (120 samples)

4) Computing MAE-based drift:
       drift = mae_long - mae_short

   Interpretation (IMPORTANT):
     • drift > 0  → long-window error is higher than short-window error
                   (recent performance improved)
     • drift < 0  → short-window error is higher than long-window error
                   (recent performance degraded) → may trigger retraining

5) Tracking sector-level regression health per horizon:
       n, mae, rmse, bias, avg_confidence

6) Writing all results back to:
       • rolling_brain.json.gz (canonical brain path via PATHS["rolling_brain"])
       • drift_report.json (for dashboards)

UPDATED (v1.3):
    ✅ Adds AION brain updates (behavioral learning):
         - confidence_bias (calibration)
         - risk_bias (risk posture)
         - aggressiveness (exposure posture)

Notes:
- This is SWING engine continuous learning (daily bars). Not intraday.
- Windows-safe: uses Path and json only.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import (
    _read_rolling,
    _read_brain,
    _read_aion_brain,    # ✅ NEW
    save_brain,
    save_aion_brain,     # ✅ NEW
    log,
    safe_float,
)
from backend.core.ai_model.core_training import train_all_models
from backend.core.ai_model.target_builder import HORIZONS


# ======================================================================
# Paths
# ======================================================================

ML_DATA_ROOT: Path = PATHS["ml_data"]
METRICS_ROOT: Path = ML_DATA_ROOT / "metrics"
DRIFT_DIR: Path = METRICS_ROOT / "drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

# Dashboard/back-compat location (kept)
DRIFT_REPORT_FILE: Path = DRIFT_DIR / "drift_report.json"

# New canonical brain-friendly location (da_brains/core/)
ROOT: Path = Path(PATHS.get("root") or Path("."))

BRAINS_ROOT: Path = Path(PATHS.get("brains_root") or (ROOT / "da_brains"))

# ✅ Fix key mismatch safely:
# Your config uses "core_brains" (not "brains_core" or "brains_core_dir")
BRAINS_CORE_DIR: Path = Path(
    PATHS.get("core_brains")
    or PATHS.get("brains_core")
    or (BRAINS_ROOT / "core")
)
BRAINS_CORE_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_REPORT_BRAIN_FILE: Path = BRAINS_CORE_DIR / "drift_report.json"


# ======================================================================
# Window lengths
# ======================================================================

SHORT_WINDOW = 30
LONG_WINDOW = 120


# ======================================================================
# Horizon → approximate trading-day lookback
# ======================================================================

# Conservative approximations (trading days)
HORIZON_DAYS: Dict[str, int] = {
    "1d": 1,
    "3d": 3,
    "1w": 5,
    "2w": 10,
    "4w": 20,
    "13w": 65,
    "26w": 130,
    "52w": 260,
}


# ======================================================================
# Helpers
# ======================================================================

def _ensure_brain(brain: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure schema exists."""
    if not isinstance(brain, dict):
        brain = {}
    brain.setdefault(
        "_meta",
        {
            "updated_at": None,
            "horizon_drift": {},
            "sector_perf": {},
        },
    )
    return brain


def _ensure_aion_brain(aion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure AION brain schema exists.

    AION brain stores BEHAVIOR:
      - confidence_bias: calibration knob
      - risk_bias: risk posture knob
      - aggressiveness: exposure posture knob
      - regime_mods: optional dict per regime label
    """
    if not isinstance(aion, dict):
        aion = {}
    aion.setdefault(
        "_meta",
        {
            "updated_at": None,
            "confidence_bias": 1.0,
            "risk_bias": 1.0,
            "aggressiveness": 1.0,
            "regime_mods": {},  # optional future expansion
            "notes": "AION brain: global behavioral memory. Updated slowly by continuous_learning.",
        },
    )
    if not isinstance(aion.get("_meta"), dict):
        aion["_meta"] = {
            "updated_at": None,
            "confidence_bias": 1.0,
            "risk_bias": 1.0,
            "aggressiveness": 1.0,
            "regime_mods": {},
        }
    if not isinstance(aion["_meta"].get("regime_mods"), dict):
        aion["_meta"]["regime_mods"] = {}
    return aion


def _append_sample(seq: List[Dict[str, Any]], sample: Dict[str, Any], max_len: int) -> List[Dict[str, Any]]:
    seq.append(sample)
    if len(seq) > max_len:
        return seq[-max_len:]
    return seq


def _window_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute regression performance statistics from samples."""
    if not samples:
        return {
            "n": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "avg_error": 0.0,
            "avg_conf": 0.0,
            "hit_ratio_dir": 0.5,
        }

    n = len(samples)

    errs = [safe_float(s.get("error", 0.0)) for s in samples]
    abs_errs = [abs(e) for e in errs]
    rmse = (sum(e * e for e in errs) / n) ** 0.5

    confs = [safe_float(s.get("confidence", 0.0)) for s in samples]

    hits = 0
    for s in samples:
        pr = safe_float(s.get("predicted_return", 0.0))
        rr = safe_float(s.get("realized_ret", 0.0))
        if pr == 0:
            hits += (abs(rr) < 0.002)
        else:
            hits += (pr * rr >= 0)

    return {
        "n": n,
        "mae": sum(abs_errs) / n,
        "rmse": rmse,
        "avg_error": sum(errs) / n,
        "avg_conf": sum(confs) / n,
        "hit_ratio_dir": hits / n,
    }


def _sorted_history(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    hist = node.get("history") or []
    if not isinstance(hist, list):
        return []
    return sorted(hist, key=lambda b: str(b.get("date") or ""))


def _realized_return_for_horizon(hist: List[Dict[str, Any]], lookback_days: int) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute realized return using (last_close - close_then)/close_then,
    where close_then is lookback_days bars before the last bar.

    Returns (realized_ret, ts_of_last).
    """
    if not hist:
        return None, None
    if lookback_days <= 0:
        return None, None
    if len(hist) <= lookback_days:
        return None, None

    last = hist[-1]
    then = hist[-(lookback_days + 1)]  # +1 because last is day 0

    c_then = safe_float(then.get("close"))
    c_last = safe_float(last.get("close"))
    if c_then <= 0:
        return None, None

    realized = (c_last - c_then) / c_then
    ts = last.get("date") or datetime.now(TIMEZONE).isoformat()
    return realized, ts


# ======================================================================
# MAIN ENGINE
# ======================================================================

def run_continuous_learning() -> Dict[str, Any]:
    log("[continuous_learning] 🧠 Running regression continuous learning (v1.3, horizon-aware + AION brain)…")

    rolling = _read_rolling() or {}
    brain = _ensure_brain(_read_brain() or {})

    # ✅ NEW: AION brain load/ensure
    aion = _ensure_aion_brain(_read_aion_brain() or {})
    aion_meta = aion.get("_meta", {})

    updated_symbols = 0

    horizon_drifts: Dict[str, List[float]] = {}
    sector_acc: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ✅ NEW: global direction-hit aggregation for primary horizon (behavior calibration)
    primary_h = "1w"
    global_hit_n = 0
    global_hit_sum = 0

    for sym, node in rolling.items():
        if str(sym).startswith("_"):
            continue
        if not isinstance(node, dict):
            continue

        hist = _sorted_history(node)
        if len(hist) < 3:
            continue

        sector = (node.get("sector") or "").upper() or "UNKNOWN"

        bnode = brain.get(sym, {})
        if not isinstance(bnode, dict):
            bnode = {}

        hperf = bnode.get("horizon_perf", {})
        if not isinstance(hperf, dict):
            hperf = {}

        preds = node.get("predictions") or {}
        if not isinstance(preds, dict) or not preds:
            brain[sym] = bnode
            continue

        any_horizon_updated = False

        for h in HORIZONS:
            block = preds.get(h)
            if not isinstance(block, dict):
                continue

            lookback = HORIZON_DAYS.get(h)
            if not lookback:
                continue

            realized_ret, ts = _realized_return_for_horizon(hist, lookback)
            if realized_ret is None:
                continue

            pred_ret = safe_float(block.get("predicted_return", 0.0))
            conf = safe_float(block.get("confidence", 0.0))

            error = realized_ret - pred_ret
            abs_err = abs(error)

            hp = hperf.get(h, {})
            if not isinstance(hp, dict):
                hp = {}

            sw = hp.get("short_window", [])
            lw = hp.get("long_window", [])
            if not isinstance(sw, list):
                sw = []
            if not isinstance(lw, list):
                lw = []

            sample = {
                "ts": ts,
                "horizon": h,
                "predicted_return": pred_ret,
                "realized_ret": realized_ret,
                "error": error,
                "confidence": conf,
            }

            sw = _append_sample(sw, sample, SHORT_WINDOW)
            lw = _append_sample(lw, sample, LONG_WINDOW)

            sw_stats = _window_stats(sw)
            lw_stats = _window_stats(lw)

            drift = float(lw_stats["mae"] - sw_stats["mae"])

            hp["short_window"] = sw
            hp["long_window"] = lw
            hp["short_stats"] = sw_stats
            hp["long_stats"] = lw_stats
            hp["drift"] = drift

            horizon_drifts.setdefault(h, []).append(drift)

            # ✅ NEW: accumulate global directional accuracy for primary horizon
            if h == primary_h and sw_stats.get("n", 0) >= 10:
                # use short_window direction hit ratio (more responsive)
                hr = safe_float(sw_stats.get("hit_ratio_dir", 0.5))
                global_hit_sum += hr
                global_hit_n += 1

            sec = sector_acc.setdefault(sector, {})
            acc = sec.setdefault(
                h,
                {
                    "n": 0.0,
                    "abs_err_sum": 0.0,
                    "err_sum": 0.0,
                    "rmse_sum": 0.0,
                    "conf_sum": 0.0,
                },
            )

            acc["n"] += 1.0
            acc["abs_err_sum"] += abs_err
            acc["err_sum"] += error
            acc["rmse_sum"] += error * error
            acc["conf_sum"] += conf

            hperf[h] = hp
            any_horizon_updated = True

        bnode["horizon_perf"] = hperf
        brain[sym] = bnode

        if any_horizon_updated:
            updated_symbols += 1

    # ==================================================================
    # Global drift summary
    # ==================================================================
    meta = brain["_meta"]
    horizon_out: Dict[str, Any] = {}
    severe = False

    for h, arr in horizon_drifts.items():
        if not arr:
            continue
        avg = float(sum(arr) / len(arr))

        retrain = False
        if avg < -0.005:
            retrain = True
            severe = True

        horizon_out[h] = {
            "avg_drift": avg,
            "n": int(len(arr)),
            "retrain_recommended": bool(retrain),
        }

    meta["horizon_drift"] = horizon_out
    meta["updated_at"] = datetime.now(TIMEZONE).isoformat()

    # ==================================================================
    # Sector regression performance
    # ==================================================================
    sector_out: Dict[str, Any] = {}
    for sector, per_h in sector_acc.items():
        sec_entry: Dict[str, Any] = {}
        for h, acc in per_h.items():
            n = acc.get("n", 0.0) or 0.0
            if n <= 0:
                continue

            mae = (acc["abs_err_sum"] / n) if n else 0.0
            rmse = (acc["rmse_sum"] / n) ** 0.5 if n else 0.0
            bias = (acc["err_sum"] / n) if n else 0.0
            avg_conf = (acc["conf_sum"] / n) if n else 0.0

            sec_entry[h] = {
                "n": int(n),
                "mae": float(mae),
                "rmse": float(rmse),
                "bias": float(bias),
                "avg_conf": float(avg_conf),
            }

        if sec_entry:
            sector_out[sector] = sec_entry

    meta["sector_perf"] = sector_out

    # ==================================================================
    # ✅ AION BRAIN UPDATE (behavioral learning — slow + bounded)
    # ==================================================================
    # Compute global directional hit ratio for primary horizon
    global_hit_ratio = (global_hit_sum / global_hit_n) if global_hit_n > 0 else 0.5

    # Use global drift on primary horizon to modulate aggressiveness slightly
    primary_drift = safe_float((horizon_out.get(primary_h) or {}).get("avg_drift", 0.0))

    # Pull existing knobs
    cb = safe_float(aion_meta.get("confidence_bias", 1.0)) or 1.0
    rb = safe_float(aion_meta.get("risk_bias", 1.0)) or 1.0
    ag = safe_float(aion_meta.get("aggressiveness", 1.0)) or 1.0

    # Confidence calibration:
    # If direction hit ratio is low, damp confidence a bit; if high, allow slight increase.
    # Very conservative step sizes to avoid thrash.
    if global_hit_n >= 25:
        if global_hit_ratio < 0.50:
            cb *= 0.985
        elif global_hit_ratio > 0.58:
            cb *= 1.008

    # Risk posture:
    # If severe degradation, reduce risk slightly; if stable, slowly relax back to 1.0
    if severe:
        rb *= 0.97
    else:
        # gentle drift back toward neutral
        if rb < 1.0:
            rb *= 1.005
        elif rb > 1.0:
            rb *= 0.997

    # Aggressiveness:
    # If primary drift strongly negative (recent worse), reduce exposure; if positive, slight increase.
    if primary_drift < -0.005:
        ag *= 0.985
    elif primary_drift > 0.003:
        ag *= 1.006

    # Bounds (prevent runaway)
    cb = float(max(0.80, min(1.20, cb)))
    rb = float(max(0.70, min(1.30, rb)))
    ag = float(max(0.70, min(1.30, ag)))

    aion_meta["confidence_bias"] = cb
    aion_meta["risk_bias"] = rb
    aion_meta["aggressiveness"] = ag
    aion_meta["updated_at"] = datetime.now(TIMEZONE).isoformat()
    aion_meta["learning_snapshot"] = {
        "primary_horizon": primary_h,
        "global_hit_ratio": float(global_hit_ratio),
        "global_hit_n": int(global_hit_n),
        "primary_avg_drift": float(primary_drift),
        "severe_degradation": bool(severe),
    }

    aion["_meta"] = aion_meta

    # ==================================================================
    # Save results (brain goes to canonical PATHS["rolling_brain"])
    # ==================================================================
    save_brain(brain)

    # ✅ Save AION brain
    save_aion_brain(aion)

    # ==================================================================
    # Drift report files (keep old + write canonical brain copy)
    # ==================================================================
    payload = {
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "drift_by_horizon": horizon_out,
        "horizon_drift": horizon_out,  # back-compat
        "sector_perf": sector_out,
    }

    try:
        DRIFT_REPORT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"[continuous_learning] 📊 Drift report updated → {DRIFT_REPORT_FILE}")
    except Exception as e:
        log(f"[continuous_learning] ⚠️ Failed to write drift_report (ml_data): {e}")

    try:
        DRIFT_REPORT_BRAIN_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"[continuous_learning] 🧠 Drift report updated → {DRIFT_REPORT_BRAIN_FILE}")
    except Exception as e:
        log(f"[continuous_learning] ⚠️ Failed to write drift_report (da_brains): {e}")

    # ==================================================================
    # Early retrain trigger (conservative)
    # ==================================================================
    retrain_result: Dict[str, Any] = {}

    if severe:
        log("[continuous_learning] 🔁 Severe regression degradation — triggering early retrain…")
        try:
            retrain_result = train_all_models(
                dataset_name="training_data_daily.parquet",
                use_optuna=False,
                n_trials=0,
            )
        except Exception as e:
            retrain_result = {"error": str(e)}
            log(f"[continuous_learning] ⚠️ Early retrain failed: {e}")

    log(f"[continuous_learning] ✅ Updated {updated_symbols} symbols.")
    log(f"[continuous_learning] 🌡 Horizon drift: {horizon_out}")
    log(f"[continuous_learning] 🏷 Sector regression health: {len(sector_out)} sectors.")
    log(f"[continuous_learning] 🧠 AION brain meta: cb={cb:.4f}, rb={rb:.4f}, ag={ag:.4f}, hit={global_hit_ratio:.3f} n={global_hit_n}")

    return {
        "symbols_updated": int(updated_symbols),
        "horizon_drift": horizon_out,
        "sector_perf": sector_out,
        "early_retrain_triggered": bool(severe),
        "early_retrain_result": retrain_result,
        "aion_brain_meta": aion_meta,
    }


# ==========================================================
# Adaptive Retraining Logic (new for ML Pipeline Feedback)
# ==========================================================

# Configuration constants for adaptive retraining
SAMPLE_SYMBOLS_LIMIT = 50  # Maximum symbols to check for performance
UNDERPERFORMING_THRESHOLD_SHARPE = 0.5  # Sharpe ratio below which symbol is underperforming
UNDERPERFORMING_THRESHOLD_RATIO = 0.30  # Fraction of symbols underperforming that triggers retraining
PERFORMANCE_WINDOW_DAYS = 3  # Days to look back for Sharpe calculation


def should_retrain_models(rolling: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if models should be retrained based on per-symbol performance.
    
    Triggers retraining if >30% of symbols have Sharpe < 0.5 over last 3 days.
    
    Args:
        rolling: Optional rolling data dict (will read if not provided)
        
    Returns:
        True if retraining recommended, False otherwise
    """
    try:
        from backend.services.model_performance_tracker import ModelPerformanceTracker
        
        tracker = ModelPerformanceTracker()
        
        # Get list of active symbols
        if rolling is None:
            rolling = _read_rolling()
        
        if not rolling or not isinstance(rolling, dict):
            return False
        
        symbols = [
            s for s in rolling.keys() 
            if not str(s).startswith("_")
        ][:SAMPLE_SYMBOLS_LIMIT]  # Sample to avoid excessive computation
        
        if not symbols:
            return False
        
        underperforming_symbols = []
        
        for symbol in symbols:
            sharpe = tracker.get_rolling_sharpe(symbol, days=PERFORMANCE_WINDOW_DAYS)
            
            if sharpe < UNDERPERFORMING_THRESHOLD_SHARPE:
                underperforming_symbols.append({
                    "symbol": symbol,
                    "sharpe": sharpe,
                })
        
        # If >threshold of symbols underperforming, trigger retraining
        underperforming_ratio = len(underperforming_symbols) / len(symbols)
        
        if underperforming_ratio > UNDERPERFORMING_THRESHOLD_RATIO:
            log(
                f"[continuous_learning] 🚨 Model underperforming: "
                f"{len(underperforming_symbols)}/{len(symbols)} symbols "
                f"({underperforming_ratio:.1%}) with Sharpe < {UNDERPERFORMING_THRESHOLD_SHARPE}"
            )
            return True
        
        return False
        
    except Exception as e:
        log(f"[continuous_learning] ⚠️ Failed to check retraining trigger: {e}")
        return False


def run_adaptive_retraining(rolling: Optional[Dict[str, Any]] = None):
    """
    Check and trigger retraining if needed.
    
    This should be called periodically (e.g., from nightly job or intraday monitor).
    
    Args:
        rolling: Optional rolling data dict (will read if not provided)
    """
    if should_retrain_models(rolling):
        log("[continuous_learning] 🧠 Adaptive retraining recommended - trigger via nightly job")
        # Note: Actual retraining is orchestrated by nightly_job.py
        # This function just checks and logs the recommendation
        return True
    return False


if __name__ == "__main__":
    out = run_continuous_learning()
    print(json.dumps(out, indent=2))
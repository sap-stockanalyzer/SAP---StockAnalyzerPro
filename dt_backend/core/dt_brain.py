"""dt_backend/core/dt_brain.py

DT brain = durable learning state for the intraday system.

Philosophy
----------
* rolling (da_brains/intraday/rolling_intraday.json.gz) is *working memory*.
  It is overwritten constantly and is safe to clear at end-of-day.
* dt_brain (da_brains/core/dt_brain.json.gz) is *long-lived memory*.
  It stores performance aggregates, calibration hints, and policy/execution meta.

This module keeps the first version intentionally small: it provides safe
read/write helpers and an update hook that can be expanded over time.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from .config_dt import DT_PATHS
from .logger_dt import log


def _brain_path() -> Path:
    return Path(DT_PATHS.get("dt_brain_file") or (Path(DT_PATHS["da_brains"]) / "core" / "dt_brain.json.gz"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_dt_brain() -> Dict[str, Any]:
    """Best-effort brain read. Never raises."""
    p = _brain_path()
    if not p.exists():
        return {"_meta": {"created_at": _utc_now_iso(), "version": "dt_brain_v1"}, "symbols": {}}
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {"_meta": {"created_at": _utc_now_iso(), "version": "dt_brain_v1"}, "symbols": {}}
        obj.setdefault("_meta", {})
        obj.setdefault("symbols", {})
        return obj
    except Exception as e:
        log(f"[dt_brain] ⚠️ failed reading {p}: {e}")
        return {"_meta": {"created_at": _utc_now_iso(), "version": "dt_brain_v1"}, "symbols": {}}


def write_dt_brain(brain: Dict[str, Any]) -> None:
    """Atomic brain write. Best-effort; never raises."""
    p = _brain_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    try:
        if not isinstance(brain, dict):
            brain = {}
        brain.setdefault("_meta", {})
        meta = brain.get("_meta")
        if isinstance(meta, dict):
            meta.setdefault("version", "dt_brain_v1")
            meta["updated_at"] = _utc_now_iso()
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(brain, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
    except Exception as e:
        log(f"[dt_brain] ⚠️ failed writing {p}: {e}")


def update_dt_brain_from_rolling(rolling: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract end-of-day learning signals from rolling into dt_brain.

    Current v1 behavior (safe + minimal):
    * Persist the latest execution audit per symbol (rolling[s]['exec_dt'])
      under brain['symbols'][s]['last_exec_dt'].
    * Persist a small daily counter of fills.

    This is intentionally conservative; you can extend it later to incorporate
    PnL, hit rates, calibration, slippage, etc.

    Returns: (brain_after, summary)
    """
    brain = read_dt_brain()
    sym_store = brain.get("symbols")
    if not isinstance(sym_store, dict):
        sym_store = {}
        brain["symbols"] = sym_store

    fills = 0
    audited = 0

    for sym, node in (rolling or {}).items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue

        exec_dt = node.get("exec_dt")
        if not isinstance(exec_dt, dict) or not exec_dt:
            continue

        audited += 1
        last_res = exec_dt.get("last_result")
        if isinstance(last_res, dict) and str(last_res.get("status") or "").lower() == "filled":
            fills += 1

        srec = sym_store.get(sym)
        if not isinstance(srec, dict):
            srec = {}
        srec["last_exec_dt"] = exec_dt
        srec["last_seen_utc"] = _utc_now_iso()
        sym_store[sym] = srec

    # global meta counters (very small / safe)
    meta = brain.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        brain["_meta"] = meta
    meta.setdefault("version", "dt_brain_v1")
    meta["updated_at"] = _utc_now_iso()
    meta["last_eod_audited_symbols"] = int(audited)
    meta["last_eod_fills"] = int(fills)

    write_dt_brain(brain)
    summary = {"status": "ok", "audited": int(audited), "fills": int(fills), "brain_path": str(_brain_path())}
    return brain, summary


# ============================================================
#  DT BRAIN KNOB ADJUSTER (Phase 1 Continuous Learning)
# ============================================================

import numpy as np
from typing import List, Tuple

# Knob configuration with defaults and valid ranges
KNOB_CONFIG = {
    "DT_EXEC_MIN_CONF": {"default": 0.25, "range": [0.15, 0.40]},
    "DT_MAX_POSITIONS": {"default": 3, "range": [1, 6]},
    "DT_STOP_LOSS_PCT": {"default": 0.02, "range": [0.01, 0.04]},
    "DT_TAKE_PROFIT_PCT": {"default": 0.04, "range": [0.02, 0.08]},
    "DT_POSITION_SIZE_BASE_USD": {"default": 1000, "range": [500, 2000]},
    "DT_MAX_ORDERS_PER_CYCLE": {"default": 3, "range": [1, 5]},
    "DT_MIN_TRADE_GAP_MINUTES": {"default": 15, "range": [5, 30]},
}


class DTBrain:
    """Day trading brain - learns from performance and adjusts knobs."""
    
    def __init__(self, brain_path: Path = None):
        if brain_path is None:
            brain_path = _brain_path().parent
        self.brain_path = Path(brain_path)
        self.brain_file = _brain_path()
        self.adjustments_log = self.brain_path / "knob_adjustments.jsonl"
    
    def update(self) -> Dict[str, Any]:
        """Main update cycle - analyze performance and adjust knobs."""
        try:
            perf = self._get_recent_performance()
            regime = self._get_current_regime()
            
            adjustments = []
            
            # === CONFIDENCE THRESHOLD ===
            win_rate = perf.get("win_rate", 0.0)
            avg_confidence = perf.get("avg_confidence", 0.0)
            
            if win_rate >= 0.60 and avg_confidence >= 0.70:
                adjustments.append(("DT_EXEC_MIN_CONF", 0.22, "high_accuracy"))
            elif win_rate < 0.45:
                adjustments.append(("DT_EXEC_MIN_CONF", 0.32, "low_win_rate"))
            
            # === MAX POSITIONS ===
            sharpe = perf.get("sharpe_ratio", 0.0)
            profit_factor = perf.get("profit_factor", 0.0)
            drawdown = perf.get("drawdown_pct", 0.0)
            
            if sharpe > 2.0 and profit_factor > 1.5:
                adjustments.append(("DT_MAX_POSITIONS", 5, "strong_performance"))
            elif drawdown > 5.0:
                adjustments.append(("DT_MAX_POSITIONS", 2, "drawdown"))
            
            # === STOP LOSS ===
            avg_loss = perf.get("avg_loss", 0.0)
            
            if avg_loss > -0.015:
                adjustments.append(("DT_STOP_LOSS_PCT", 0.015, "tight_losses"))
            elif avg_loss < -0.03:
                adjustments.append(("DT_STOP_LOSS_PCT", 0.025, "wide_losses"))
            
            # === TAKE PROFIT ===
            avg_win_hold = perf.get("avg_win_hold_time", 0)
            
            if avg_win_hold > 60:
                adjustments.append(("DT_TAKE_PROFIT_PCT", 0.06, "runners"))
            elif avg_win_hold < 20:
                adjustments.append(("DT_TAKE_PROFIT_PCT", 0.025, "scalps"))
            
            # === POSITION SIZING ===
            consecutive_wins = perf.get("consecutive_wins", 0)
            consecutive_losses = perf.get("consecutive_losses", 0)
            
            if consecutive_wins >= 5:
                adjustments.append(("DT_POSITION_SIZE_BASE_USD", 1500, "hot_streak"))
            elif consecutive_losses >= 3:
                adjustments.append(("DT_POSITION_SIZE_BASE_USD", 700, "cold_streak"))
            
            # === REGIME ADJUSTMENTS ===
            if regime == "volatile":
                adjustments.append(("DT_STOP_LOSS_PCT", 0.03, "volatile_regime"))
                adjustments.append(("DT_MAX_POSITIONS", 2, "volatile_regime"))
            elif regime == "range":
                adjustments.append(("DT_STOP_LOSS_PCT", 0.015, "range_regime"))
                adjustments.append(("DT_MAX_ORDERS_PER_CYCLE", 4, "range_regime"))
            
            # Apply adjustments with EMA smoothing
            applied = []
            for knob, target, reason in adjustments:
                if self._apply_adjustment(knob, target, reason):
                    applied.append((knob, target, reason))
            
            self._save_brain_knobs()
            
            return {
                "status": "success",
                "adjustments_applied": len(applied),
                "adjustments": applied,
                "performance": perf,
            }
            
        except Exception as e:
            log(f"[dt_brain] ⚠️ Error in update: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance metrics."""
        try:
            from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
            analyzer = TradeOutcomeAnalyzer()
            return analyzer.get_performance_window(days=7)
        except Exception as e:
            log(f"[dt_brain] ⚠️ Error getting performance: {e}")
            return {}
    
    def _get_current_regime(self) -> str:
        """Get current market regime."""
        try:
            brain = read_dt_brain()
            return brain.get("_meta", {}).get("regime", "unknown")
        except Exception:
            return "unknown"
    
    def _get_current_value(self, knob: str) -> float:
        """Get current knob value from brain."""
        try:
            brain = read_dt_brain()
            knobs = brain.get("knobs", {})
            config = KNOB_CONFIG.get(knob, {})
            return float(knobs.get(knob, config.get("default", 0.0)))
        except Exception:
            return KNOB_CONFIG.get(knob, {}).get("default", 0.0)
    
    def _apply_adjustment(self, knob: str, target: float, reason: str) -> bool:
        """Apply knob adjustment with EMA smoothing (20% per update)."""
        try:
            current = self._get_current_value(knob)
            config = KNOB_CONFIG.get(knob, {})
            
            if not config:
                return False
            
            # EMA: 80% old + 20% target
            new_value = 0.8 * current + 0.2 * target
            min_val, max_val = config["range"]
            new_value = float(np.clip(new_value, min_val, max_val))
            
            # Only apply if change is significant
            if abs(new_value - current) > 0.01:
                log(f"[dt_brain] 🧠 {knob}: {current:.3f} → {new_value:.3f} ({reason})")
                self._set_knob(knob, new_value)
                self._log_adjustment(knob, current, new_value, reason)
                return True
            
            return False
            
        except Exception as e:
            log(f"[dt_brain] ⚠️ Error applying adjustment: {e}")
            return False
    
    def _set_knob(self, knob: str, value: float) -> None:
        """Set knob value in brain."""
        try:
            brain = read_dt_brain()
            brain.setdefault("knobs", {})
            brain["knobs"][knob] = float(value)
            write_dt_brain(brain)
        except Exception as e:
            log(f"[dt_brain] ⚠️ Error setting knob: {e}")
    
    def _save_brain_knobs(self) -> None:
        """Ensure knobs are persisted."""
        # Already saved in _set_knob
        pass
    
    def _log_adjustment(self, knob: str, old_val: float, new_val: float, reason: str) -> None:
        """Log knob adjustment to file."""
        try:
            record = {
                "timestamp": _utc_now_iso(),
                "knob": knob,
                "old_value": old_val,
                "new_value": new_val,
                "reason": reason,
            }
            
            with open(self.adjustments_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        except Exception as e:
            log(f"[dt_brain] ⚠️ Error logging adjustment: {e}")


def update_dt_brain() -> Dict[str, Any]:
    """Entry point for post-market brain update."""
    try:
        brain = DTBrain()
        return brain.update()
    except Exception as e:
        log(f"[dt_brain] ⚠️ Error in update_dt_brain: {e}")
        return {"status": "error", "error": str(e)}

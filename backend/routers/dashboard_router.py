# backend/routers/dashboard_router.py
"""
Dashboard API Router — AION Analytics

Purpose:
    Provide UI-facing endpoints expected by the dashboard:
      • GET /dashboard/metrics
      • GET /dashboard/top/{horizon}

This router is a thin adapter layer over existing backend artifacts.
It does NOT recompute anything — it only reads what nightly + intraday
already produce.

Safe behaviors:
    - Missing files return graceful defaults
    - Always returns ranked arrays (never dicts)
"""

from __future__ import annotations

import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException

from backend.core.config import PATHS
from backend.core.data_pipeline import log, safe_float

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ACCURACY_DIR = Path(PATHS.get("accuracy", Path("ml_data") / "metrics" / "accuracy"))
ACCURACY_LATEST = ACCURACY_DIR / "accuracy_latest.json"

INSIGHTS_DIR = Path(PATHS.get("insights", Path("ml_data") / "insights"))
PRED_LATEST = INSIGHTS_DIR / "predictions_latest.json"

STOCK_CACHE = Path(PATHS.get("stock_cache", "data_cache"))
BOT_STATE_DIR = STOCK_CACHE / "master" / "bot"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        log(f"[dashboard_router] ⚠️ Failed reading {path}: {e}")
        return {}


def _extract_accuracy_30d(latest: Dict[str, Any]) -> float:
    """
    UI expects a single accuracy_30d number.
    We compute a weighted mean across horizons where available.
    """
    horizons = latest.get("horizons")
    if not isinstance(horizons, dict):
        return 0.0

    vals = []
    weights = []

    for h, blk in horizons.items():
        windows = blk.get("windows") if isinstance(blk, dict) else None
        w30 = windows.get("30") if isinstance(windows, dict) else None
        if not isinstance(w30, dict) or w30.get("status") != "ok":
            continue

        acc = safe_float(w30.get("directional_accuracy", 0.0))
        n = int(w30.get("n", 0) or 0)

        if acc > 0 and n > 0:
            vals.append(acc * n)
            weights.append(n)

    if not weights:
        return 0.0

    return sum(vals) / sum(weights)


def _load_gz_dict(path: Path) -> Optional[dict]:
    """Load a gzipped JSON file."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_bot_state(path: Path) -> Optional[dict]:
    """Load bot state from either .gz or plain .json."""
    if not path.exists():
        return None
    if path.suffix.endswith(".gz"):
        return _load_gz_dict(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _calculate_execution_accuracy() -> Optional[float]:
    """
    Calculate execution accuracy from closed positions in bot states.
    
    Returns:
        Win rate (0.0 to 1.0) or None if insufficient data
    """
    try:
        if not BOT_STATE_DIR.exists():
            return None
        
        # Find all bot state files
        state_files = list(BOT_STATE_DIR.glob("rolling_*.json.gz")) + list(BOT_STATE_DIR.glob("rolling_*.json"))
        
        all_closed_positions = []
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for state_file in state_files:
            state = _load_bot_state(state_file)
            if not isinstance(state, dict):
                continue
            
            closed_positions = state.get("closed_positions") or []
            if isinstance(closed_positions, list):
                all_closed_positions.extend(closed_positions)
        
        # Filter to last 30 days and count wins
        wins = 0
        total = 0
        
        for pos in all_closed_positions:
            if not isinstance(pos, dict):
                continue
            
            # Check exit date
            exit_date_str = pos.get("exit_date") or pos.get("close_date") or ""
            if exit_date_str:
                try:
                    exit_date = datetime.fromisoformat(exit_date_str.replace("Z", "+00:00"))
                    if exit_date < cutoff_date:
                        continue
                except Exception:
                    pass
            
            # Determine if position was a win
            pnl = pos.get("pnl")
            entry_price = pos.get("entry_price") or pos.get("avg_entry_price")
            exit_price = pos.get("exit_price") or pos.get("close_price")
            
            # Try using pnl first
            if pnl is not None:
                try:
                    if float(pnl) > 0:
                        wins += 1
                    total += 1
                    continue
                except Exception:
                    pass
            
            # Fall back to comparing prices
            if entry_price is not None and exit_price is not None:
                try:
                    if float(exit_price) > float(entry_price):
                        wins += 1
                    total += 1
                except Exception:
                    pass
        
        # Require at least 10 trades for meaningful accuracy
        if total < 10:
            return None
        
        return wins / total if total > 0 else None
        
    except Exception as e:
        log(f"[dashboard_router] ⚠️ Failed calculating execution accuracy: {e}")
        return None


def _load_ranked_predictions(horizon: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Returns ranked predictions for a given horizon.
    Output is ALWAYS a list sorted by score descending.
    """
    js = _read_json(PRED_LATEST)
    rows = js.get("predictions")

    if not isinstance(rows, list):
        return []

    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("horizon")) != horizon:
            continue
        out.append(r)

    out.sort(key=lambda x: safe_float(x.get("score", 0.0)), reverse=True)
    return out[:limit]


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@router.get("/metrics")
def dashboard_metrics() -> Dict[str, Any]:
    """
    Dashboard metrics summary with REAL trading accuracy.

    Response:
      {
        "accuracy_30d": 0.62,  # Combined prediction + execution accuracy
        "model_accuracy": 0.57,  # Model prediction accuracy only
        "execution_accuracy": 0.68,  # Trade execution win rate
        "updated_at": "2026-01-15T12:00:00Z"
      }
    """
    latest = _read_json(ACCURACY_LATEST)

    # Get model accuracy
    model_acc = _extract_accuracy_30d(latest)
    updated = latest.get("updated_at")

    # Get execution accuracy from closed trades
    exec_acc = _calculate_execution_accuracy()
    
    # Calculate combined accuracy
    # Weight execution higher (60%) since it's real money
    if exec_acc is not None:
        combined_acc = (model_acc * 0.4) + (exec_acc * 0.6)
    else:
        # If no execution data, use model accuracy only
        combined_acc = model_acc

    result = {
        "accuracy_30d": round(float(combined_acc), 4),
        "model_accuracy": round(float(model_acc), 4),
        "updated_at": updated,
    }
    
    # Only include execution_accuracy if we have data
    if exec_acc is not None:
        result["execution_accuracy"] = round(float(exec_acc), 4)
    
    return result


@router.get("/top/{horizon}")
def dashboard_top_predictions(horizon: str, limit: int = 50) -> Dict[str, Any]:
    """
    Top-ranked predictions for a horizon.

    Horizon examples:
        1d, 3d, 1w, 2w, 4w, 13w, 26w, 52w
    """
    rows = _load_ranked_predictions(horizon, limit=limit)

    return {
        "horizon": horizon,
        "count": len(rows),
        "results": rows,
    }

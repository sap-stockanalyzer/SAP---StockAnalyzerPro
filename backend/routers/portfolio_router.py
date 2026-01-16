# backend/routers/portfolio_router.py
"""
Portfolio API Router — AION Analytics

Purpose:
    Provide portfolio-specific endpoints:
      • GET /portfolio/holdings/top/{horizon}

This router reads bot states and calculates actual portfolio performance.
"""

from __future__ import annotations

import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException

from backend.core.config import PATHS

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ML_DATA = Path(PATHS.get("ml_data", "ml_data"))
STOCK_CACHE = Path(PATHS.get("stock_cache", "data_cache"))
BOT_STATE_DIR = STOCK_CACHE / "master" / "bot"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_gz_dict(path: Path) -> Optional[dict]:
    """Load a gzipped JSON file."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_json(path: Path) -> Optional[dict]:
    """Load a plain JSON file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        v = float(x)
        return v if v == v else default  # NaN check
    except Exception:
        return default


def _load_bot_state(path: Path) -> Optional[dict]:
    """Load bot state from either .gz or plain .json."""
    if not path.exists():
        return None
    if path.suffix.endswith(".gz"):
        return _load_gz_dict(path)
    return _load_json(path)


def _load_prices_from_rolling() -> Dict[str, float]:
    """
    Load latest prices from core rolling.json.gz.
    
    Returns:
        Dict mapping ticker symbols to current prices
    """
    data: Dict[str, Any] = {}

    # Try to use core helper first
    try:
        from backend.core.data_pipeline import _read_rolling  # type: ignore
        data = _read_rolling() or {}
    except Exception:
        # Fallback: direct file read from ml_data/rolling.json.gz
        roll_path = ML_DATA / "rolling.json.gz"
        if roll_path.exists():
            maybe = _load_gz_dict(roll_path)
            if isinstance(maybe, dict):
                data = maybe

    if not isinstance(data, dict):
        return {}

    prices: Dict[str, float] = {}
    for sym, node in data.items():
        if isinstance(sym, str) and sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue

        price = (
            node.get("price")
            or node.get("last")
            or node.get("close")
            or node.get("c")
        )
        try:
            price_f = float(price)
        except Exception:
            continue
        if price_f > 0:
            prices[str(sym).upper()] = price_f
    return prices


def _calculate_days_held(entry_date_str: str) -> int:
    """Calculate number of days a position has been held."""
    try:
        entry_date = datetime.fromisoformat(entry_date_str.replace("Z", "+00:00"))
        now = datetime.now(entry_date.tzinfo or None)
        delta = now - entry_date
        return max(0, delta.days)
    except Exception:
        return 0


def _parse_horizon_days(horizon: str) -> int:
    """Convert horizon string to days."""
    horizon = horizon.lower()
    if horizon == "1w":
        return 7
    elif horizon in ("1m", "4w"):
        return 28
    elif horizon == "2w":
        return 14
    else:
        return 7  # default


def _load_all_positions() -> List[Dict[str, Any]]:
    """
    Load all current positions from all bot state files.
    
    Returns:
        List of position dicts with ticker, entry details, etc.
    """
    positions = []
    
    # Ensure bot state directory exists
    if not BOT_STATE_DIR.exists():
        return positions
    
    # Find all bot state files (both .gz and .json)
    state_files = list(BOT_STATE_DIR.glob("rolling_*.json.gz")) + list(BOT_STATE_DIR.glob("rolling_*.json"))
    
    for state_file in state_files:
        state = _load_bot_state(state_file)
        if not isinstance(state, dict):
            continue
        
        # Extract positions from state
        raw_positions = state.get("positions") or {}
        if not isinstance(raw_positions, dict):
            continue
        
        # Get bot metadata
        bot_key = state_file.stem.replace("rolling_", "").replace(".json", "")
        
        for symbol, pos_data in raw_positions.items():
            if not isinstance(pos_data, dict):
                continue
            
            symbol_upper = str(symbol).upper()
            
            # Extract position details
            entry_price = _safe_float(pos_data.get("entry") or 0.0, 0.0)
            quantity = _safe_float(pos_data.get("qty") or 0.0, 0.0)
            
            if quantity <= 0:
                continue
            
            # Try to get entry date (may not always be present)
            entry_date = pos_data.get("entry_date") or pos_data.get("date") or ""
            
            positions.append({
                "ticker": symbol_upper,
                "bot_key": bot_key,
                "quantity": quantity,
                "avg_entry_price": entry_price,
                "entry_date": entry_date,
            })
    
    return positions


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@router.get("/holdings/top/{horizon}")
def get_top_holdings_by_pnl(horizon: str, limit: int = 3) -> Dict[str, Any]:
    """
    Returns top N holdings ranked by PnL for a given time horizon.
    
    Args:
        horizon: "1w" (7 days) or "1m"/"4w" (28 days)
        limit: Number of top holdings to return (default 3)
    
    Returns:
        {
            "horizon": "1w",
            "count": 3,
            "holdings": [
                {
                    "ticker": "AAPL",
                    "current_price": 185.50,
                    "avg_entry_price": 175.00,
                    "pnl_dollars": 105.00,
                    "pnl_percent": 6.0,
                    "quantity": 10,
                    "days_held": 8,
                    "entry_date": "2026-01-08"
                }
            ]
        }
    """
    try:
        # Parse horizon to days
        min_days = _parse_horizon_days(horizon)
        
        # Load all positions from bot states
        all_positions = _load_all_positions()
        
        # Load current prices
        current_prices = _load_prices_from_rolling()
        
        # Calculate PnL for each position
        holdings_with_pnl = []
        
        for pos in all_positions:
            ticker = pos["ticker"]
            entry_price = pos["avg_entry_price"]
            quantity = pos["quantity"]
            entry_date = pos["entry_date"]
            
            # Get current price
            current_price = current_prices.get(ticker)
            if current_price is None or current_price <= 0:
                continue
            
            # Calculate days held
            days_held = _calculate_days_held(entry_date) if entry_date else 0
            
            # Filter by horizon - only include positions held >= min_days
            if days_held < min_days:
                continue
            
            # Calculate PnL
            pnl_dollars = (current_price - entry_price) * quantity
            pnl_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            
            holdings_with_pnl.append({
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "avg_entry_price": round(entry_price, 2),
                "pnl_dollars": round(pnl_dollars, 2),
                "pnl_percent": round(pnl_percent, 2),
                "quantity": quantity,
                "days_held": days_held,
                "entry_date": entry_date,
            })
        
        # Sort by pnl_percent descending and take top N
        holdings_with_pnl.sort(key=lambda x: x["pnl_percent"], reverse=True)
        top_holdings = holdings_with_pnl[:limit]
        
        return {
            "horizon": horizon,
            "count": len(top_holdings),
            "holdings": top_holdings,
        }
        
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-2000:],
            },
        )

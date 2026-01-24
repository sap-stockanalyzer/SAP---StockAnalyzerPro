"""P&L attribution and dashboard endpoints (PR #4).

Provides comprehensive P&L tracking with attribution by strategy
and feature importance.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
from pathlib import Path

from dt_backend.core.logger_dt import log

router = APIRouter(prefix="/api/pnl", tags=["P&L Dashboard"])


@router.get("/dashboard")
async def get_pnl_dashboard() -> Dict[str, Any]:
    """
    Get comprehensive P&L dashboard with attribution by strategy and feature.
    
    Returns:
    {
        "daily_pnl": 150.25,
        "weekly_pnl": 650.50,
        "monthly_pnl": 2100.00,
        "ytd_pnl": 12500.00,
        "win_rate": 0.58,
        "total_trades": 50,
        "avg_trade_size": 125.50,
        "max_drawdown_pct": 5.2,
        "sharpe_ratio": 1.45,
        "pnl_by_strategy": {
            "VWAP_MR": {"pnl": 450.00, "trades": 12, "win_rate": 0.65},
            "ORB": {"pnl": 200.00, "trades": 8, "win_rate": 0.50},
        },
        "pnl_by_feature": {
            "momentum": {"pnl": 300.00, "contribution": 0.32},
            "volatility": {"pnl": 150.00, "contribution": 0.16},
            "volume": {"pnl": 75.00, "contribution": 0.08},
        },
        "daily_history": [
            {"date": "2026-01-24", "pnl": 150.25, "trades": 5},
            {"date": "2026-01-23", "pnl": 200.00, "trades": 6},
        ],
    }
    """
    try:
        # Read positions and trades
        try:
            from dt_backend.services.position_manager_dt import read_positions_state
            pos_state = read_positions_state()
        except Exception:
            pos_state = {}
        
        # Calculate P&L
        total_pnl = _calculate_total_pnl(pos_state)
        daily_pnl = _calculate_daily_pnl()
        weekly_pnl = _calculate_weekly_pnl()
        monthly_pnl = _calculate_monthly_pnl()
        
        # Calculate win rate
        trades = _get_recent_trades(days=30)
        win_rate = _calculate_win_rate(trades)
        
        # P&L by strategy
        pnl_by_strategy = _calculate_pnl_by_strategy(trades)
        
        # P&L by feature
        pnl_by_feature = _calculate_pnl_by_feature(trades)
        
        # Daily history
        daily_history = _get_daily_pnl_history(days=30)
        
        # Calculate metrics
        sharpe = _calculate_sharpe_ratio(daily_history)
        max_dd = _calculate_max_drawdown(daily_history)
        
        return {
            "daily_pnl": daily_pnl,
            "weekly_pnl": weekly_pnl,
            "monthly_pnl": monthly_pnl,
            "ytd_pnl": total_pnl,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "avg_trade_size": _calculate_avg_trade_size(trades),
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio": sharpe,
            "pnl_by_strategy": pnl_by_strategy,
            "pnl_by_feature": pnl_by_feature,
            "daily_history": daily_history,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    except Exception as e:
        log(f"[pnl_dashboard] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_pnl_metrics() -> Dict[str, float]:
    """Get key P&L metrics for monitoring."""
    dashboard = await get_pnl_dashboard()
    return {
        "daily_pnl": dashboard["daily_pnl"],
        "win_rate": dashboard["win_rate"],
        "sharpe_ratio": dashboard["sharpe_ratio"],
        "max_drawdown_pct": dashboard["max_drawdown_pct"],
    }


@router.get("/attribution/{symbol}")
async def get_attribution_for_symbol(symbol: str) -> Dict[str, Any]:
    """Get P&L attribution for specific symbol."""
    try:
        trades = _get_trades_for_symbol(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "total_pnl": sum(t.get("pnl", 0) for t in trades),
            "trades": len(trades),
            "win_rate": _calculate_win_rate(trades),
            "avg_pnl_per_trade": _calculate_avg_trade_size(trades),
            "trades_detail": trades[-10:],  # Last 10 trades
        }
    except Exception as e:
        log(f"[pnl_attribution] ❌ Error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _calculate_total_pnl(pos_state: Dict) -> float:
    """Calculate total realized P&L from positions state."""
    total = 0.0
    for sym, pos in pos_state.items():
        if isinstance(pos, dict) and "realized_pnl" in pos:
            total += float(pos["realized_pnl"])
    return total


def _calculate_daily_pnl() -> float:
    """Calculate today's P&L."""
    trades = _get_recent_trades(days=1)
    return sum(t.get("pnl", 0) for t in trades)


def _calculate_weekly_pnl() -> float:
    """Calculate this week's P&L."""
    trades = _get_recent_trades(days=7)
    return sum(t.get("pnl", 0) for t in trades)


def _calculate_monthly_pnl() -> float:
    """Calculate this month's P&L."""
    trades = _get_recent_trades(days=30)
    return sum(t.get("pnl", 0) for t in trades)


def _get_recent_trades(days: int = 30) -> List[Dict]:
    """Get trades from last N days."""
    # Load from dt_trades.jsonl
    try:
        trades_file = Path("da_brains/intraday/dt_trades.jsonl")
        if not trades_file.exists():
            return []
        
        trades = []
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with open(trades_file, "r") as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    ts = trade.get("ts", "")
                    if ts:
                        # Handle various ISO formats
                        ts_clean = ts.replace("Z", "+00:00")
                        try:
                            trade_dt = datetime.fromisoformat(ts_clean)
                        except:
                            # Try parsing without timezone
                            trade_dt = datetime.fromisoformat(ts.split("+")[0].split("Z")[0])
                            trade_dt = trade_dt.replace(tzinfo=None)
                            cutoff_naive = cutoff.replace(tzinfo=None)
                            if trade_dt > cutoff_naive:
                                trades.append(trade)
                            continue
                        
                        if trade_dt > cutoff:
                            trades.append(trade)
                except Exception:
                    continue
        
        return trades
    except Exception as e:
        log(f"[trades] Error reading trades: {e}")
        return []


def _calculate_win_rate(trades: List[Dict]) -> float:
    """Calculate win rate from trades."""
    if not trades:
        return 0.0
    
    winning = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return winning / len(trades)


def _calculate_pnl_by_strategy(trades: List[Dict]) -> Dict[str, Dict]:
    """Group P&L by strategy."""
    by_strategy = {}
    for trade in trades:
        strat = trade.get("strategy", "unknown")
        if strat not in by_strategy:
            by_strategy[strat] = {"pnl": 0, "trades": 0, "wins": 0}
        
        by_strategy[strat]["pnl"] += trade.get("pnl", 0)
        by_strategy[strat]["trades"] += 1
        if trade.get("pnl", 0) > 0:
            by_strategy[strat]["wins"] += 1
    
    # Add win rate
    for strat in by_strategy:
        trades_count = by_strategy[strat]["trades"]
        by_strategy[strat]["win_rate"] = (
            by_strategy[strat]["wins"] / trades_count if trades_count > 0 else 0
        )
    
    return by_strategy


def _calculate_pnl_by_feature(trades: List[Dict]) -> Dict[str, Dict]:
    """Calculate P&L contribution by feature."""
    # Use feature importance to attribute P&L
    by_feature = {}
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    
    if total_pnl <= 0:
        return by_feature
    
    # Get feature importance for recent trades
    for trade in trades[-20:]:  # Last 20 trades
        features = trade.get("features", {})
        if not features:
            continue
            
        for feature, value in features.items():
            if feature not in by_feature:
                by_feature[feature] = {"pnl": 0, "contribution": 0}
            
            # Distribute trade P&L proportionally
            by_feature[feature]["pnl"] += trade.get("pnl", 0) / len(features)
    
    # Calculate contribution percentage
    for feature in by_feature:
        by_feature[feature]["contribution"] = (
            by_feature[feature]["pnl"] / total_pnl if total_pnl > 0 else 0
        )
    
    return by_feature


def _calculate_avg_trade_size(trades: List[Dict]) -> float:
    """Calculate average P&L per trade."""
    if not trades:
        return 0.0
    return sum(t.get("pnl", 0) for t in trades) / len(trades)


def _get_daily_pnl_history(days: int = 30) -> List[Dict]:
    """Get daily P&L history."""
    history = {}
    trades = _get_recent_trades(days=days)
    
    for trade in trades:
        ts = trade.get("ts", "")
        if ts:
            trade_date = ts.split("T")[0]
            if trade_date not in history:
                history[trade_date] = {"pnl": 0, "trades": 0}
            
            history[trade_date]["pnl"] += trade.get("pnl", 0)
            history[trade_date]["trades"] += 1
    
    # Convert to list
    return [
        {"date": date, **data}
        for date, data in sorted(history.items())
    ]


def _calculate_sharpe_ratio(daily_history: List[Dict], risk_free_rate: float = 0.05) -> float:
    """Calculate Sharpe ratio from daily P&L."""
    if len(daily_history) < 2:
        return 0.0
    
    pnls = [d["pnl"] for d in daily_history]
    
    # Calculate daily return
    mean_return = sum(pnls) / len(pnls)
    variance = sum((x - mean_return) ** 2 for x in pnls) / len(pnls)
    std_dev = variance ** 0.5
    
    if std_dev == 0:
        return 0.0
    
    # Annual Sharpe ratio
    daily_rf = risk_free_rate / 252
    sharpe = (mean_return - daily_rf) / std_dev * (252 ** 0.5)
    return sharpe


def _calculate_max_drawdown(daily_history: List[Dict]) -> float:
    """Calculate maximum drawdown."""
    if not daily_history:
        return 0.0
    
    cumulative_pnl = 0
    peak = 0
    max_dd = 0
    
    for day in daily_history:
        cumulative_pnl += day["pnl"]
        if cumulative_pnl > peak:
            peak = cumulative_pnl
        
        drawdown = peak - cumulative_pnl
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd / peak if peak > 0 else 0


def _get_trades_for_symbol(symbol: str) -> List[Dict]:
    """Get all trades for specific symbol."""
    all_trades = _get_recent_trades(days=90)
    return [t for t in all_trades if t.get("symbol", "").upper() == symbol]

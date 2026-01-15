"""Prometheus metrics exporter for dt_backend intraday trading."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter()

# In-memory metrics storage
_metrics: Dict[str, Any] = {
    "trades_total": 0,
    "trades_long": 0,
    "trades_short": 0,
    "cycle_duration_sum": 0.0,
    "cycle_count": 0,
    "open_positions": 0,
    "equity_dollars": 100000.0,
    "daily_pnl_dollars": 0.0,
    "errors_total": 0,
}


def update_metric(name: str, value: float, labels: Dict[str, str] = None):
    """Update a metric value."""
    if labels:
        key = f"{name}_{','.join(f'{k}={v}' for k, v in labels.items())}"
        _metrics[key] = value
    else:
        _metrics[name] = value


def increment_metric(name: str, delta: float = 1.0, labels: Dict[str, str] = None):
    """Increment a counter metric."""
    if labels:
        key = f"{name}_{','.join(f'{k}={v}' for k, v in labels.items())}"
        _metrics[key] = _metrics.get(key, 0) + delta
    else:
        _metrics[name] = _metrics.get(name, 0) + delta


def load_metrics_from_state():
    """Load current metrics from dt_state.json and dt_metrics.json."""
    try:
        truth_dir = os.getenv("DT_TRUTH_DIR", "da_brains")
        
        # Load dt_metrics.json for equity and positions
        metrics_file = Path(truth_dir) / "intraday" / "dt_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                _metrics["equity_dollars"] = float(metrics.get("equity", 100000.0))
                _metrics["open_positions"] = int(metrics.get("open_positions", 0))
                _metrics["daily_pnl_dollars"] = float(metrics.get("realized_pnl_today", 0.0))
        
        # Load dt_trades.jsonl for trade counts
        trades_file = Path(truth_dir) / "intraday" / "dt_trades.jsonl"
        if trades_file.exists():
            trades_total = 0
            trades_long = 0
            trades_short = 0
            
            with open(trades_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get("type") == "entry":
                            trades_total += 1
                            side = event.get("side", "").lower()
                            if side == "buy" or side == "long":
                                trades_long += 1
                            elif side == "sell" or side == "short":
                                trades_short += 1
                    except Exception:
                        continue
            
            _metrics["trades_total"] = trades_total
            _metrics["trades_long"] = trades_long
            _metrics["trades_short"] = trades_short
    except Exception:
        pass


def generate_prometheus_metrics() -> str:
    """Generate Prometheus-compatible metrics output."""
    # Load latest metrics from state files
    load_metrics_from_state()
    
    lines = []
    
    # Trade counter
    lines.append("# HELP dt_trades_total Total trades executed")
    lines.append("# TYPE dt_trades_total counter")
    lines.append(f'dt_trades_total{{side="long",bot="DT"}} {_metrics["trades_long"]}')
    lines.append(f'dt_trades_total{{side="short",bot="DT"}} {_metrics["trades_short"]}')
    
    # Cycle duration histogram (simplified as gauge for now)
    lines.append("# HELP dt_cycle_duration_seconds Cycle execution time")
    lines.append("# TYPE dt_cycle_duration_seconds gauge")
    if _metrics["cycle_count"] > 0:
        avg_duration = _metrics["cycle_duration_sum"] / _metrics["cycle_count"]
        lines.append(f"dt_cycle_duration_seconds {avg_duration:.3f}")
    else:
        lines.append("dt_cycle_duration_seconds 0.0")
    
    # Open positions gauge
    lines.append("# HELP dt_open_positions Number of open positions")
    lines.append("# TYPE dt_open_positions gauge")
    lines.append(f'dt_open_positions {_metrics["open_positions"]}')
    
    # Equity gauge
    lines.append("# HELP dt_equity_dollars Current equity")
    lines.append("# TYPE dt_equity_dollars gauge")
    lines.append(f'dt_equity_dollars {_metrics["equity_dollars"]:.2f}')
    
    # Daily P&L gauge
    lines.append("# HELP dt_daily_pnl_dollars Daily PnL")
    lines.append("# TYPE dt_daily_pnl_dollars gauge")
    lines.append(f'dt_daily_pnl_dollars {_metrics["daily_pnl_dollars"]:.2f}')
    
    # Error counter
    lines.append("# HELP dt_errors_total Total errors")
    lines.append("# TYPE dt_errors_total counter")
    lines.append(f'dt_errors_total {_metrics["errors_total"]}')
    
    return "\n".join(lines) + "\n"


@router.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    metrics_text = generate_prometheus_metrics()
    return PlainTextResponse(content=metrics_text, media_type="text/plain; charset=utf-8")


@router.post("/metrics/cycle")
def record_cycle_duration(duration_seconds: float):
    """Record a cycle duration (called by daytrading_job)."""
    _metrics["cycle_duration_sum"] += duration_seconds
    _metrics["cycle_count"] += 1
    return {"status": "recorded", "duration_seconds": duration_seconds}


@router.post("/metrics/error")
def record_error(error_type: str = "unknown"):
    """Record an error (called by error handlers)."""
    increment_metric("errors_total")
    return {"status": "recorded", "error_type": error_type}


@router.get("/metrics/summary")
def metrics_summary():
    """JSON summary of current metrics (for debugging)."""
    load_metrics_from_state()
    return {
        "trades": {
            "total": _metrics["trades_total"],
            "long": _metrics["trades_long"],
            "short": _metrics["trades_short"],
        },
        "positions": {
            "open": _metrics["open_positions"],
        },
        "equity": {
            "current": _metrics["equity_dollars"],
            "daily_pnl": _metrics["daily_pnl_dollars"],
        },
        "performance": {
            "cycle_count": _metrics["cycle_count"],
            "avg_cycle_duration": (
                _metrics["cycle_duration_sum"] / _metrics["cycle_count"]
                if _metrics["cycle_count"] > 0 else 0.0
            ),
        },
        "errors": {
            "total": _metrics["errors_total"],
        },
    }

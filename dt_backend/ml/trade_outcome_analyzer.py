"""
dt_backend/ml/trade_outcome_analyzer.py

Analyze closed trades and maintain performance metrics.

Tracks:
- Win rate per symbol/sector/regime/time-of-day
- Average win/loss sizes
- Hold durations
- Model confidence vs actual outcomes
- Profit factor, Sharpe ratio
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dt_backend.core.config_dt import DT_PATHS
    from dt_backend.core.logger_dt import log
except Exception:
    DT_PATHS = {}
    def log(msg: str) -> None:
        print(msg, flush=True)


@dataclass
class TradeOutcome:
    """Structured trade outcome for analysis."""
    symbol: str
    entry_time: str  # ISO format
    exit_time: str   # ISO format
    side: str  # "LONG" | "SHORT" | "BUY" | "SELL"
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    hold_duration_minutes: int
    
    # ML signal at entry
    model_label: str
    model_confidence: float
    lgb_prob: float
    lstm_prob: Optional[float] = None
    transformer_prob: Optional[float] = None
    
    # Context
    regime: str = "unknown"
    vix: Optional[float] = None
    sector: Optional[str] = None
    time_of_day: str = "unknown"  # "open" | "mid_morning" | "lunch" | "afternoon" | "close"
    
    # Outcome
    success: bool = False
    quality: str = "mediocre"  # "great" | "good" | "mediocre" | "bad"
    exit_reason: str = "unknown"


class TradeOutcomeAnalyzer:
    """Analyze trade outcomes and maintain performance windows."""
    
    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            # Use learning path from DT_PATHS
            learning_path = DT_PATHS.get("learning")
            if learning_path:
                self.data_path = Path(learning_path)
            else:
                # Fallback
                da_brains = DT_PATHS.get("da_brains", Path("da_brains"))
                self.data_path = Path(da_brains) / "dt_learning"
        else:
            self.data_path = Path(data_path)
        
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.trades_file = self.data_path / "trade_outcomes.jsonl.gz"
        self.metrics_file = self.data_path / "performance_metrics.json"
        self.baseline_file = self.data_path / "baseline_performance.json"
        
    def process_closed_trade(self, trade_dict: Dict[str, Any]) -> Optional[TradeOutcome]:
        """Process a closed trade and extract learning signals.
        
        Expected trade_dict format from dt_trades.jsonl:
        {
            "type": "fill" | "exit" | ...,
            "symbol": "AAPL",
            "side": "BUY" | "SELL",
            "price": 180.50,
            "qty": 10,
            "confidence": 0.75,
            "timestamp": "2026-01-16T14:30:00Z",
            ...
        }
        """
        try:
            # Extract basic trade info
            symbol = trade_dict.get("symbol", "")
            if not symbol:
                return None
            
            # Determine if this is an exit event
            trade_type = trade_dict.get("type", "")
            if trade_type not in ["exit", "fill_exit", "close"]:
                # Not a closed trade
                return None
            
            side = trade_dict.get("side", "BUY")
            exit_price = float(trade_dict.get("price", 0.0))
            exit_time_str = trade_dict.get("timestamp", datetime.now(timezone.utc).isoformat())
            
            # Get entry info (would need to be tracked separately)
            # For now, use simplified extraction
            entry_price = float(trade_dict.get("entry_price", exit_price))
            entry_time_str = trade_dict.get("entry_timestamp", exit_time_str)
            
            # Calculate PnL
            if side.upper() in ["BUY", "LONG"]:
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
            else:  # SHORT/SELL
                pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0
            
            qty = float(trade_dict.get("qty", 1.0))
            pnl = pnl_pct * entry_price * qty
            
            # Calculate hold duration
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
                hold_duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
            except Exception:
                hold_duration_minutes = 0
            
            # Extract ML signals
            confidence = float(trade_dict.get("confidence", 0.5))
            model_label = trade_dict.get("label", "HOLD")
            lgb_prob = float(trade_dict.get("lgb_prob", confidence))
            lstm_prob = trade_dict.get("lstm_prob")
            transformer_prob = trade_dict.get("transformer_prob")
            
            # Context
            regime = trade_dict.get("regime", "unknown")
            vix = trade_dict.get("vix")
            sector = trade_dict.get("sector")
            
            # Determine time of day
            time_of_day = self._classify_time_of_day(entry_time_str)
            
            # Classify quality
            quality = self._classify_quality(pnl_pct)
            success = pnl_pct > 0
            
            exit_reason = trade_dict.get("exit_reason", trade_dict.get("reason", "unknown"))
            
            outcome = TradeOutcome(
                symbol=symbol,
                entry_time=entry_time_str,
                exit_time=exit_time_str,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                hold_duration_minutes=hold_duration_minutes,
                model_label=model_label,
                model_confidence=confidence,
                lgb_prob=lgb_prob,
                lstm_prob=lstm_prob,
                transformer_prob=transformer_prob,
                regime=regime,
                vix=vix,
                sector=sector,
                time_of_day=time_of_day,
                success=success,
                quality=quality,
                exit_reason=exit_reason,
            )
            
            # Append to trade outcomes file
            self._append_outcome(outcome)
            
            return outcome
            
        except Exception as e:
            log(f"[trade_analyzer] ⚠️ Error processing trade: {e}")
            return None
    
    def _classify_time_of_day(self, timestamp_str: str) -> str:
        """Classify time of day for a timestamp."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            # Convert to NY time
            try:
                from zoneinfo import ZoneInfo
                ny_tz = ZoneInfo("America/New_York")
                dt_ny = dt.astimezone(ny_tz)
            except Exception:
                dt_ny = dt
            
            hour = dt_ny.hour
            minute = dt_ny.minute
            time_min = hour * 60 + minute
            
            # Market hours: 9:30 - 16:00 ET
            open_time = 9 * 60 + 30      # 9:30
            mid_morning = 10 * 60 + 30   # 10:30
            lunch = 12 * 60              # 12:00
            afternoon = 14 * 60          # 14:00
            close = 15 * 60 + 45         # 15:45
            
            if time_min < open_time + 30:
                return "open"
            elif time_min < mid_morning:
                return "mid_morning"
            elif time_min < afternoon:
                return "lunch"
            elif time_min < close:
                return "afternoon"
            else:
                return "close"
        except Exception:
            return "unknown"
    
    def _classify_quality(self, pnl_pct: float) -> str:
        """Classify trade quality based on PnL percentage."""
        if pnl_pct >= 0.03:
            return "great"
        elif pnl_pct >= 0.01:
            return "good"
        elif pnl_pct >= -0.01:
            return "mediocre"
        else:
            return "bad"
    
    def _append_outcome(self, outcome: TradeOutcome) -> None:
        """Append outcome to trade outcomes file."""
        try:
            # Convert to dict
            outcome_dict = asdict(outcome)
            outcome_dict["_ts"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            # Append to gzipped jsonl
            with gzip.open(self.trades_file, "at", encoding="utf-8") as f:
                f.write(json.dumps(outcome_dict, ensure_ascii=False) + "\n")
        except Exception as e:
            log(f"[trade_analyzer] ⚠️ Error appending outcome: {e}")
    
    def update_metrics(self, outcome: TradeOutcome) -> None:
        """Update rolling performance metrics."""
        try:
            # Load existing metrics
            metrics = self._load_metrics()
            
            # Update global metrics
            metrics.setdefault("global", {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "win_rate": 0.0,
            })
            
            global_stats = metrics["global"]
            global_stats["total_trades"] += 1
            global_stats["total_pnl"] += outcome.pnl
            global_stats["total_pnl_pct"] += outcome.pnl_pct
            
            if outcome.success:
                global_stats["wins"] += 1
            else:
                global_stats["losses"] += 1
            
            # Calculate win rate
            if global_stats["total_trades"] > 0:
                global_stats["win_rate"] = global_stats["wins"] / global_stats["total_trades"]
            
            # Update per-symbol stats
            metrics.setdefault("by_symbol", {})
            sym_stats = metrics["by_symbol"].setdefault(outcome.symbol, {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
            })
            sym_stats["trades"] += 1
            sym_stats["pnl"] += outcome.pnl
            if outcome.success:
                sym_stats["wins"] += 1
            
            # Update per-regime stats
            metrics.setdefault("by_regime", {})
            regime_stats = metrics["by_regime"].setdefault(outcome.regime, {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
            })
            regime_stats["trades"] += 1
            regime_stats["pnl"] += outcome.pnl
            if outcome.success:
                regime_stats["wins"] += 1
            
            # Update per-time-of-day stats
            metrics.setdefault("by_time_of_day", {})
            tod_stats = metrics["by_time_of_day"].setdefault(outcome.time_of_day, {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
            })
            tod_stats["trades"] += 1
            tod_stats["pnl"] += outcome.pnl
            if outcome.success:
                tod_stats["wins"] += 1
            
            # Update confidence calibration buckets
            metrics.setdefault("confidence_buckets", {})
            conf_bucket = f"{int(outcome.model_confidence * 10) * 10}"
            conf_stats = metrics["confidence_buckets"].setdefault(conf_bucket, {
                "trades": 0,
                "wins": 0,
                "avg_confidence": 0.0,
                "sum_confidence": 0.0,
            })
            conf_stats["trades"] += 1
            conf_stats["sum_confidence"] += outcome.model_confidence
            conf_stats["avg_confidence"] = conf_stats["sum_confidence"] / conf_stats["trades"]
            if outcome.success:
                conf_stats["wins"] += 1
            
            # Save updated metrics
            self._save_metrics(metrics)
            
        except Exception as e:
            log(f"[trade_analyzer] ⚠️ Error updating metrics: {e}")
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load metrics from file."""
        if not self.metrics_file.exists():
            return {}
        try:
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to file."""
        try:
            metrics["_updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"[trade_analyzer] ⚠️ Error saving metrics: {e}")
    
    def get_performance_window(self, days: int = 7, trades: Optional[int] = None) -> Dict[str, Any]:
        """Get performance metrics for recent window."""
        try:
            # Read recent trades from outcomes file
            recent_trades = self._read_recent_trades(days=days, max_trades=trades)
            
            if not recent_trades:
                return {
                    "win_rate": 0.0,
                    "accuracy": 0.0,
                    "total_trades": 0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0,
                }
            
            wins = [t for t in recent_trades if t["success"]]
            losses = [t for t in recent_trades if not t["success"]]
            
            win_rate = len(wins) / len(recent_trades) if recent_trades else 0.0
            
            avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0.0
            avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0.0
            
            total_win_pnl = sum(t["pnl_pct"] for t in wins)
            total_loss_pnl = abs(sum(t["pnl_pct"] for t in losses))
            profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 0.0
            
            # Simple Sharpe approximation
            pnl_pcts = [t["pnl_pct"] for t in recent_trades]
            avg_return = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0.0
            std_return = (sum((r - avg_return) ** 2 for r in pnl_pcts) / len(pnl_pcts)) ** 0.5 if len(pnl_pcts) > 1 else 1.0
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0.0
            
            return {
                "win_rate": win_rate,
                "accuracy": win_rate,
                "total_trades": len(recent_trades),
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "avg_confidence": sum(t["model_confidence"] for t in recent_trades) / len(recent_trades),
                "consecutive_wins": self._count_consecutive_wins(recent_trades),
                "consecutive_losses": self._count_consecutive_losses(recent_trades),
                "avg_win_hold_time": sum(t["hold_duration_minutes"] for t in wins) / len(wins) if wins else 0,
                "drawdown_pct": self._calculate_drawdown(recent_trades),
            }
        except Exception as e:
            log(f"[trade_analyzer] ⚠️ Error getting performance window: {e}")
            return {}
    
    def _read_recent_trades(self, days: int = 7, max_trades: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read recent trades from outcomes file."""
        if not self.trades_file.exists():
            return []
        
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            trades = []
            
            with gzip.open(self.trades_file, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    trade = json.loads(line)
                    
                    # Check if within window
                    exit_time_str = trade.get("exit_time", "")
                    try:
                        exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
                        if exit_time >= cutoff:
                            trades.append(trade)
                    except Exception:
                        continue
            
            # Sort by exit time (most recent last)
            trades.sort(key=lambda t: t.get("exit_time", ""))
            
            # Limit to max_trades if specified
            if max_trades and len(trades) > max_trades:
                trades = trades[-max_trades:]
            
            return trades
        except Exception as e:
            log(f"[trade_analyzer] ⚠️ Error reading recent trades: {e}")
            return []
    
    def _count_consecutive_wins(self, trades: List[Dict[str, Any]]) -> int:
        """Count consecutive wins at the end of the trades list."""
        count = 0
        for trade in reversed(trades):
            if trade.get("success", False):
                count += 1
            else:
                break
        return count
    
    def _count_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Count consecutive losses at the end of the trades list."""
        count = 0
        for trade in reversed(trades):
            if not trade.get("success", False):
                count += 1
            else:
                break
        return count
    
    def _calculate_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown percentage."""
        if not trades:
            return 0.0
        
        cumulative_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        
        for trade in trades:
            cumulative_pnl += trade.get("pnl_pct", 0.0)
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def get_baseline_performance(self) -> Dict[str, Any]:
        """Get baseline performance from last retrain."""
        if not self.baseline_file.exists():
            return {}
        
        try:
            with open(self.baseline_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def save_baseline_performance(self, performance: Dict[str, Any]) -> None:
        """Save baseline performance after retrain."""
        try:
            performance["_saved_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            with open(self.baseline_file, "w", encoding="utf-8") as f:
                json.dump(performance, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"[trade_analyzer] ⚠️ Error saving baseline: {e}")


def analyze_trade_outcome(trade: Dict[str, Any]) -> None:
    """Main entry point to process a closed trade."""
    try:
        analyzer = TradeOutcomeAnalyzer()
        outcome = analyzer.process_closed_trade(trade)
        if outcome:
            analyzer.update_metrics(outcome)
            log(f"[trade_analyzer] ✅ Processed {outcome.symbol} {outcome.side} PnL={outcome.pnl_pct:.2%}")
    except Exception as e:
        log(f"[trade_analyzer] ⚠️ Error in analyze_trade_outcome: {e}")


def analyze_all_trades_today() -> Dict[str, Any]:
    """Analyze all closed trades from today."""
    try:
        analyzer = TradeOutcomeAnalyzer()
        
        # This would read from dt_trades.jsonl for today
        # For now, return summary
        metrics = analyzer._load_metrics()
        
        return {
            "status": "success",
            "trades_analyzed": metrics.get("global", {}).get("total_trades", 0),
            "win_rate": metrics.get("global", {}).get("win_rate", 0.0),
        }
    except Exception as e:
        log(f"[trade_analyzer] ⚠️ Error analyzing trades today: {e}")
        return {"status": "error", "error": str(e)}

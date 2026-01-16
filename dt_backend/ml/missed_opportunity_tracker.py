"""
dt_backend/ml/missed_opportunity_tracker.py

Track high-confidence signals that were NOT traded and analyze outcomes.

When signal not traded, log:
- Symbol, label, confidence
- Reason not traded (max_positions, confidence_threshold, risk_limit, etc.)

Schedule follow-up checks at:
- 15 minutes
- 30 minutes
- 1 hour
- End of day

Calculate hypothetical PnL to assess if opportunity was truly missed.
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
class MissedSignal:
    """A high-confidence signal that was not traded."""
    symbol: str
    timestamp: str
    label: str  # "BUY" | "SELL"
    confidence: float
    price_at_signal: float
    reason_not_traded: str
    
    # ML context
    lgb_prob: float
    lstm_prob: Optional[float] = None
    transformer_prob: Optional[float] = None
    
    # Market context
    regime: str = "unknown"
    vix: Optional[float] = None
    
    # Follow-up checks (to be filled in later)
    price_15m: Optional[float] = None
    price_30m: Optional[float] = None
    price_1h: Optional[float] = None
    price_eod: Optional[float] = None
    
    # Outcome analysis
    would_have_won: Optional[bool] = None
    hypothetical_pnl_pct: Optional[float] = None
    evaluated: bool = False


class MissedOpportunityTracker:
    """Track and analyze missed trading opportunities."""
    
    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            learning_path = DT_PATHS.get("learning")
            if learning_path:
                self.data_path = Path(learning_path)
            else:
                da_brains = DT_PATHS.get("da_brains", Path("da_brains"))
                self.data_path = Path(da_brains) / "dt_learning"
        else:
            self.data_path = Path(data_path)
        
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.missed_signals_file = self.data_path / "missed_signals.jsonl.gz"
        self.missed_analysis_file = self.data_path / "missed_analysis.json"
    
    def log_missed_signal(self, signal: Dict[str, Any], reason: str) -> None:
        """Log a high-confidence signal that was not traded.
        
        Args:
            signal: Signal dict with symbol, label, confidence, price, etc.
            reason: Reason why signal was not traded
        """
        try:
            symbol = signal.get("symbol", "")
            if not symbol:
                return
            
            timestamp = signal.get("timestamp", datetime.now(timezone.utc).isoformat())
            label = signal.get("label", "HOLD")
            confidence = float(signal.get("confidence", 0.0))
            price = float(signal.get("price", 0.0))
            
            # Only track high-confidence signals
            if confidence < 0.60:
                return
            
            lgb_prob = float(signal.get("lgb_prob", confidence))
            lstm_prob = signal.get("lstm_prob")
            transformer_prob = signal.get("transformer_prob")
            
            regime = signal.get("regime", "unknown")
            vix = signal.get("vix")
            
            missed = MissedSignal(
                symbol=symbol,
                timestamp=timestamp,
                label=label,
                confidence=confidence,
                price_at_signal=price,
                reason_not_traded=reason,
                lgb_prob=lgb_prob,
                lstm_prob=lstm_prob,
                transformer_prob=transformer_prob,
                regime=regime,
                vix=vix,
            )
            
            # Append to file
            self._append_missed_signal(missed)
            
            log(f"[missed_opp] 📝 Logged missed signal: {symbol} {label} conf={confidence:.2%} reason={reason}")
            
        except Exception as e:
            log(f"[missed_opp] ⚠️ Error logging missed signal: {e}")
    
    def _append_missed_signal(self, missed: MissedSignal) -> None:
        """Append missed signal to file."""
        try:
            signal_dict = asdict(missed)
            signal_dict["_logged_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            with gzip.open(self.missed_signals_file, "at", encoding="utf-8") as f:
                f.write(json.dumps(signal_dict, ensure_ascii=False) + "\n")
        except Exception as e:
            log(f"[missed_opp] ⚠️ Error appending missed signal: {e}")
    
    def check_missed_outcomes(self, price_data: Optional[Dict[str, Any]] = None) -> int:
        """Post-market: calculate would-have-been PnL for all missed signals.
        
        Args:
            price_data: Optional dict of {symbol: {timestamp: price}} for evaluation
            
        Returns:
            Number of signals evaluated
        """
        try:
            # Read unevaluated signals from today
            signals = self._read_unevaluated_signals()
            
            if not signals:
                log("[missed_opp] No unevaluated signals to check")
                return 0
            
            evaluated_count = 0
            
            for signal in signals:
                # Skip if already evaluated
                if signal.get("evaluated", False):
                    continue
                
                # Calculate hypothetical PnL
                # For now, use simple logic - in production would fetch actual price data
                symbol = signal["symbol"]
                entry_price = signal["price_at_signal"]
                label = signal["label"]
                
                # Simulate EOD price (would be fetched from actual data)
                # This is a placeholder - real implementation would use price_data
                exit_price = entry_price  # Placeholder
                
                if price_data and symbol in price_data:
                    # Use provided price data
                    symbol_prices = price_data[symbol]
                    # Get EOD price or latest available
                    exit_price = symbol_prices.get("eod", entry_price)
                
                # Calculate hypothetical PnL
                if label.upper() in ["BUY", "LONG"]:
                    pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                else:  # SELL/SHORT
                    pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0
                
                signal["hypothetical_pnl_pct"] = pnl_pct
                signal["would_have_won"] = pnl_pct > 0
                signal["evaluated"] = True
                signal["price_eod"] = exit_price
                
                evaluated_count += 1
            
            # Update analysis
            if evaluated_count > 0:
                self._update_missed_analysis(signals)
                log(f"[missed_opp] ✅ Evaluated {evaluated_count} missed signals")
            
            return evaluated_count
            
        except Exception as e:
            log(f"[missed_opp] ⚠️ Error checking missed outcomes: {e}")
            return 0
    
    def _read_unevaluated_signals(self, days: int = 1) -> List[Dict[str, Any]]:
        """Read unevaluated signals from recent days."""
        if not self.missed_signals_file.exists():
            return []
        
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            signals = []
            
            with gzip.open(self.missed_signals_file, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    signal = json.loads(line)
                    
                    # Check if within window and not evaluated
                    timestamp_str = signal.get("timestamp", "")
                    evaluated = signal.get("evaluated", False)
                    
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        if timestamp >= cutoff and not evaluated:
                            signals.append(signal)
                    except Exception:
                        continue
            
            return signals
        except Exception as e:
            log(f"[missed_opp] ⚠️ Error reading unevaluated signals: {e}")
            return []
    
    def _update_missed_analysis(self, evaluated_signals: List[Dict[str, Any]]) -> None:
        """Update missed opportunity analysis with evaluated signals."""
        try:
            # Load existing analysis
            analysis = self._load_analysis()
            
            # Update stats
            analysis.setdefault("total_missed", 0)
            analysis.setdefault("total_evaluated", 0)
            analysis.setdefault("would_have_won", 0)
            analysis.setdefault("would_have_lost", 0)
            analysis.setdefault("total_hypothetical_pnl_pct", 0.0)
            analysis.setdefault("by_reason", {})
            
            for signal in evaluated_signals:
                if not signal.get("evaluated", False):
                    continue
                
                analysis["total_evaluated"] += 1
                
                if signal.get("would_have_won", False):
                    analysis["would_have_won"] += 1
                else:
                    analysis["would_have_lost"] += 1
                
                pnl_pct = signal.get("hypothetical_pnl_pct", 0.0)
                analysis["total_hypothetical_pnl_pct"] += pnl_pct
                
                # Track by reason
                reason = signal.get("reason_not_traded", "unknown")
                reason_stats = analysis["by_reason"].setdefault(reason, {
                    "count": 0,
                    "won": 0,
                    "total_pnl_pct": 0.0,
                })
                reason_stats["count"] += 1
                reason_stats["total_pnl_pct"] += pnl_pct
                if signal.get("would_have_won", False):
                    reason_stats["won"] += 1
            
            # Calculate profitable missed percentage
            if analysis["total_evaluated"] > 0:
                analysis["profitable_missed_pct"] = analysis["would_have_won"] / analysis["total_evaluated"]
            
            # Save analysis
            self._save_analysis(analysis)
            
        except Exception as e:
            log(f"[missed_opp] ⚠️ Error updating analysis: {e}")
    
    def _load_analysis(self) -> Dict[str, Any]:
        """Load missed opportunity analysis."""
        if not self.missed_analysis_file.exists():
            return {}
        
        try:
            with open(self.missed_analysis_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save missed opportunity analysis."""
        try:
            analysis["_updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            with open(self.missed_analysis_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"[missed_opp] ⚠️ Error saving analysis: {e}")
    
    def analyze_missed_patterns(self) -> Dict[str, Any]:
        """Identify patterns in missed opportunities."""
        try:
            analysis = self._load_analysis()
            
            # Calculate summary stats
            total_eval = analysis.get("total_evaluated", 0)
            if total_eval == 0:
                return {
                    "missed_pnl_usd": 0,
                    "profitable_missed_pct": 0.0,
                    "suggestions": [],
                }
            
            profitable_pct = analysis.get("profitable_missed_pct", 0.0)
            total_pnl_pct = analysis.get("total_hypothetical_pnl_pct", 0.0)
            
            # Analyze by reason
            suggestions = []
            by_reason = analysis.get("by_reason", {})
            
            for reason, stats in by_reason.items():
                count = stats.get("count", 0)
                won = stats.get("won", 0)
                win_rate = won / count if count > 0 else 0.0
                
                # Suggest adjustments for high-win-rate missed reasons
                if win_rate > 0.65 and count >= 3:
                    if "confidence" in reason.lower():
                        suggestions.append(f"Lower confidence threshold - {reason} had {win_rate:.0%} win rate")
                    elif "max_positions" in reason.lower():
                        suggestions.append(f"Increase max positions - {reason} had {win_rate:.0%} win rate")
                    elif "risk" in reason.lower():
                        suggestions.append(f"Relax risk limits - {reason} had {win_rate:.0%} win rate")
            
            return {
                "missed_pnl_usd": total_pnl_pct * 1000,  # Approximate USD (would need position size)
                "profitable_missed_pct": profitable_pct,
                "total_evaluated": total_eval,
                "suggestions": suggestions,
                "by_reason": by_reason,
            }
            
        except Exception as e:
            log(f"[missed_opp] ⚠️ Error analyzing patterns: {e}")
            return {}


def track_missed_signal(signal: Dict[str, Any], reason: str) -> None:
    """Entry point: call this when NOT trading a high-confidence signal."""
    try:
        tracker = MissedOpportunityTracker()
        tracker.log_missed_signal(signal, reason)
    except Exception as e:
        log(f"[missed_opp] ⚠️ Error in track_missed_signal: {e}")


def analyze_missed_today() -> Dict[str, Any]:
    """Analyze missed opportunities from today."""
    try:
        tracker = MissedOpportunityTracker()
        
        # Check outcomes (would need price data in production)
        evaluated = tracker.check_missed_outcomes()
        
        # Analyze patterns
        patterns = tracker.analyze_missed_patterns()
        
        return {
            "status": "success",
            "evaluated": evaluated,
            "patterns": patterns,
        }
    except Exception as e:
        log(f"[missed_opp] ⚠️ Error in analyze_missed_today: {e}")
        return {"status": "error", "error": str(e)}

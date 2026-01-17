"""dt_backend/services/decision_recorder.py

Records all intraday trading decisions for replay and analysis.

Every decision during a trading cycle (symbol selection, entry, exit, etc.)
is recorded to a JSONL file for later replay with modified parameters.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Dict, List, Any, Optional
import uuid


class DecisionRecorder:
    """Records all intraday trading decisions for replay/analysis."""
    
    def __init__(self):
        from dt_backend.core.config_dt import DT_PATHS
        ml_data_dt = DT_PATHS.get("ml_data_dt", Path("ml_data_dt"))
        if not isinstance(ml_data_dt, Path):
            ml_data_dt = Path(ml_data_dt)
        ml_data_dt.mkdir(parents=True, exist_ok=True)
        self.decisions_file = ml_data_dt / "dt_decisions.jsonl"
        self.current_cycle_id = uuid.uuid4().hex[:12]
    
    def start_cycle(self, cycle_id: Optional[str] = None) -> str:
        """Start recording a new cycle.
        
        Args:
            cycle_id: Optional cycle ID. If not provided, generates a new one.
            
        Returns:
            The cycle ID for this recording session.
        """
        self.current_cycle_id = cycle_id or uuid.uuid4().hex[:12]
        return self.current_cycle_id
    
    def record_decision(
        self,
        phase: str,  # "symbol_selection", "entry", "exit", "rebalance"
        action: str,  # "selected_symbols", "executed_buy", "took_profit"
        details: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Record a trading decision.
        
        Args:
            phase: The trading phase (e.g., "symbol_selection", "entry", "exit")
            action: The specific action taken
            details: Dict with decision details (symbol, qty, price, etc.)
            metrics: Optional dict with additional metrics and context
        """
        decision = {
            "cycle_id": self.current_cycle_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "action": action,
            "details": details,
            "metrics": metrics or {},
        }
        
        # Append to decisions log
        with open(self.decisions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision) + "\n")
    
    def record_symbol_selection(
        self,
        selected_symbols: List[str],
        ranking: Dict[str, float],  # symbol -> score
        **context
    ):
        """Record symbol selection decision.
        
        Args:
            selected_symbols: List of symbols selected for trading
            ranking: Dict mapping symbol to ranking score
            **context: Additional context (max_symbols, criteria, etc.)
        """
        self.record_decision(
            phase="symbol_selection",
            action="selected_symbols",
            details={
                "symbols": selected_symbols,
                "count": len(selected_symbols),
            },
            metrics={
                "ranking": ranking,
                **context
            }
        )
    
    def record_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reason: str,
        **context
    ):
        """Record entry execution.
        
        Args:
            symbol: The ticker symbol
            side: "BUY" or "SELL"
            qty: Quantity traded
            price: Execution price
            reason: Reason for entry
            **context: Additional context (confidence, signal_strength, etc.)
        """
        self.record_decision(
            phase="entry",
            action=f"executed_{side.lower()}",
            details={
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "reason": reason,
            },
            metrics=context
        )
    
    def record_exit(
        self,
        symbol: str,
        qty: float,
        price: float,
        reason: str,
        pnl: float,
        **context
    ):
        """Record exit execution.
        
        Args:
            symbol: The ticker symbol
            qty: Quantity traded
            price: Exit price
            reason: Reason for exit (stop_loss, take_profit, eod_flatten, etc.)
            pnl: Realized P&L for this exit
            **context: Additional context (hold_duration, exit_type, etc.)
        """
        self.record_decision(
            phase="exit",
            action="executed_sell",
            details={
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "reason": reason,
                "pnl": pnl,
            },
            metrics=context
        )
    
    def get_cycle_decisions(self, cycle_id: str) -> List[dict]:
        """Retrieve all decisions from a specific cycle.
        
        Args:
            cycle_id: The cycle ID to retrieve decisions for
            
        Returns:
            List of decision dicts for the specified cycle
        """
        if not self.decisions_file.exists():
            return []
        
        decisions = []
        with open(self.decisions_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    decision = json.loads(line)
                    if decision.get("cycle_id") == cycle_id:
                        decisions.append(decision)
                except json.JSONDecodeError:
                    continue
        
        return decisions

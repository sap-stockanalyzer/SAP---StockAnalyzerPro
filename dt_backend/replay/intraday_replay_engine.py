"""dt_backend/replay/intraday_replay_engine.py

Replay historical trading cycles with modified parameters and compare P&L.

This engine allows you to:
1. Load historical cycle decisions
2. Apply modified knobs (stop_loss_pct, position_size, etc.)
3. Simulate execution with new parameters
4. Compare original vs replay P&L
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
from typing import Dict, List, Any, Optional, Tuple
import uuid


class IntraDayReplayEngine:
    """Replay historical trading cycles with modified parameters."""
    
    def __init__(self):
        from dt_backend.core.config_dt import DT_PATHS
        ml_data_dt = DT_PATHS.get("ml_data_dt", Path("ml_data_dt"))
        if not isinstance(ml_data_dt, Path):
            ml_data_dt = Path(ml_data_dt)
        self.replay_results_dir = ml_data_dt / "replay_results"
        self.replay_results_dir.mkdir(parents=True, exist_ok=True)
    
    def replay_cycle(
        self,
        cycle_id: str,
        modified_knobs: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, dict]] = None,
    ) -> Dict[str, Any]:
        """
        Replay a cycle with optional modified knobs.
        
        This method:
        1. Loads original decisions from the cycle
        2. Calculates original P&L
        3. Simulates replay execution with modified knobs
        4. Returns comparison of original vs replay results
        
        Args:
            cycle_id: Original cycle to replay
            modified_knobs: Dict of {knob_name: new_value} to override
                           Examples: {"stop_loss_pct": 0.02, "take_profit_pct": 0.05}
            market_data: Optional historical bar data for replay
        
        Returns:
            Dict with comparison of original vs replay results including:
            - original_pnl: P&L from original execution
            - replay_pnl: P&L from simulated replay
            - pnl_difference: Difference between replay and original
            - improvement_pct: Percentage improvement
        """
        from dt_backend.services.decision_recorder import DecisionRecorder
        
        # Get original decisions
        recorder = DecisionRecorder()
        original_decisions = recorder.get_cycle_decisions(cycle_id)
        
        if not original_decisions:
            return {"error": f"Cycle {cycle_id} not found"}
        
        # Extract original metrics
        original_pnl = self._calculate_cycle_pnl(original_decisions)
        
        # Create replay ID
        replay_id = uuid.uuid4().hex[:12]
        
        # Simulate replay with modified knobs
        # In a real implementation, this would re-execute using modified parameters
        # and potentially historical market data
        replay_pnl = self._simulate_replay_execution(
            original_decisions,
            modified_knobs,
            market_data
        )
        
        # Generate comparison
        result = {
            "original_cycle_id": cycle_id,
            "replay_id": replay_id,
            "original_pnl": original_pnl,
            "replay_pnl": replay_pnl,
            "pnl_difference": replay_pnl - original_pnl,
            "improvement_pct": ((replay_pnl - original_pnl) / abs(original_pnl) * 100) 
                              if original_pnl != 0 else 0,
            "modified_knobs": modified_knobs or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decisions_count": len(original_decisions),
        }
        
        # Save result
        self._save_replay_result(replay_id, result)
        
        return result
    
    def _calculate_cycle_pnl(self, decisions: List[dict]) -> float:
        """Calculate total P&L from cycle decisions.
        
        Sums up all realized P&L from exit decisions in the cycle.
        
        Args:
            decisions: List of decision dicts from a cycle
            
        Returns:
            Total P&L for the cycle
        """
        total_pnl = 0.0
        
        for decision in decisions:
            if decision.get("phase") == "exit" and "pnl" in decision.get("details", {}):
                total_pnl += decision["details"]["pnl"]
        
        return total_pnl
    
    def _simulate_replay_execution(
        self,
        original_decisions: List[dict],
        modified_knobs: Optional[Dict[str, Any]],
        market_data: Optional[Dict[str, dict]],
    ) -> float:
        """
        Simulate replay execution with modified parameters.
        
        This is a simplified implementation that adjusts P&L based on
        modified knobs. A full implementation would:
        1. Re-execute entry logic with modified position sizing
        2. Re-simulate exits with modified stop_loss/take_profit levels
        3. Use historical market data to calculate realistic fills
        
        Args:
            original_decisions: Original decisions from the cycle
            modified_knobs: Modified parameters to apply
            market_data: Historical price data (optional)
            
        Returns:
            Simulated P&L after applying modified knobs
        """
        # Simplified adjustment factors for knob modifications
        # These are rough estimates - real implementation would use historical data
        STOP_LOSS_ADJUSTMENT_FACTOR = 0.1   # 10% impact per percentage point
        TAKE_PROFIT_ADJUSTMENT_FACTOR = 0.05  # 5% impact per percentage point
        
        replay_pnl = 0.0
        
        # For each original decision, apply modified knobs and recalculate
        for decision in original_decisions:
            if decision["phase"] == "exit":
                original_pnl = decision["details"].get("pnl", 0)
                
                # Apply knob modifications
                # Example: if stop_loss_pct is tighter, exits might be earlier
                if modified_knobs:
                    # Simulate impact of modified stop_loss_pct
                    if "stop_loss_pct" in modified_knobs:
                        # Tighter stop loss might reduce losses or cut gains short
                        adjustment = modified_knobs["stop_loss_pct"] * STOP_LOSS_ADJUSTMENT_FACTOR
                        replay_pnl += original_pnl * (1 + adjustment)
                    # Simulate impact of modified take_profit_pct
                    elif "take_profit_pct" in modified_knobs:
                        # Higher take profit might capture more gains
                        adjustment = modified_knobs["take_profit_pct"] * TAKE_PROFIT_ADJUSTMENT_FACTOR
                        replay_pnl += original_pnl * (1 + adjustment)
                    else:
                        replay_pnl += original_pnl
                else:
                    replay_pnl += original_pnl
        
        return replay_pnl
    
    def _save_replay_result(self, replay_id: str, result: Dict[str, Any]):
        """Save replay result to disk for future reference.
        
        Args:
            replay_id: Unique ID for this replay
            result: Result dict to save
        """
        result_file = self.replay_results_dir / f"replay_{replay_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    
    def get_replay_results(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get all recent replay results.
        
        Args:
            days: Number of days back to retrieve results for
            
        Returns:
            List of replay result dicts from the last N days
        """
        if not self.replay_results_dir.exists():
            return []
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        results = []
        
        for result_file in sorted(self.replay_results_dir.glob("replay_*.json")):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                    # Safely parse timestamp
                    ts_str = result.get("timestamp")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts > cutoff:
                                results.append(result)
                        except (ValueError, TypeError):
                            # Skip files with invalid timestamps
                            continue
            except (json.JSONDecodeError, IOError):
                # Skip files that can't be parsed or read
                pass
        
        return results

"""dt_backend/ml/walk_forward_validator.py

Walk-forward validation for DT strategy on rolling windows.

This validator:
1. Creates rolling train/test windows over historical data
2. Evaluates strategy performance on each test window
3. Calculates metrics like Sharpe ratio, win rate, max drawdown
4. Aggregates results across all windows for overall assessment
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
from typing import Dict, List, Any, Optional
import numpy as np

# Constants for numeric stability
EPSILON = 1e-9  # Small value to prevent division by zero


class WalkForwardValidator:
    """Walk-forward validation for DT strategy."""
    
    def __init__(self, window_days: int = 5, lookback_days: int = 20):
        """Initialize walk-forward validator.
        
        Args:
            window_days: Size of each test window in days
            lookback_days: Size of training window in days (before test window)
        """
        from dt_backend.core.config_dt import DT_PATHS
        ml_data_dt = DT_PATHS.get("ml_data_dt", Path("ml_data_dt"))
        if not isinstance(ml_data_dt, Path):
            ml_data_dt = Path(ml_data_dt)
        self.wf_results_dir = ml_data_dt / "walk_forward_results"
        self.wf_results_dir.mkdir(parents=True, exist_ok=True)
        self.window_days = window_days
        self.lookback_days = lookback_days
    
    def run_validation(self, days_back: int = 60) -> Dict[str, Any]:
        """
        Run walk-forward validation on historical data.
        
        Creates rolling windows (train/test splits) and evaluates
        strategy performance on each test window.
        
        Args:
            days_back: How many days of history to validate
        
        Returns:
            Summary of validation results across all windows including:
            - windows: Number of test windows evaluated
            - total_pnl: Aggregate P&L across all windows
            - avg_sharpe: Average Sharpe ratio
            - avg_win_rate: Average win rate
            - consistent: Boolean indicating if all windows were profitable
        """
        from dt_backend.services.dt_truth_store import read_json, trades_path
        
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=days_back)).date()
        
        window_results = []
        
        # Load all trades from dt_trades.jsonl
        all_trades = self._load_trades_from_jsonl(trades_path())
        
        # Create rolling windows
        current_date = start_date
        while current_date < now.date():
            test_start = current_date
            test_end = test_start + timedelta(days=self.window_days)
            train_start = test_start - timedelta(days=self.lookback_days)
            
            # Filter trades in test window
            window_trades = [
                t for t in all_trades
                if self._parse_date(t.get("ts", "")) >= test_start and 
                   self._parse_date(t.get("ts", "")) < test_end and
                   t.get("type") in ["fill", "exit"]
            ]
            
            if window_trades:
                # Calculate metrics for this window
                metrics = self._calculate_window_metrics(window_trades)
                
                window_results.append({
                    "train_period": f"{train_start} to {test_start}",
                    "test_period": f"{test_start} to {test_end}",
                    "trades": len(window_trades),
                    **metrics,
                })
            
            current_date = test_end
        
        # Aggregate results across all windows
        summary = self._aggregate_results(window_results)
        
        # Save validation results to disk
        self._save_validation_results(summary, window_results)
        
        return summary
    
    def _load_trades_from_jsonl(self, trades_file: Path) -> List[dict]:
        """Load trades from dt_trades.jsonl file.
        
        Args:
            trades_file: Path to dt_trades.jsonl
            
        Returns:
            List of trade event dicts
        """
        if not trades_file.exists():
            return []
        
        trades = []
        with open(trades_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    trade = json.loads(line)
                    trades.append(trade)
                except json.JSONDecodeError:
                    continue
        
        return trades
    
    def _parse_date(self, ts_str: str) -> datetime.date:
        """Parse ISO timestamp string to date.
        
        Args:
            ts_str: ISO format timestamp string
            
        Returns:
            Date object, or epoch date if parsing fails
        """
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).date()
        except Exception:
            return datetime(1970, 1, 1).date()
    
    def _calculate_window_metrics(self, trades: List[dict]) -> Dict[str, float]:
        """Calculate performance metrics for a test window.
        
        Computes standard trading metrics:
        - Total P&L
        - Average P&L per trade
        - Win rate (% of profitable trades)
        - Sharpe ratio (risk-adjusted returns)
        - Max drawdown (largest peak-to-trough decline)
        
        Args:
            trades: List of trade events in the window
            
        Returns:
            Dict with calculated metrics
        """
        if not trades:
            return {}
        
        # Extract P&L values from trades
        pnls = []
        for t in trades:
            # Try to get pnl from different possible locations
            pnl = t.get("pnl")
            if pnl is None and "details" in t:
                pnl = t["details"].get("pnl")
            if pnl is not None:
                pnls.append(float(pnl))
        
        if not pnls:
            return {}
        
        total_pnl = sum(pnls)
        wins = len([p for p in pnls if p > 0])
        losses = len([p for p in pnls if p < 0])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Calculate Sharpe ratio
        # Annualized assuming 252 trading days
        returns = np.array(pnls)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (np.abs(running_max) + EPSILON)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            "total_pnl": float(total_pnl),
            "avg_pnl_per_trade": float(np.mean(pnls)),
            "win_rate": float(win_rate),
            "wins": wins,
            "losses": losses,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
        }
    
    def _aggregate_results(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all test windows.
        
        Computes summary statistics across all windows to assess
        overall strategy consistency and performance.
        
        Args:
            window_results: List of per-window result dicts
            
        Returns:
            Aggregated summary dict
        """
        if not window_results:
            return {"status": "no_data"}
        
        sharpes = [w.get("sharpe_ratio", 0) for w in window_results]
        win_rates = [w.get("win_rate", 0) for w in window_results]
        pnls = [w.get("total_pnl", 0) for w in window_results]
        
        return {
            "windows": len(window_results),
            "total_pnl": float(sum(pnls)),
            "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0,
            "avg_win_rate": float(np.mean(win_rates)) if win_rates else 0,
            "min_sharpe": float(min(sharpes)) if sharpes else 0,
            "max_sharpe": float(max(sharpes)) if sharpes else 0,
            "consistent": all(s > 0.5 for s in sharpes),  # All windows profitable
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _save_validation_results(
        self,
        summary: Dict[str, Any],
        window_results: List[Dict[str, Any]]
    ):
        """Save validation results to disk for future reference.
        
        Args:
            summary: Aggregated summary dict
            window_results: List of per-window results
        """
        timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
        result_file = self.wf_results_dir / f"wf_validation_{timestamp}.json"
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "windows": window_results,
            }, f, indent=2)

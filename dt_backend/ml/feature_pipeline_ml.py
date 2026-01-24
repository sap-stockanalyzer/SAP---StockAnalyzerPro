"""
Feature pipeline for ML model training and inference.

Builds training-ready features including:
- Technical indicators (momentum, volatility, volume)
- Attribution features (pnl_by_feature, pnl_by_strategy)
- Regime and context features
- Execution quality features (slippage, hold_time)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from dt_backend.core.constants_dt import FEATURE_IMPORTANCE_TOP_N
from dt_backend.ml.feature_importance_tracker import FeatureImportanceTracker

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


class MLFeaturePipeline:
    """Generate ML-ready features with P&L attribution."""
    
    def __init__(self):
        self.importance_tracker = FeatureImportanceTracker()
        self.trades_cache = {}
    
    def build_features(
        self, 
        rolling: Dict[str, Any],
        trades: List[Dict],
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Build complete feature set for ML training.
        
        Features include:
        - Technical: momentum, volatility, volume, RSI, MACD
        - Attribution: pnl_by_feature, pnl_by_strategy, contribution_pct
        - Regime: bull/bear/chop, VIX level, market regime
        - Execution: fill_rate, slippage, hold_time_mins
        - Target: next_1h_return, win_rate_next_5trades
        
        Args:
            rolling: Current rolling state
            trades: Recent trades for attribution
            symbols: Universe of symbols
        
        Returns:
            DataFrame with features and target
        """
        features_list = []
        
        for symbol in symbols:
            if symbol.startswith("_"):
                continue
            
            node = rolling.get(symbol, {})
            features_dt = node.get("features_dt", {})
            
            # Skip if no features
            if not features_dt:
                continue
            
            # Build feature row
            row = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            # 1. Technical features
            row.update(self._extract_technical_features(features_dt))
            
            # 2. Attribution features
            symbol_trades = [t for t in trades if t.get("symbol") == symbol]
            row.update(self._extract_attribution_features(symbol_trades))
            
            # 3. Regime features
            global_state = rolling.get("_GLOBAL_DT", {})
            row.update(self._extract_regime_features(global_state))
            
            # 4. Execution features
            position = node.get("position_dt", {})
            row.update(self._extract_execution_features(position))
            
            # 5. Policy features
            policy = node.get("policy_dt", {})
            row.update(self._extract_policy_features(policy))
            
            # 6. Target: next 1h return (if we have recent trades)
            if symbol_trades:
                row["target_pnl_1h"] = self._calculate_target_pnl(symbol_trades[-1])
                row["target_win_next_5"] = self._calculate_win_rate_next_n(symbol_trades, n=5)
            else:
                row["target_pnl_1h"] = 0.0
                row["target_win_next_5"] = 0.5
            
            features_list.append(row)
        
        df = pd.DataFrame(features_list)
        
        # Validate features
        self._validate_features(df)
        
        return df
    
    def _extract_technical_features(self, features_dt: Dict) -> Dict[str, float]:
        """Extract technical indicator features."""
        return {
            "last_price": float(features_dt.get("last_price", 0.0)),
            "atr_14": float(features_dt.get("atr_14", 0.0)),
            "rsi_14": float(features_dt.get("rsi_14", 50.0)),
            "macd": float(features_dt.get("macd", 0.0)),
            "bb_width": float(features_dt.get("bb_width", 0.0)),
            "momentum_1h": float(features_dt.get("momentum_1h", 0.0)),
            "volume_sma": float(features_dt.get("volume_sma", 0.0)),
            "vwap_deviation": float(features_dt.get("vwap_deviation", 0.0)),
        }
    
    def _extract_attribution_features(self, trades: List[Dict]) -> Dict[str, float]:
        """Extract P&L attribution features."""
        if not trades:
            return {
                "recent_pnl_5trades": 0.0,
                "win_rate_recent": 0.5,
                "avg_trade_size": 0.0,
                "max_win_recent": 0.0,
                "max_loss_recent": 0.0,
            }
        
        recent = trades[-5:]  # Last 5 trades
        pnls = [t.get("pnl", 0) for t in recent]
        
        return {
            "recent_pnl_5trades": sum(pnls),
            "win_rate_recent": sum(1 for p in pnls if p > 0) / len(pnls),
            "avg_trade_size": np.mean(pnls),
            "max_win_recent": max(pnls) if pnls else 0.0,
            "max_loss_recent": min(pnls) if pnls else 0.0,
        }
    
    def _extract_regime_features(self, global_state: Dict) -> Dict[str, float]:
        """Extract market regime features."""
        regime_dt = global_state.get("regime_dt", {})
        regime_label = regime_dt.get("label", "chop")
        
        # One-hot encode regime
        regime_encoding = {
            "bull": [1, 0, 0, 0, 0],
            "bear": [0, 1, 0, 0, 0],
            "chop": [0, 0, 1, 0, 0],
            "panic": [0, 0, 0, 1, 0],
            "stress": [0, 0, 0, 0, 1],
        }
        
        encoding = regime_encoding.get(regime_label, [0, 0, 1, 0, 0])
        
        return {
            "regime_bull": float(encoding[0]),
            "regime_bear": float(encoding[1]),
            "regime_chop": float(encoding[2]),
            "regime_panic": float(encoding[3]),
            "regime_stress": float(encoding[4]),
            "vix_level": float(global_state.get("vix_level", 20.0)),
        }
    
    def _extract_execution_features(self, position: Dict) -> Dict[str, float]:
        """Extract execution quality features."""
        if not position or position.get("qty", 0) == 0:
            return {
                "position_age_mins": 0.0,
                "slippage_pct": 0.0,
                "fill_rate": 1.0,
            }
        
        entry_ts = position.get("entry_ts", "")
        if entry_ts:
            try:
                entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                hold_minutes = (now - entry_dt).total_seconds() / 60.0
            except (ValueError, TypeError, AttributeError) as e:
                log(f"[ml_pipeline] ⚠️ Error parsing entry_ts '{entry_ts}': {e}")
                hold_minutes = 0.0
        else:
            hold_minutes = 0.0
        
        return {
            "position_age_mins": float(hold_minutes),
            "slippage_pct": float(position.get("slippage_pct", 0.0)),
            "fill_rate": float(position.get("fill_rate", 1.0)),
        }
    
    def _extract_policy_features(self, policy: Dict) -> Dict[str, float]:
        """Extract policy decision features."""
        if not policy:
            return {
                "confidence": 0.5,
                "score": 0.0,
                "p_hit": 0.5,
            }
        
        # One-hot encode action
        action = str(policy.get("action", "HOLD")).upper()
        action_encoding = {
            "BUY": [1, 0, 0, 0],
            "SELL": [0, 1, 0, 0],
            "HOLD": [0, 0, 1, 0],
            "STAND_DOWN": [0, 0, 0, 1],
        }
        
        encoding = action_encoding.get(action, [0, 0, 1, 0])
        
        return {
            "action_buy": float(encoding[0]),
            "action_sell": float(encoding[1]),
            "action_hold": float(encoding[2]),
            "action_stand_down": float(encoding[3]),
            "confidence": float(policy.get("confidence", 0.5)),
            "score": float(policy.get("score", 0.0)),
            "p_hit": float(policy.get("p_hit", 0.5)),
            "trade_gate": float(policy.get("trade_gate", False)),
        }
    
    def _calculate_target_pnl(self, trade: Dict) -> float:
        """Calculate target P&L for this trade."""
        # Look ahead 1 hour to see if trade made money
        # (in production, this would use future data during training)
        return float(trade.get("pnl", 0.0))
    
    def _calculate_win_rate_next_n(self, trades: List[Dict], n: int) -> float:
        """Calculate win rate for next N trades."""
        if len(trades) < n:
            return 0.5
        
        next_n = trades[-n:]
        wins = sum(1 for t in next_n if t.get("pnl", 0) > 0)
        return wins / n
    
    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate features are in expected ranges."""
        # Check for NaN
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            log(f"[ml_pipeline] ⚠️ NaN columns: {nan_cols}")
            df.fillna(0.0, inplace=True)
        
        # Check for inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df[numeric_cols])
        inf_cols = numeric_cols[inf_mask.any()].tolist()
        if inf_cols:
            log(f"[ml_pipeline] ⚠️ Inf columns: {inf_cols}")
            df.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        log(f"[ml_pipeline] ✅ Validated {len(df)} feature rows")
    
    def save_features(self, df: pd.DataFrame, filename: str = "features_latest.parquet"):
        """Save features to file for model training."""
        try:
            output_path = Path(f"ml_data_dt/{filename}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            log(f"[ml_pipeline] ✅ Saved {len(df)} features to {filename}")
        except Exception as e:
            log(f"[ml_pipeline] ❌ Error saving features: {e}")

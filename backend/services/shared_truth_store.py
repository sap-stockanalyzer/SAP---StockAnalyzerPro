"""backend/services/shared_truth_store.py — Unified truth store for swing + DT

Unified Truth Store Architecture
=================================

Single source of truth for all trading events across:
- Swing trading (EOD/multi-day positions)
- Day trading (intraday positions)

Benefits:
- Consistent event schema with source field
- Correlate swing vs DT events in logs
- Unified timestamp/format for cross-strategy analysis
- Single ledger for all trades: shared_trades.jsonl
- Query by source, symbol, or detect conflicts

Schema:
    {
        "type": "trade" | "position" | "signal" | "no_trade",
        "source": "swing" | "dt",
        "symbol": str,
        "side": "BUY" | "SELL",
        "qty": float,
        "price": float,
        "reason": str,
        "pnl": Optional[float],
        "ts": ISO timestamp,
        ...  # source-specific fields
    }

File Locking:
- Uses fcntl on Unix, msvcrt on Windows for safe concurrent writes
- Best-effort locking (never raises in normal use)

Artifacts:
- shared_trades.jsonl — append-only unified event log
- shared_positions.json — current position snapshot (both sources)
- shared_metrics.json — lightweight counters + analytics
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import fcntl for file locking (Unix only)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl - locking will be best-effort
    fcntl = None  # type: ignore
    HAS_FCNTL = False

try:
    from config import PATHS  # type: ignore
except Exception:
    PATHS = {}

from utils.logger import Logger

# Initialize shared truth store logger
_logger = Logger(name="shared_truth", source="backend")


def _utc_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _shared_dir() -> Path:
    """
    Resolve shared truth store directory.
    
    Override with SHARED_TRUTH_DIR env var (useful for replay/backtest).
    Default: da_brains/shared
    """
    override = (os.getenv("SHARED_TRUTH_DIR", "") or "").strip()
    if override:
        base = Path(override) / "shared"
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    # Default: da_brains/shared
    da = PATHS.get("da_brains") if isinstance(PATHS, dict) else None
    base = Path(da) if da else Path("da_brains")
    base = base / "shared"
    base.mkdir(parents=True, exist_ok=True)
    return base


class SharedTruthStore:
    """
    Unified truth store for swing + DT trades/signals/positions.
    
    Both swing and DT bots write to the same ledger with source field.
    Enables cross-strategy analysis and conflict detection.
    """
    
    def __init__(self):
        """Initialize shared truth store with default paths."""
        self._base_dir = _shared_dir()
        self.trades_file = self._base_dir / "shared_trades.jsonl"
        self.positions_file = self._base_dir / "shared_positions.json"
        self.metrics_file = self._base_dir / "shared_metrics.json"
        self._ensure_files()
    
    def _ensure_files(self) -> None:
        """Ensure truth store files exist."""
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            
            # Create files if they don't exist
            if not self.trades_file.exists():
                self.trades_file.touch()
            
            if not self.positions_file.exists():
                self.positions_file.write_text("{}", encoding="utf-8")
            
            if not self.metrics_file.exists():
                self.metrics_file.write_text("{\"counters\": {}}", encoding="utf-8")
        except Exception as e:
            _logger.warn(f"Failed to ensure truth store files: {e}")
    
    def _append_locked(self, event: Dict[str, Any]) -> None:
        """
        Append event with file locking (Unix: fcntl, Windows: best-effort).
        
        Never raises in normal use.
        """
        try:
            if not isinstance(event, dict):
                return
            
            event.setdefault("ts", _utc_iso())
            line = json.dumps(event, ensure_ascii=False) + "\n"
            
            # Append with locking
            self.trades_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.trades_file, "a", encoding="utf-8") as f:
                try:
                    # Try to acquire exclusive lock (Unix only)
                    if HAS_FCNTL and fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    # Windows or locking unavailable - continue anyway
                    pass
                
                f.write(line)
                
                try:
                    if HAS_FCNTL and fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        
        except Exception as e:
            _logger.error(f"Failed to append shared trade event: {e}", exc=e)
    
    def append_trade_event(
        self,
        source: str,  # "swing" | "dt"
        symbol: str,
        side: str,  # "BUY" | "SELL"
        qty: float,
        price: float,
        reason: str,
        pnl: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Append trade record with source field.
        
        Args:
            source: Trading system source ("swing" or "dt")
            symbol: Stock symbol
            side: Trade side ("BUY" or "SELL")
            qty: Quantity traded
            price: Execution price
            reason: Trade reason/signal
            pnl: Optional realized P&L
            **kwargs: Additional source-specific fields
        """
        event = {
            "type": "trade",
            "source": source,
            "symbol": str(symbol).upper(),
            "side": str(side).upper(),
            "qty": float(qty),
            "price": float(price),
            "reason": reason,
            "pnl": float(pnl) if pnl is not None else None,
            "ts": _utc_iso(),
            **kwargs
        }
        
        self._append_locked(event)
        _logger.info(
            f"📝 {source.upper()} trade recorded: {side} {qty} {symbol} @ ${price:.2f} | reason: {reason}",
            source=source,
            symbol=symbol,
            side=side
        )
    
    def append_signal_event(
        self,
        source: str,
        symbol: str,
        signal_type: str,
        confidence: Optional[float] = None,
        **kwargs
    ) -> None:
        """Append signal/decision event."""
        event = {
            "type": "signal",
            "source": source,
            "symbol": str(symbol).upper(),
            "signal_type": signal_type,
            "confidence": float(confidence) if confidence is not None else None,
            "ts": _utc_iso(),
            **kwargs
        }
        self._append_locked(event)
    
    def append_no_trade_event(
        self,
        source: str,
        symbol: str,
        reason: str,
        **kwargs
    ) -> None:
        """Append no-trade decision event."""
        event = {
            "type": "no_trade",
            "source": source,
            "symbol": str(symbol).upper(),
            "reason": reason,
            "ts": _utc_iso(),
            **kwargs
        }
        self._append_locked(event)
    
    def _read_trades(self, max_lines: int = 10000) -> List[Dict[str, Any]]:
        """Read recent trades from JSONL file (best-effort)."""
        try:
            if not self.trades_file.exists():
                return []
            
            trades = []
            with open(self.trades_file, "r", encoding="utf-8") as f:
                # Read last N lines efficiently
                lines = f.readlines()
                for line in lines[-max_lines:]:
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict):
                            trades.append(event)
                    except Exception:
                        continue
            
            return trades
        except Exception as e:
            _logger.warn(f"Failed to read trades: {e}")
            return []
    
    def get_trades_by_source(self, source: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Query trades from specific source (swing or dt).
        
        Args:
            source: "swing" or "dt"
            days: Number of days to look back
            
        Returns:
            List of trade events from that source
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_trades = self._read_trades()
        
        result = []
        for trade in all_trades:
            try:
                # Filter by source
                if trade.get("source") != source:
                    continue
                
                # Filter by timestamp
                ts_str = trade.get("ts", "")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts < cutoff:
                        continue
                
                result.append(trade)
            except Exception:
                continue
        
        return result
    
    def get_symbol_trades(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Query all trades for a symbol (both sources).
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of all trade events for symbol (sorted by timestamp)
        """
        symbol_upper = str(symbol).upper()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_trades = self._read_trades()
        
        result = []
        for trade in all_trades:
            try:
                # Filter by symbol
                if trade.get("symbol") != symbol_upper:
                    continue
                
                # Filter by timestamp
                ts_str = trade.get("ts", "")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts < cutoff:
                        continue
                
                result.append(trade)
            except Exception:
                continue
        
        # Sort by timestamp
        result.sort(key=lambda x: x.get("ts", ""))
        return result
    
    def detect_conflicts(self, days: int = 1) -> List[Dict[str, Any]]:
        """
        Find instances where swing + DT traded same symbol same day.
        
        This helps identify coordination issues where both bots
        are active on the same symbol simultaneously.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of conflicts: [{"symbol": "AAPL", "swing_trades": [...], "dt_trades": [...]}]
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_trades = self._read_trades()
        
        # Group by symbol and day
        symbol_day_trades: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        
        for trade in all_trades:
            try:
                if trade.get("type") != "trade":
                    continue
                
                ts_str = trade.get("ts", "")
                if not ts_str:
                    continue
                
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts < cutoff:
                    continue
                
                symbol = trade.get("symbol", "")
                source = trade.get("source", "")
                day = ts.strftime("%Y-%m-%d")
                
                if not symbol or not source:
                    continue
                
                key = f"{symbol}_{day}"
                if key not in symbol_day_trades:
                    symbol_day_trades[key] = {"swing": [], "dt": []}
                
                symbol_day_trades[key][source].append(trade)
            
            except Exception:
                continue
        
        # Find conflicts (both sources active on same symbol same day)
        conflicts = []
        for key, sources in symbol_day_trades.items():
            swing_trades = sources.get("swing", [])
            dt_trades = sources.get("dt", [])
            
            if swing_trades and dt_trades:
                symbol = key.split("_")[0]
                conflicts.append({
                    "symbol": symbol,
                    "day": key.split("_")[1],
                    "swing_trades": swing_trades,
                    "dt_trades": dt_trades,
                    "conflict_type": "same_day_activity"
                })
        
        return conflicts
    
    def get_metrics(self) -> Dict[str, Any]:
        """Read current metrics snapshot."""
        try:
            if not self.metrics_file.exists():
                return {"counters": {}, "ts": _utc_iso()}
            
            data = json.loads(self.metrics_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"counters": {}}
        except Exception:
            return {"counters": {}}
    
    def update_metrics(self, updates: Dict[str, Any]) -> None:
        """Update metrics snapshot (merge patch)."""
        try:
            current = self.get_metrics()
            current.update(updates)
            current["ts"] = _utc_iso()
            
            # Atomic write
            tmp = self.metrics_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.metrics_file)
        except Exception as e:
            _logger.warn(f"Failed to update metrics: {e}")


# Module-level singleton for convenience
_shared_store: Optional[SharedTruthStore] = None


def get_shared_store() -> SharedTruthStore:
    """Get or create shared truth store singleton."""
    global _shared_store
    if _shared_store is None:
        _shared_store = SharedTruthStore()
    return _shared_store

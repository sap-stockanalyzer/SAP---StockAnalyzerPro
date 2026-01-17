"""dt_backend/services/execution_ledger.py — v1.0

Execution ledger for saga pattern (3-phase commit).

Implements atomic execution tracking to prevent race conditions and data loss:

Phase 1 (Pending): Record intent BEFORE broker API call
Phase 2 (Confirmed): Update after broker confirms fill
Phase 3 (Recorded): Mark complete after position/truth store updates

This ledger is append-only and provides:
- Idempotent restarts (can resume after crashes)
- Full audit trail of all execution attempts
- Recovery capability (replay to reconstruct state)
- No silent data loss

File: dt_execution_ledger.jsonl (append-only)

Example flow:
    1. record_pending("AAPL", "BUY", 10, 180.0) -> "exec_abc123"
    2. [broker API call happens]
    3. record_confirmed("exec_abc123", "broker_12345", fill_ts)
    4. [update positions + truth store atomically]
    5. record_recorded("exec_abc123", position_snapshot)
    
If any step fails, the ledger shows exactly what happened.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dt_backend.core.file_locking import AppendLocked
from dt_backend.core.logger_dt import log
from dt_backend.core.time_override_dt import now_utc as _now_utc_override

# Ledger location
def _intraday_dir() -> Path:
    """Resolve intraday artifact directory."""
    override = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if override:
        base = Path(override) / "intraday"
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    try:
        from dt_backend.core import DT_PATHS
        da = DT_PATHS.get("da_brains")
        if isinstance(da, Path):
            base = da / "intraday"
        else:
            base = Path("da_brains") / "intraday"
    except Exception:
        base = Path("da_brains") / "intraday"
    
    base.mkdir(parents=True, exist_ok=True)
    return base


def _ledger_path() -> Path:
    """Path to execution ledger file."""
    p = _intraday_dir() / "dt_execution_ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _utc_iso(dt: Optional[datetime] = None) -> str:
    """UTC ISO timestamp for ledger entries."""
    if dt is None:
        dt = _now_utc_override()
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class ExecutionRecord:
    """Represents one execution attempt in the ledger."""
    
    execution_id: str
    symbol: str
    side: str  # BUY or SELL
    qty: float
    price: float  # Intent price
    status: str  # pending, confirmed, recorded, failed
    ts: str  # Initial timestamp
    
    # Phase 2 fields (after broker confirmation)
    fill_ts: Optional[str] = None
    broker_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    
    # Phase 3 fields (after position update)
    recorded_ts: Optional[str] = None
    position_state: Optional[Dict[str, Any]] = None
    
    # Error tracking
    error_msg: Optional[str] = None
    error_ts: Optional[str] = None
    
    # Metadata
    bot: Optional[str] = None
    confidence: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def _generate_execution_id() -> str:
    """Generate unique execution ID."""
    return f"exec_{uuid.uuid4().hex[:12]}"


def record_pending(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    *,
    bot: Optional[str] = None,
    confidence: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
) -> str:
    """Record pending execution (Phase 1 of saga).
    
    Call this BEFORE submitting to broker API.
    
    Args:
        symbol: Ticker symbol
        side: "BUY" or "SELL"
        qty: Quantity
        price: Intent price
        bot: Bot identifier (optional)
        confidence: Signal confidence (optional)
        meta: Additional metadata (optional)
        now_utc: Timestamp override for replay (optional)
        
    Returns:
        str: Execution ID for tracking through phases
    """
    exec_id = _generate_execution_id()
    ts = _utc_iso(now_utc)
    
    record = ExecutionRecord(
        execution_id=exec_id,
        symbol=str(symbol).upper().strip(),
        side=str(side).upper().strip(),
        qty=float(qty),
        price=float(price),
        status="pending",
        ts=ts,
        bot=bot,
        confidence=confidence,
        meta=meta,
    )
    
    # Append to ledger with locking
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    success = AppendLocked(_ledger_path(), line, timeout=5.0)
    
    if success:
        log(f"[exec_ledger] ✅ Phase 1 (pending): {exec_id} {side} {qty} {symbol} @ {price}")
    else:
        log(f"[exec_ledger] ⚠️ Failed to record pending: {exec_id}")
    
    return exec_id


def record_confirmed(
    execution_id: str,
    broker_order_id: str,
    fill_price: float,
    *,
    fill_ts: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> bool:
    """Record confirmed fill (Phase 2 of saga).
    
    Call this AFTER broker confirms the fill.
    
    Args:
        execution_id: ID from record_pending()
        broker_order_id: Broker's order ID
        fill_price: Actual fill price
        fill_ts: Fill timestamp from broker (optional)
        now_utc: Timestamp override for replay (optional)
        
    Returns:
        bool: True if successfully recorded
    """
    if fill_ts is None:
        fill_ts = _utc_iso(now_utc)
    
    # Read existing record to get base info
    existing = _find_record(execution_id)
    if existing is None:
        log(f"[exec_ledger] ⚠️ Cannot confirm unknown execution: {execution_id}")
        return False
    
    # Create confirmed record
    record = ExecutionRecord(
        execution_id=execution_id,
        symbol=existing.get("symbol", ""),
        side=existing.get("side", ""),
        qty=existing.get("qty", 0.0),
        price=existing.get("price", 0.0),
        status="confirmed",
        ts=existing.get("ts", ""),
        fill_ts=fill_ts,
        broker_order_id=broker_order_id,
        fill_price=float(fill_price),
        bot=existing.get("bot"),
        confidence=existing.get("confidence"),
        meta=existing.get("meta"),
    )
    
    # Append to ledger
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    success = AppendLocked(_ledger_path(), line, timeout=5.0)
    
    if success:
        log(f"[exec_ledger] ✅ Phase 2 (confirmed): {execution_id} filled @ {fill_price}")
    else:
        log(f"[exec_ledger] ⚠️ Failed to record confirmed: {execution_id}")
    
    return success


def record_recorded(
    execution_id: str,
    position_state: Dict[str, Any],
    *,
    now_utc: Optional[datetime] = None,
) -> bool:
    """Record final state (Phase 3 of saga).
    
    Call this AFTER atomically updating positions + truth store.
    
    Args:
        execution_id: ID from record_pending()
        position_state: Snapshot of position after update
        now_utc: Timestamp override for replay (optional)
        
    Returns:
        bool: True if successfully recorded
    """
    recorded_ts = _utc_iso(now_utc)
    
    # Read existing record
    existing = _find_record(execution_id)
    if existing is None:
        log(f"[exec_ledger] ⚠️ Cannot finalize unknown execution: {execution_id}")
        return False
    
    # Create recorded record
    record = ExecutionRecord(
        execution_id=execution_id,
        symbol=existing.get("symbol", ""),
        side=existing.get("side", ""),
        qty=existing.get("qty", 0.0),
        price=existing.get("price", 0.0),
        status="recorded",
        ts=existing.get("ts", ""),
        fill_ts=existing.get("fill_ts"),
        broker_order_id=existing.get("broker_order_id"),
        fill_price=existing.get("fill_price"),
        recorded_ts=recorded_ts,
        position_state=position_state,
        bot=existing.get("bot"),
        confidence=existing.get("confidence"),
        meta=existing.get("meta"),
    )
    
    # Append to ledger
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    success = AppendLocked(_ledger_path(), line, timeout=5.0)
    
    if success:
        log(f"[exec_ledger] ✅ Phase 3 (recorded): {execution_id} complete")
    else:
        log(f"[exec_ledger] ⚠️ Failed to record finalized: {execution_id}")
    
    return success


def record_failed(
    execution_id: str,
    error_msg: str,
    *,
    now_utc: Optional[datetime] = None,
) -> bool:
    """Record failed execution.
    
    Call this when any phase fails.
    
    Args:
        execution_id: ID from record_pending()
        error_msg: Error description
        now_utc: Timestamp override for replay (optional)
        
    Returns:
        bool: True if successfully recorded
    """
    error_ts = _utc_iso(now_utc)
    
    # Read existing record
    existing = _find_record(execution_id)
    if existing is None:
        log(f"[exec_ledger] ⚠️ Cannot mark unknown execution as failed: {execution_id}")
        return False
    
    # Create failed record
    record = ExecutionRecord(
        execution_id=execution_id,
        symbol=existing.get("symbol", ""),
        side=existing.get("side", ""),
        qty=existing.get("qty", 0.0),
        price=existing.get("price", 0.0),
        status="failed",
        ts=existing.get("ts", ""),
        fill_ts=existing.get("fill_ts"),
        broker_order_id=existing.get("broker_order_id"),
        fill_price=existing.get("fill_price"),
        error_msg=str(error_msg)[:500],  # Truncate to keep ledger manageable
        error_ts=error_ts,
        bot=existing.get("bot"),
        confidence=existing.get("confidence"),
        meta=existing.get("meta"),
    )
    
    # Append to ledger
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    success = AppendLocked(_ledger_path(), line, timeout=5.0)
    
    if success:
        log(f"[exec_ledger] ❌ Recorded failure: {execution_id} - {error_msg}")
    else:
        log(f"[exec_ledger] ⚠️ Failed to record failure: {execution_id}")
    
    return success


def _find_record(execution_id: str) -> Optional[Dict[str, Any]]:
    """Find most recent record for execution_id in ledger.
    
    Scans ledger backward to find latest state.
    Memory-efficient: processes one line at a time.
    """
    ledger_file = _ledger_path()
    if not ledger_file.exists():
        return None
    
    try:
        # Read file backward line by line for memory efficiency
        result = None
        with open(ledger_file, "r", encoding="utf-8") as f:
            # Seek to end
            f.seek(0, 2)
            file_size = f.tell()
            
            # Read backward in chunks
            chunk_size = 8192
            position = file_size
            lines = []
            
            while position > 0:
                chunk_size = min(chunk_size, position)
                position -= chunk_size
                f.seek(position)
                chunk = f.read(chunk_size)
                
                # Split into lines and reverse
                chunk_lines = chunk.split('\n')
                lines = chunk_lines + lines
                
                # Process complete lines from the end
                while len(lines) > 1:
                    line = lines.pop()
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line.strip())
                        if record.get("execution_id") == execution_id:
                            return record
                    except Exception:
                        continue
        
        # Check remaining first line
        if lines and lines[0].strip():
            try:
                record = json.loads(lines[0].strip())
                if record.get("execution_id") == execution_id:
                    return record
            except Exception:
                pass
        
        return None
        
    except Exception as e:
        log(f"[exec_ledger] ⚠️ Error finding record {execution_id}: {e}")
        return None


def get_pending() -> List[Dict[str, Any]]:
    """Get all pending executions (not confirmed or recorded).
    
    Used for recovery on startup to detect incomplete executions.
    
    Returns:
        List of execution records in pending or confirmed state
    """
    ledger_file = _ledger_path()
    if not ledger_file.exists():
        return []
    
    try:
        # Build map of execution_id -> latest state
        executions: Dict[str, Dict[str, Any]] = {}
        
        with open(ledger_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    exec_id = record.get("execution_id")
                    if exec_id:
                        executions[exec_id] = record
                except Exception:
                    continue
        
        # Filter to pending/confirmed (not recorded or failed)
        pending = [
            rec for rec in executions.values()
            if rec.get("status") in {"pending", "confirmed"}
        ]
        
        return pending
        
    except Exception as e:
        log(f"[exec_ledger] ⚠️ Error getting pending executions: {e}")
        return []


def get_recent(limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent execution records.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of most recent execution records
    """
    ledger_file = _ledger_path()
    if not ledger_file.exists():
        return []
    
    try:
        # Build map of execution_id -> latest state
        executions: Dict[str, Dict[str, Any]] = {}
        
        with open(ledger_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    exec_id = record.get("execution_id")
                    if exec_id:
                        # Keep only latest record per execution
                        executions[exec_id] = record
                except Exception:
                    continue
        
        # Sort by timestamp (descending) and limit
        records = sorted(
            executions.values(),
            key=lambda r: r.get("ts", ""),
            reverse=True
        )
        
        return records[:limit]
        
    except Exception as e:
        log(f"[exec_ledger] ⚠️ Error getting recent executions: {e}")
        return []


def replay_ledger() -> Dict[str, Any]:
    """Replay ledger to reconstruct state (for crash recovery).
    
    This analyzes the ledger to determine:
    - Which executions completed successfully
    - Which are stuck in pending/confirmed (need manual review)
    - Overall execution success rate
    
    Returns:
        Dict with statistics and incomplete executions
    """
    ledger_file = _ledger_path()
    if not ledger_file.exists():
        return {
            "status": "no_ledger",
            "total": 0,
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "incomplete": [],
        }
    
    try:
        # Build map of execution_id -> latest state
        executions: Dict[str, Dict[str, Any]] = {}
        
        with open(ledger_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    exec_id = record.get("execution_id")
                    if exec_id:
                        executions[exec_id] = record
                except Exception:
                    continue
        
        # Analyze final states
        total = len(executions)
        completed = sum(1 for r in executions.values() if r.get("status") == "recorded")
        failed = sum(1 for r in executions.values() if r.get("status") == "failed")
        pending = sum(1 for r in executions.values() if r.get("status") in {"pending", "confirmed"})
        
        # Get incomplete executions for manual review
        incomplete = [
            r for r in executions.values()
            if r.get("status") in {"pending", "confirmed"}
        ]
        
        result = {
            "status": "ok",
            "total": total,
            "completed": completed,
            "pending": pending,
            "failed": failed,
            "incomplete": incomplete,
            "ledger_path": str(ledger_file),
        }
        
        log(f"[exec_ledger] 📊 Replay: {total} total, {completed} completed, {pending} pending, {failed} failed")
        
        return result
        
    except Exception as e:
        log(f"[exec_ledger] ⚠️ Error replaying ledger: {e}")
        return {
            "status": "error",
            "error": str(e),
            "total": 0,
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "incomplete": [],
        }

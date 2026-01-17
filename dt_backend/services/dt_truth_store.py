"""dt_backend/services/dt_truth_store.py — v0.3 (Phase 0 + Unified Truth Store)

Truth + safety helpers for dt_backend.

UPDATED: Now wraps SharedTruthStore for unified logging:
  - All DT trades go to shared_trades.jsonl with source="dt"
  - Backward compatible API maintained
  - State and metrics remain DT-specific
  - Enables cross-strategy analysis (swing vs DT)

This module intentionally avoids external dependencies.

Artifacts (append-only or atomic replace):
  • dt_state.json     — current regime/risk/bots/kill-switches snapshot
  • dt_trades.jsonl   — DEPRECATED: now forwards to shared store
  • dt_metrics.json   — rolling metrics snapshot (equity, positions, realized pnl, counters)

Shared artifacts (written under da_brains/shared):
  • shared_trades.jsonl — unified trades from swing + DT

Locking:
  • dt_scheduler.lock — prevents multiple dt_scheduler processes from running.
  • dt_cycle.lock     — prevents overlapping daytrading cycles across processes.
  • dt_bars_fetch.lock — prevents overlapping intraday bars fetch loops (optional)

All functions are best-effort and should never raise in normal use.
"""

from __future__ import annotations

import json
import uuid
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dt_backend.core.logger_dt import log
from dt_backend.core.time_override_dt import utc_iso
from dt_backend.core.file_locking import AppendLocked

# Import shared truth store for unified logging
try:
    from backend.services.shared_truth_store import get_shared_store
except Exception:  # pragma: no cover
    # Fallback if shared store not available
    def get_shared_store():  # type: ignore
        return None

# DT_PATHS source: prefer config_dt; fall back gracefully if core exports it.
try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:  # pragma: no cover
    try:
        from dt_backend.core import DT_PATHS  # type: ignore
    except Exception:  # pragma: no cover
        DT_PATHS = {}  # type: ignore


# Shared store event field exclusions (for forwarding events)
_SHARED_STORE_TRADE_FIELDS = {"source", "symbol", "side", "qty", "price", "reason", "pnl", "realized_pnl", "ts"}
_SHARED_STORE_SIGNAL_FIELDS = {"type", "source", "symbol", "signal_type", "confidence", "ts"}
_SHARED_STORE_NO_TRADE_FIELDS = {"type", "source", "symbol", "reason", "ts"}


def _utc_iso() -> str:
    # Backward-compat wrapper (modules may import this symbol).
    return utc_iso()


def _intraday_dir() -> Path:
    """Resolve the DT intraday artifact directory (best-effort).

    Replay/backtest can override the artifact root by setting:
        DT_TRUTH_DIR=/abs/path/to/run_dir

    In that case, artifacts are written under <DT_TRUTH_DIR>/intraday.
    """
    override = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if override:
        base = Path(override) / "intraday"
        base.mkdir(parents=True, exist_ok=True)
        return base

    # Prefer configured artifact dir if present.
    da = DT_PATHS.get("da_brains") if isinstance(DT_PATHS, dict) else None
    try:
        if da:
            base = Path(str(da)) / "intraday"
        else:
            base = Path("da_brains") / "intraday"
    except Exception:
        base = Path("da_brains") / "intraday"

    base.mkdir(parents=True, exist_ok=True)
    return base


def _fallback(name: str) -> Path:
    return _intraday_dir() / name


# Canonical artifact locations (prefer ROOT config.py DT_PATHS)
# IMPORTANT: replay/backtest must be able to redirect artifacts using DT_TRUTH_DIR.
# We therefore treat these as *defaults* and apply the DT_TRUTH_DIR override at call-time.
def _as_path(x: Any) -> Optional[Path]:
    try:
        if isinstance(x, Path):
            return x
        if isinstance(x, str) and x.strip():
            return Path(x.strip())
    except Exception:
        return None
    return None


STATE_PATH_DEFAULT = _as_path(DT_PATHS.get("dt_state_file") if isinstance(DT_PATHS, dict) else None) or _fallback("dt_state.json")
TRADES_PATH_DEFAULT = _as_path(DT_PATHS.get("dt_trades_file") if isinstance(DT_PATHS, dict) else None) or _fallback("dt_trades.jsonl")
METRICS_PATH_DEFAULT = _as_path(DT_PATHS.get("dt_metrics_file") if isinstance(DT_PATHS, dict) else None) or _fallback("dt_metrics.json")

# Upgrade Package v1 (Phase 1–2): missed opportunity evidence loop
MISSED_PATH_DEFAULT = _as_path(DT_PATHS.get("dt_missed_opportunities_file") if isinstance(DT_PATHS, dict) else None) or _fallback("dt_missed_opportunities.jsonl")
MISSED_EVALS_PATH_DEFAULT = _as_path(DT_PATHS.get("dt_missed_evals_file") if isinstance(DT_PATHS, dict) else None) or _fallback("dt_missed_evals.jsonl")

# Locks (same rule)
SCHED_LOCK_PATH_DEFAULT = _as_path(DT_PATHS.get("dt_scheduler_lock_file") if isinstance(DT_PATHS, dict) else None) or _fallback(".dt_scheduler.lock")
CYCLE_LOCK_PATH_DEFAULT = _as_path(DT_PATHS.get("dt_cycle_lock_file") if isinstance(DT_PATHS, dict) else None) or _fallback(".dt_cycle.lock")
BARS_FETCH_LOCK_PATH_DEFAULT = _as_path(DT_PATHS.get("dt_bars_fetch_lock_file") if isinstance(DT_PATHS, dict) else None) or _fallback(".dt_bars_fetch.lock")


def _override_path(default_path: Path, name: str) -> Path:
    """If DT_TRUTH_DIR is set, return <DT_TRUTH_DIR>/intraday/<name>.

    This makes replay/backtests hermetic: they never clobber live artifacts.
    """
    override = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if override:
        p = Path(override) / "intraday" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return default_path


# Exported paths (backward-compatible names)
STATE_PATH = STATE_PATH_DEFAULT
TRADES_PATH = TRADES_PATH_DEFAULT
METRICS_PATH = METRICS_PATH_DEFAULT
MISSED_PATH = MISSED_PATH_DEFAULT
MISSED_EVALS_PATH = MISSED_EVALS_PATH_DEFAULT
SCHED_LOCK_PATH = SCHED_LOCK_PATH_DEFAULT
CYCLE_LOCK_PATH = CYCLE_LOCK_PATH_DEFAULT
BARS_FETCH_LOCK_PATH = BARS_FETCH_LOCK_PATH_DEFAULT


def state_path() -> Path:
    return _override_path(STATE_PATH_DEFAULT, "dt_state.json")


def trades_path() -> Path:
    return _override_path(TRADES_PATH_DEFAULT, "dt_trades.jsonl")


def metrics_path() -> Path:
    return _override_path(METRICS_PATH_DEFAULT, "dt_metrics.json")


def missed_path() -> Path:
    return _override_path(MISSED_PATH_DEFAULT, "dt_missed_opportunities.jsonl")


def missed_evals_path() -> Path:
    return _override_path(MISSED_EVALS_PATH_DEFAULT, "dt_missed_evals.jsonl")


def sched_lock_path() -> Path:
    return _override_path(SCHED_LOCK_PATH_DEFAULT, ".dt_scheduler.lock")


def cycle_lock_path() -> Path:
    return _override_path(CYCLE_LOCK_PATH_DEFAULT, ".dt_cycle.lock")


def bars_fetch_lock_path() -> Path:
    return _override_path(BARS_FETCH_LOCK_PATH_DEFAULT, ".dt_bars_fetch.lock")


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def _read_lock_pid(lock_path: Path) -> int:
    try:
        raw = lock_path.read_text(encoding="utf-8", errors="ignore").strip()
        parts = raw.split()
        return int(parts[0]) if parts else -1
    except Exception:
        return -1


@dataclass
class LockHandle:
    path: Path
    acquired: bool

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        self.acquired = False


def acquire_lock(lock_path: Path, *, timeout_s: float = 10.0) -> LockHandle:
    """Acquire a PID lock file.

    Best-effort stale lock cleanup:
      - if PID is not alive, we remove the lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(0.0, float(timeout_s))

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                payload = f"{os.getpid()} {_utc_iso()}"
                os.write(fd, payload.encode("utf-8", errors="ignore"))
            finally:
                os.close(fd)
            return LockHandle(path=lock_path, acquired=True)

        except FileExistsError:
            pid = _read_lock_pid(lock_path)
            if pid > 0 and not _pid_alive(pid):
                try:
                    lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                    continue
                except Exception:
                    pass

            if time.time() >= deadline:
                return LockHandle(path=lock_path, acquired=False)
            time.sleep(0.15)

        except Exception:
            return LockHandle(path=lock_path, acquired=False)


# ---------------------------------------------------------------------------
# State / trades / metrics
# ---------------------------------------------------------------------------

def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        log(f"[dt_truth] ⚠️ failed to write {path.name}: {e}")


def update_dt_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge-patch dt_state.json (shallow merge), add timestamps."""
    state = read_json(state_path(), {})
    if not isinstance(state, dict):
        state = {}
    if not isinstance(patch, dict):
        patch = {}
    state.update(patch)
    state.setdefault("created_at", _utc_iso())
    state["updated_at"] = _utc_iso()
    atomic_write_json(state_path(), state)
    return state


def read_dt_state() -> Dict[str, Any]:
    """Read dt_state.json (best-effort)."""
    s = read_json(state_path(), {})
    return s if isinstance(s, dict) else {}


def append_trade_event(event: Dict[str, Any]) -> None:
    """Append one event line to shared truth store.
    
    Now writes to shared_trades.jsonl with source="dt" for unified logging.
    Also maintains local dt_trades.jsonl for backward compatibility.
    """
    """Append one event line to dt_trades.jsonl with file locking."""
    try:
        if not isinstance(event, dict):
            return
        event.setdefault("ts", _utc_iso())
        
        # Write to shared truth store with source="dt"
        shared_store = get_shared_store()
        if shared_store is not None:
            # Determine event type
            event_type = event.get("type", "trade")
            
            if event_type in {"trade", "order_submitted", "bracket_set", "fill_exit", "exit"} and "symbol" in event:
                # Normalize to trade event
                shared_store.append_trade_event(
                    source="dt",
                    symbol=event.get("symbol", ""),
                    side=event.get("side", ""),
                    qty=event.get("qty", 0.0),
                    price=event.get("price", 0.0),
                    reason=event.get("reason", ""),
                    pnl=event.get("pnl") or event.get("realized_pnl"),
                    **{k: v for k, v in event.items() if k not in _SHARED_STORE_TRADE_FIELDS}
                )
            elif event_type == "signal" and "symbol" in event:
                shared_store.append_signal_event(
                    source="dt",
                    symbol=event.get("symbol", ""),
                    signal_type=event.get("signal_type", ""),
                    confidence=event.get("confidence"),
                    **{k: v for k, v in event.items() if k not in _SHARED_STORE_SIGNAL_FIELDS}
                )
            elif event_type == "no_trade" and "symbol" in event:
                shared_store.append_no_trade_event(
                    source="dt",
                    symbol=event.get("symbol", ""),
                    reason=event.get("reason", ""),
                    **{k: v for k, v in event.items() if k not in _SHARED_STORE_NO_TRADE_FIELDS}
                )
        
        # Also write to local file for backward compatibility
        p = trades_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Use locked append to prevent race conditions
        line = json.dumps(event, ensure_ascii=False)
        success = AppendLocked(p, line, timeout=5.0)
        
        if not success:
            log(f"[dt_truth] ⚠️ failed to acquire lock for dt_trades append")
    except Exception as e:
        log(f"[dt_truth] ⚠️ failed to append dt_trades event: {e}")


# ---------------------------------------------------------------------------
# Upgrade Package v1 (Phase 1–2): missed opportunities
# ---------------------------------------------------------------------------


def append_missed_opportunity(event: Dict[str, Any]) -> None:
    """Append a 'missed opportunity candidate' line with file locking.

    This is a Phase 1 artifact: a record of a trade-like setup that was rejected
    by a *soft* gate (thresholds, EV filters, etc.).
    """
    try:
        if not isinstance(event, dict):
            return
        event.setdefault("ts", _utc_iso())
        # Ensure an id exists for later evaluation + attribution joins.
        # Keep it simple: unique, not necessarily deterministic.
        event.setdefault("event_id", uuid.uuid4().hex)
        p = missed_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Use locked append
        line = json.dumps(event, ensure_ascii=False)
        success = AppendLocked(p, line, timeout=5.0)
        
        if not success:
            log(f"[dt_truth] ⚠️ failed to acquire lock for missed opportunity append")
    except Exception as e:
        log(f"[dt_truth] ⚠️ failed to append missed opportunity: {e}")


def append_missed_eval(event: Dict[str, Any]) -> None:
    """Append a labeled outcome line for a missed candidate (Phase 2) with file locking."""
    try:
        if not isinstance(event, dict):
            return
        event.setdefault("ts", _utc_iso())
        p = missed_evals_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Use locked append
        line = json.dumps(event, ensure_ascii=False)
        success = AppendLocked(p, line, timeout=5.0)
        
        if not success:
            log(f"[dt_truth] ⚠️ failed to acquire lock for missed eval append")
    except Exception as e:
        log(f"[dt_truth] ⚠️ failed to append missed eval: {e}")


def _iter_broker_ledgers() -> Iterable[Path]:
    try:
        d = _intraday_dir() / "brokers"
        if not d.exists():
            return []
        return sorted(p for p in d.glob("bot_*.json") if p.is_file())
    except Exception:
        return []


def write_metrics_snapshot(*, rolling: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write dt_metrics.json.

    Lightweight snapshot, not a full analytics engine.
    Summarizes each bot ledger (cash, positions, equity estimate) + counters.
    """
    rolling = rolling if isinstance(rolling, dict) else {}

    # Last prices from rolling (best-effort)
    last_px: Dict[str, float] = {}
    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        feat = node.get("features_dt")
        if isinstance(feat, dict):
            try:
                px = float(feat.get("last_price") or 0.0)
                if px > 0:
                    last_px[str(sym).upper()] = px
                    continue
            except Exception:
                pass

        bars = node.get("bars_intraday")
        if isinstance(bars, list) and bars:
            b = bars[-1] if isinstance(bars[-1], dict) else None
            if b:
                try:
                    px = float(b.get("c") or b.get("close") or 0.0)
                    if px > 0:
                        last_px[str(sym).upper()] = px
                except Exception:
                    pass

    bots: Dict[str, Any] = {}
    for p in _iter_broker_ledgers():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue

            bot_id = str(raw.get("bot_id") or p.stem.replace("bot_", ""))
            cash = float(raw.get("cash") or 0.0)
            positions = _positions_bucket_view(raw.get("positions") or {})
            fills = raw.get("fills") or []

            realized = 0.0
            if isinstance(fills, list):
                for f in fills[-500:]:
                    if isinstance(f, dict):
                        try:
                            realized += float(f.get("realized_pnl") or 0.0)
                        except Exception:
                            pass

            pos_val = 0.0
            pos_count = 0
            if isinstance(positions, dict):
                for sym, pos in positions.items():
                    if not isinstance(pos, dict):
                        continue
                    try:
                        qty = float(pos.get("qty") or 0.0)
                        if qty == 0:
                            continue
                        pos_count += 1
                        sym_u = str(sym).upper()
                        px = last_px.get(sym_u, float(pos.get("avg_price") or 0.0))
                        pos_val += qty * float(px or 0.0)
                    except Exception:
                        continue

            equity = cash + pos_val
            bots[bot_id] = {
                "cash": cash,
                "positions": pos_count,
                "positions_value_est": pos_val,
                "equity_est": equity,
                "realized_pnl_est": realized,
                "ledger_path": str(p),
            }
        except Exception:
            continue

    out = {
        "ts": _utc_iso(),
        "bots": bots,
    }
    atomic_write_json(metrics_path(), out)
    return out


def bump_metric(name: str, amount: float = 1.0) -> None:
    """Increment a lightweight counter inside dt_metrics.json."""
    try:
        if not isinstance(name, str) or not name.strip():
            return
        amt = float(amount)
    except Exception:
        return

    m = read_json(metrics_path(), {})
    if not isinstance(m, dict):
        m = {}

    counters = m.get("counters")
    if not isinstance(counters, dict):
        counters = {}

    try:
        cur = float(counters.get(name) or 0.0)
    except Exception:
        cur = 0.0
    counters[name] = cur + amt

    m["counters"] = counters
    m["ts"] = _utc_iso()
    atomic_write_json(metrics_path(), m)
def _positions_bucket_view(positions: Any) -> Dict[str, Any]:
    """Normalize ledger positions to a flat {SYMBOL: {qty, avg_price}} dict.

    Supports both:
      - legacy flat dict: {"AAPL": {"qty":..,"avg_price":..}, ...}
      - bucketed schema: {"ACTIVE": {...}, "CARRY": {...}}

    Uses DT_LEDGER_READ_SCOPE to select ACTIVE/CARRY/ALL (default ACTIVE).
    """
    if not isinstance(positions, dict):
        return {}
    # Bucketed?
    if "ACTIVE" in positions or "CARRY" in positions:
        scope = (os.getenv("DT_LEDGER_READ_SCOPE", "ACTIVE") or "ACTIVE").strip().upper()
        active = positions.get("ACTIVE") if isinstance(positions.get("ACTIVE"), dict) else {}
        carry = positions.get("CARRY") if isinstance(positions.get("CARRY"), dict) else {}
        if scope == "ALL":
            out = dict(active)
            out.update(carry)
            return out
        if scope == "CARRY":
            return dict(carry)
        return dict(active)
    # Legacy flat
    return dict(positions)


def count_trades_today(symbol: str) -> int:
    """Count how many trades (entries) we made on this symbol today.
    
    Reads from dt_trades.jsonl for today's date.
    """
    try:
        from datetime import datetime, timezone
        import json
        from pathlib import Path
        import os
        
        # Find today's trades file
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades_file = Path(os.getenv("DT_TRUTH_DIR", "da_brains")) / "intraday" / "dt_trades.jsonl"
        
        if not trades_file.exists():
            return 0
        
        sym = str(symbol).upper().strip()
        count = 0
        
        with open(trades_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    evt = json.loads(line)
                    evt_date = str(evt.get("ts", ""))[:10]  # YYYY-MM-DD
                    
                    if evt_date != today:
                        continue
                    
                    # Count entry signals (not exits)
                    if evt.get("type") in {"order_submitted", "bracket_set"}:
                        if str(evt.get("symbol", "")).upper() == sym:
                            if str(evt.get("side", "")).upper() == "BUY":
                                count += 1
                except Exception:
                    continue
        
        return count
        
    except Exception:
        return 0


def get_symbol_pnl_today(symbol: str) -> float:
    """Calculate total realized P&L for symbol today (Phase 3).
    
    Scans dt_trades.jsonl for today's exit events for the given symbol
    and sums up the realized P&L.
    
    Returns:
        float: Total P&L for symbol today (negative = loss, positive = profit)
    """
    try:
        from datetime import datetime, timezone
        import json
        from pathlib import Path
        import os
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades_file = Path(os.getenv("DT_TRUTH_DIR", "da_brains")) / "intraday" / "dt_trades.jsonl"
        
        if not trades_file.exists():
            return 0.0
        
        sym = str(symbol).upper().strip()
        pnl = 0.0
        
        with open(trades_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    evt = json.loads(line)
                    evt_date = str(evt.get("ts", ""))[:10]  # YYYY-MM-DD
                    
                    if evt_date != today:
                        continue
                    
                    # Look for exit events with realized P&L
                    if evt.get("type") in {"exit", "fill_exit", "order_filled"}:
                        if str(evt.get("symbol", "")).upper() == sym:
                            # Add P&L from this exit
                            evt_pnl = evt.get("pnl") or evt.get("realized_pnl") or 0.0
                            try:
                                pnl += float(evt_pnl)
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    continue
        
        return float(pnl)
        
    except Exception:
        return 0.0



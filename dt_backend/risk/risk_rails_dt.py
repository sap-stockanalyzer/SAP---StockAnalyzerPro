# dt_backend/risk/risk_rails_dt.py — v1.2 (Phase 3)
"""Hard risk rails for dt_backend (Phase 3).

This is the "adult supervision" layer. It prevents the daytrading loop from
continuing to trade once predefined limits are violated.

Enforced constraints (env overrides)
-----------------------------------
- DT_DAILY_LOSS_LIMIT_USD (default 300.0)
- DT_DAILY_DRAWDOWN_PCT (default 0.02 => 2%)
- DT_MAX_WEEKLY_DRAWDOWN_PCT (default 8.0) — NEW in v1.2
- DT_MAX_MONTHLY_DRAWDOWN_PCT (default 15.0) — NEW in v1.2
- DT_MAX_OPEN_POSITIONS (default 3)
- DT_MAX_EXPOSURE_FRAC (default 0.55)
- DT_COOLDOWN_AFTER_LOSS_DELTAS (default 3)
- DT_COOLDOWN_MINUTES (default 20)
- DT_VIX_SPIKE_THRESHOLD (default 35.0) — NEW in v1.2
- DT_PAUSE_ON_VIX_SPIKE (default 1) — NEW in v1.2

v1.2 additions (Phase 3)
------------------------
- Weekly drawdown cap protection
- Monthly drawdown cap protection
- VIX spike protection (prevents trading during high volatility events)

v1.1 additions
--------------
- Optional broker equity/PnL source:
    DT_RISK_EQUITY_SOURCE = auto|broker|ledger  (default: auto)
    DT_RISK_BROKER_TTL_SEC = 180 (default)

  In auto mode we prefer broker equity when available, falling back to the
  local dt_metrics.json estimates.

Why this exists
---------------
Local ledgers are great for strategy-level accounting, but they can drift or be
out-of-sync during manual intervention or after crashes. Using broker-reported
account equity for *daily kill-switches* avoids false "daily_loss_limit_hit".

We still keep the local estimate around for exposure/position heuristics.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from dt_backend.core import DT_PATHS
from dt_backend.core.logger_dt import log
from dt_backend.core.position_registry import load_registry
from dt_backend.services.dt_truth_store import metrics_path, read_json, atomic_write_json

# Optional broker hooks (safe if unavailable)
try:
    from dt_backend.engines.broker_api import BrokerAPI  # type: ignore
except Exception:  # pragma: no cover
    BrokerAPI = None  # type: ignore


def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw != "" else float(default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return int(float(raw)) if raw != "" else int(default)
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ny_tz():
    if ZoneInfo is not None:
        return ZoneInfo("America/New_York")
    return timezone.utc


def _session_date_str(now_utc: Optional[datetime] = None) -> str:
    """Session date in NY time (most natural for US equities)."""
    now_utc = now_utc or _utc_now()
    try:
        return now_utc.astimezone(_ny_tz()).date().isoformat()
    except Exception:
        return now_utc.date().isoformat()


def _intraday_dir() -> Path:
    """Resolve intraday artifact directory (supports DT_TRUTH_DIR override)."""
    override = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if override:
        base = Path(override) / "intraday"
        base.mkdir(parents=True, exist_ok=True)
        return base

    da = DT_PATHS.get("da_brains")
    base = da / "intraday" if isinstance(da, Path) else Path("da_brains") / "intraday"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _state_path() -> Path:
    p = _intraday_dir() / "risk" / "risk_rails_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _sum_equity_and_exposure(metrics: Dict[str, Any]) -> Dict[str, float]:
    bots = metrics.get("bots") if isinstance(metrics, dict) else None
    if not isinstance(bots, dict):
        return {"equity": 0.0, "pos_val": 0.0, "realized": 0.0, "positions": 0.0}

    equity = 0.0
    pos_val = 0.0
    realized = 0.0
    positions = 0.0
    for _, b in bots.items():
        if not isinstance(b, dict):
            continue
        equity += _safe_float(b.get("equity_est"), 0.0)
        pos_val += _safe_float(b.get("positions_value_est"), 0.0)
        realized += _safe_float(b.get("realized_pnl_est"), 0.0)
        positions += _safe_float(b.get("positions"), 0.0)
    return {
        "equity": float(equity),
        "pos_val": float(pos_val),
        "realized": float(realized),
        "positions": float(positions),
    }


def _read_state() -> Dict[str, Any]:
    try:
        p = _state_path()
        if not p.exists():
            return {}
        data = read_json(p, {})
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(st: Dict[str, Any]) -> None:
    try:
        atomic_write_json(_state_path(), st)
    except Exception:
        pass


def _broker_equity_snapshot(ttl_sec: int = 180) -> Tuple[Optional[float], Dict[str, Any]]:
    """Return (equity, raw_snapshot). equity=None if unavailable."""
    if BrokerAPI is None:
        return None, {"status": "unavailable"}
    try:
        b = BrokerAPI()
        # New in broker_api v2.3
        if hasattr(b, "get_account_cached"):
            snap = b.get_account_cached(ttl_sec=ttl_sec)  # type: ignore[attr-defined]
        else:
            snap = None
        if not isinstance(snap, dict) or not snap:
            return None, {"status": "empty"}
        eq = snap.get("equity")
        eq_f = _safe_float(eq, 0.0)
        if eq_f > 0:
            return float(eq_f), snap
        # Some accounts only expose portfolio_value
        pv_f = _safe_float(snap.get("portfolio_value"), 0.0)
        if pv_f > 0:
            return float(pv_f), snap
        return None, snap
    except Exception as e:
        return None, {"status": "error", "error": str(e)[:200]}


def _get_week_start_date(dt: datetime) -> str:
    """Get the Monday of the week containing dt (NY timezone)."""
    try:
        ny_time = dt.astimezone(_ny_tz())
        days_since_monday = ny_time.weekday()
        monday = ny_time - timedelta(days=days_since_monday)
        return monday.date().isoformat()
    except Exception:
        return dt.date().isoformat()


def _get_month_start_date(dt: datetime) -> str:
    """Get the first day of the month containing dt (NY timezone)."""
    try:
        ny_time = dt.astimezone(_ny_tz())
        return ny_time.replace(day=1).date().isoformat()
    except Exception:
        return dt.date().isoformat()


def assess_and_update_risk_rails(*, now_utc: Optional[datetime] = None, rolling: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate rails and return summary.

    This function may also update the persistent rail state.
    
    Args:
        now_utc: Current UTC timestamp (defaults to now)
        rolling: Rolling cache data for VIX level checking (optional)
    """
    now = now_utc or _utc_now()
    session_date = _session_date_str(now)
    week_start = _get_week_start_date(now)
    month_start = _get_month_start_date(now)

    daily_loss_limit = _env_float("DT_DAILY_LOSS_LIMIT_USD", 300.0)
    daily_dd_pct = _env_float("DT_DAILY_DRAWDOWN_PCT", 0.02)
    max_weekly_dd_pct = _env_float("DT_MAX_WEEKLY_DRAWDOWN_PCT", 8.0)
    max_monthly_dd_pct = _env_float("DT_MAX_MONTHLY_DRAWDOWN_PCT", 15.0)
    max_open_positions = _env_int("DT_MAX_OPEN_POSITIONS", 3)
    max_exposure_frac = _env_float("DT_MAX_EXPOSURE_FRAC", 0.55)
    cooldown_after_losses = _env_int("DT_COOLDOWN_AFTER_LOSS_DELTAS", 3)
    cooldown_minutes = _env_int("DT_COOLDOWN_MINUTES", 20)
    
    # VIX spike protection
    vix_threshold = _env_float("DT_VIX_SPIKE_THRESHOLD", 35.0)
    pause_on_vix_spike = _env_int("DT_PAUSE_ON_VIX_SPIKE", 1) != 0

    equity_source_pref = _env_str("DT_RISK_EQUITY_SOURCE", "auto").lower()  # auto|broker|ledger
    broker_ttl = _env_int("DT_RISK_BROKER_TTL_SEC", 180)

    # Local metrics (still useful for exposure / position heuristics)
    metrics = read_json(metrics_path(), {})
    msum = _sum_equity_and_exposure(metrics if isinstance(metrics, dict) else {})
    equity_ledger = float(msum["equity"])
    pos_val = float(msum["pos_val"])
    realized_total = float(msum["realized"])
    open_positions = int(msum["positions"])

    # Broker equity (preferred for daily kill-switches when available)
    broker_equity, broker_snap = _broker_equity_snapshot(ttl_sec=broker_ttl)

    use_broker = False
    if equity_source_pref == "broker":
        use_broker = broker_equity is not None
    elif equity_source_pref == "ledger":
        use_broker = False
    else:  # auto
        use_broker = broker_equity is not None

    equity = float(broker_equity) if use_broker and broker_equity is not None else float(equity_ledger)
    equity_source = "broker" if use_broker and broker_equity is not None else "ledger"

    exposure_frac = (pos_val / equity) if equity > 1e-9 else 0.0

    st = _read_state()
    reset_on_source_change = _env_int("DT_RISK_RESET_ON_SOURCE_CHANGE", 1) != 0

    if st.get("date") != session_date:
        # new day: reset daily, check if we need to reset weekly/monthly
        old_week_start = st.get("week_start", "")
        old_month_start = st.get("month_start", "")
        
        # Reset weekly tracking if new week
        if old_week_start != week_start:
            week_start_equity = equity
            week_peak_equity = equity
        else:
            week_start_equity = _safe_float(st.get("week_start_equity"), equity)
            week_peak_equity = max(_safe_float(st.get("week_peak_equity"), equity), equity)
        
        # Reset monthly tracking if new month
        if old_month_start != month_start:
            month_start_equity = equity
            month_peak_equity = equity
        else:
            month_start_equity = _safe_float(st.get("month_start_equity"), equity)
            month_peak_equity = max(_safe_float(st.get("month_peak_equity"), equity), equity)
        
        st = {
            "date": session_date,
            "week_start": week_start,
            "month_start": month_start,
            "equity_source": equity_source,
            "start_equity": equity,
            "peak_equity": equity,
            "week_start_equity": week_start_equity,
            "week_peak_equity": week_peak_equity,
            "month_start_equity": month_start_equity,
            "month_peak_equity": month_peak_equity,
            "last_realized": realized_total,
            "consec_loss_deltas": 0,
            "cooldown_until": "",
            "ts": now.isoformat().replace("+00:00", "Z"),
        }
    else:
        # Same day, but if the equity source changed (e.g., broker becomes available),
        # avoid mixing baselines which can generate massive false PnL.
        if reset_on_source_change and st.get("equity_source") and st.get("equity_source") != equity_source:
            st["equity_source"] = equity_source
            st["start_equity"] = equity
            st["peak_equity"] = equity
        
        # Check if week changed during the day (rare but possible)
        if st.get("week_start") != week_start:
            st["week_start"] = week_start
            st["week_start_equity"] = equity
            st["week_peak_equity"] = equity
        
        # Check if month changed during the day (rare but possible)
        if st.get("month_start") != month_start:
            st["month_start"] = month_start
            st["month_start_equity"] = equity
            st["month_peak_equity"] = equity

    start_equity = _safe_float(st.get("start_equity"), equity)
    peak_equity = max(_safe_float(st.get("peak_equity"), equity), equity)
    week_start_equity = _safe_float(st.get("week_start_equity"), equity)
    week_peak_equity = max(_safe_float(st.get("week_peak_equity"), equity), equity)
    month_start_equity = _safe_float(st.get("month_start_equity"), equity)
    month_peak_equity = max(_safe_float(st.get("month_peak_equity"), equity), equity)

    pnl_today = equity - start_equity
    drawdown_from_peak = equity - peak_equity
    dd_pct = (-drawdown_from_peak / peak_equity) if peak_equity > 1e-9 and drawdown_from_peak < 0 else 0.0
    
    # Weekly drawdown
    weekly_drawdown_from_peak = equity - week_peak_equity
    weekly_dd_pct = (-weekly_drawdown_from_peak / week_peak_equity) if week_peak_equity > 1e-9 and weekly_drawdown_from_peak < 0 else 0.0
    
    # Monthly drawdown
    monthly_drawdown_from_peak = equity - month_peak_equity
    monthly_dd_pct = (-monthly_drawdown_from_peak / month_peak_equity) if month_peak_equity > 1e-9 and monthly_drawdown_from_peak < 0 else 0.0
    
    # VIX level check
    vix_level = 0.0
    vix_spike = False
    if rolling and isinstance(rolling, dict):
        gdt = rolling.get("_GLOBAL_DT")
        if isinstance(gdt, dict):
            vix_level = _safe_float(gdt.get("vix_level"), 0.0)
            vix_spike = bool(gdt.get("vix_spike", False)) or (vix_level >= vix_threshold)

    # consecutive loss deltas: based on realized_pnl_est changes (ledger-only, by design)
    last_realized = _safe_float(st.get("last_realized"), realized_total)
    delta_realized = realized_total - last_realized
    consec = int(_safe_float(st.get("consec_loss_deltas"), 0))
    if delta_realized < 0:
        consec += 1
    elif delta_realized > 0:
        consec = 0

    # cooldown
    cooldown_until = ""
    cooldown_active = False
    try:
        cu = str(st.get("cooldown_until") or "").strip()
        if cu:
            dt = datetime.fromisoformat(cu.replace("Z", "+00:00"))
            cooldown_active = now < dt.astimezone(timezone.utc)
            cooldown_until = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        cooldown_active = False
        cooldown_until = ""

    # If we hit consecutive loss threshold, start cooldown
    if not cooldown_active and cooldown_after_losses > 0 and consec >= cooldown_after_losses:
        dt = now + timedelta(minutes=max(1, cooldown_minutes))
        cooldown_active = True
        cooldown_until = dt.isoformat().replace("+00:00", "Z")
        consec = 0  # reset after triggering

    # Position ownership safety: if our local strategy registry disagrees with broker reality,
    # stand down and force a reconcile. This prevents DT from accidentally selling Swing holdings.
    stand_down = False
    reason = ""
    
    try:
        reg = load_registry()
        mismatch = reg.get("mismatch") if isinstance(reg, dict) else {}
        if isinstance(mismatch, dict) and mismatch.get("has_mismatch") is True:
            st_m = mismatch.get("symbols")
            stand_down = True
            reason = f"position_registry_mismatch symbols={len(st_m) if isinstance(st_m, dict) else 0}"
    except Exception:
        pass

    # Determine stand_down
    if cooldown_active:
        stand_down = True
        reason = f"cooldown_until={cooldown_until}"

    # daily loss kill switch (absolute)
    if not stand_down and daily_loss_limit > 0 and pnl_today <= -abs(daily_loss_limit):
        stand_down = True
        reason = f"daily_loss_limit_hit pnl_today={pnl_today:.2f} source={equity_source}"

    # drawdown percent kill switch (daily)
    if not stand_down and daily_dd_pct > 0 and dd_pct >= abs(daily_dd_pct):
        stand_down = True
        reason = f"daily_drawdown_pct_hit dd={dd_pct:.4f} source={equity_source}"
    
    # weekly drawdown cap (NEW in v1.2)
    # Note: max_weekly_dd_pct is stored as percentage (e.g., 8.0), 
    # but weekly_dd_pct is calculated as decimal (e.g., 0.08)
    if not stand_down and max_weekly_dd_pct > 0 and weekly_dd_pct >= abs(max_weekly_dd_pct / 100.0):
        stand_down = True
        reason = f"weekly_drawdown_limit dd={weekly_dd_pct * 100:.2f}% threshold={max_weekly_dd_pct}%"
    
    # monthly drawdown cap (NEW in v1.2)
    # Note: Same unit conversion as weekly
    if not stand_down and max_monthly_dd_pct > 0 and monthly_dd_pct >= abs(max_monthly_dd_pct / 100.0):
        stand_down = True
        reason = f"monthly_drawdown_limit dd={monthly_dd_pct * 100:.2f}% threshold={max_monthly_dd_pct}%"
    
    # VIX spike protection (NEW in v1.2)
    if not stand_down and pause_on_vix_spike and vix_spike and vix_level >= vix_threshold:
        stand_down = True
        reason = f"vix_spike vix={vix_level:.2f} threshold={vix_threshold:.2f}"

    # max exposure (still ledger-estimated; conservative)
    if not stand_down and max_exposure_frac > 0 and exposure_frac >= max_exposure_frac:
        stand_down = True
        reason = f"max_exposure_hit exposure={exposure_frac:.3f}"

    # max positions
    if not stand_down and max_open_positions > 0 and open_positions > max_open_positions:
        stand_down = True
        reason = f"max_open_positions_hit open={open_positions}"

    st["date"] = session_date
    st["week_start"] = week_start
    st["month_start"] = month_start
    st["equity_source"] = equity_source
    st["start_equity"] = float(start_equity)
    st["peak_equity"] = float(max(peak_equity, equity))
    st["week_start_equity"] = float(week_start_equity)
    st["week_peak_equity"] = float(max(week_peak_equity, equity))
    st["month_start_equity"] = float(month_start_equity)
    st["month_peak_equity"] = float(max(month_peak_equity, equity))
    st["last_realized"] = float(realized_total)
    st["consec_loss_deltas"] = int(consec)
    st["cooldown_until"] = cooldown_until
    st["ts"] = now.isoformat().replace("+00:00", "Z")
    _write_state(st)

    summary = {
        "ts": st["ts"],
        "date": session_date,
        "week_start": week_start,
        "month_start": month_start,
        "stand_down": bool(stand_down),
        "reason": reason,

        # Equity/PnL (now *source-aware*)
        "equity_source": equity_source,
        "equity": float(equity),
        "start_equity": float(start_equity),
        "pnl_today": float(pnl_today),
        "peak_equity": float(st["peak_equity"]),
        "drawdown_from_peak": float(equity - float(st["peak_equity"])),
        "drawdown_pct_from_peak": float(dd_pct),
        
        # Weekly tracking (NEW in v1.2)
        "week_start_equity": float(week_start_equity),
        "week_peak_equity": float(st["week_peak_equity"]),
        "weekly_drawdown_from_peak": float(weekly_drawdown_from_peak),
        "weekly_drawdown_pct": float(weekly_dd_pct * 100),  # Convert to percentage
        
        # Monthly tracking (NEW in v1.2)
        "month_start_equity": float(month_start_equity),
        "month_peak_equity": float(st["month_peak_equity"]),
        "monthly_drawdown_from_peak": float(monthly_drawdown_from_peak),
        "monthly_drawdown_pct": float(monthly_dd_pct * 100),  # Convert to percentage
        
        # VIX tracking (NEW in v1.2)
        "vix_level": float(vix_level),
        "vix_spike": bool(vix_spike),

        # Exposure/position heuristics (ledger-estimated)
        "open_positions_est": int(open_positions),
        "positions_value_est": float(pos_val),
        "exposure_frac_est": float(exposure_frac),

        # Realized delta (ledger-estimated)
        "realized_pnl_est": float(realized_total),
        "delta_realized_est": float(delta_realized),

        # Cooldown
        "cooldown_until": cooldown_until,

        # Debug broker snapshot (small)
        "broker": {
            "available": bool(broker_equity is not None),
            "ttl_sec": int(broker_ttl),
            "status": broker_snap.get("status"),
            "equity": broker_snap.get("equity"),
            "portfolio_value": broker_snap.get("portfolio_value"),
            "last_equity": broker_snap.get("last_equity"),
            "timestamp": broker_snap.get("ts"),
        },
    }

    try:
        if stand_down:
            log(f"[risk_rails] 🛑 stand_down: {reason}")
    except Exception:
        pass

    return summary


if __name__ == "__main__":  # pragma: no cover
    print(assess_and_update_risk_rails())

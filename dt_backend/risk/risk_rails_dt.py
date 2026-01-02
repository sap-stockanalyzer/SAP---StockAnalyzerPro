# dt_backend/risk/risk_rails_dt.py — v1.0 (Phase 0)
"""Hard risk rails for dt_backend (Phase 0).

This is the "adult supervision" layer. It prevents the daytrading loop from
continuing to trade once predefined limits are violated.

Enforced constraints (env overrides)
----------------------------------
- DT_DAILY_LOSS_LIMIT_USD (default 300.0)
- DT_DAILY_DRAWDOWN_PCT (default 0.02 => 2%)
- DT_MAX_OPEN_POSITIONS (default 3)
- DT_MAX_EXPOSURE_FRAC (default 0.55)
- DT_COOLDOWN_AFTER_LOSS_DELTAS (default 3)
- DT_COOLDOWN_MINUTES (default 20)

Notes
-----
We don't assume a broker feed for exact equity. We use dt_metrics.json
(estimates based on local ledgers + last prices in rolling). This is meant
to be conservative and *trip early* rather than late.
"""

from __future__ import annotations

import os
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from dt_backend.core import DT_PATHS
from dt_backend.core.logger_dt import log
from dt_backend.services.dt_truth_store import metrics_path, read_json, atomic_write_json


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


def assess_and_update_risk_rails(*, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """Evaluate rails and return summary.

    This function may also update the persistent rail state.
    """
    now = now_utc or _utc_now()
    session_date = _session_date_str(now)

    daily_loss_limit = _env_float("DT_DAILY_LOSS_LIMIT_USD", 300.0)
    daily_dd_pct = _env_float("DT_DAILY_DRAWDOWN_PCT", 0.02)
    max_open_positions = _env_int("DT_MAX_OPEN_POSITIONS", 3)
    max_exposure_frac = _env_float("DT_MAX_EXPOSURE_FRAC", 0.55)
    cooldown_after_losses = _env_int("DT_COOLDOWN_AFTER_LOSS_DELTAS", 3)
    cooldown_minutes = _env_int("DT_COOLDOWN_MINUTES", 20)

    metrics = read_json(metrics_path(), {})
    msum = _sum_equity_and_exposure(metrics if isinstance(metrics, dict) else {})
    equity = float(msum["equity"])
    pos_val = float(msum["pos_val"])
    realized_total = float(msum["realized"])
    open_positions = int(msum["positions"])

    exposure_frac = (pos_val / equity) if equity > 1e-9 else 0.0

    st = _read_state()
    if st.get("date") != session_date:
        # new day: reset
        st = {
            "date": session_date,
            "start_equity": equity,
            "peak_equity": equity,
            "last_realized": realized_total,
            "consec_loss_deltas": 0,
            "cooldown_until": "",
            "ts": now.isoformat().replace("+00:00", "Z"),
        }

    start_equity = _safe_float(st.get("start_equity"), equity)
    peak_equity = max(_safe_float(st.get("peak_equity"), equity), equity)

    pnl_today = equity - start_equity
    drawdown_from_peak = equity - peak_equity
    dd_pct = (-drawdown_from_peak / peak_equity) if peak_equity > 1e-9 and drawdown_from_peak < 0 else 0.0

    # consecutive loss deltas: based on realized_pnl_est changes
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

    # Determine stand_down
    stand_down = False
    reason = ""

    if cooldown_active:
        stand_down = True
        reason = f"cooldown_until={cooldown_until}"

    # daily loss kill switch (absolute)
    if not stand_down and daily_loss_limit > 0 and pnl_today <= -abs(daily_loss_limit):
        stand_down = True
        reason = f"daily_loss_limit_hit pnl_today={pnl_today:.2f}"

    # drawdown percent kill switch
    if not stand_down and daily_dd_pct > 0 and dd_pct >= abs(daily_dd_pct):
        stand_down = True
        reason = f"daily_drawdown_pct_hit dd={dd_pct:.4f}"

    # max exposure
    if not stand_down and max_exposure_frac > 0 and exposure_frac >= max_exposure_frac:
        stand_down = True
        reason = f"max_exposure_hit exposure={exposure_frac:.3f}"

    # max positions
    if not stand_down and max_open_positions > 0 and open_positions > max_open_positions:
        stand_down = True
        reason = f"max_open_positions_hit open={open_positions}"

    st["date"] = session_date
    st["start_equity"] = float(start_equity)
    st["peak_equity"] = float(max(peak_equity, equity))
    st["last_realized"] = float(realized_total)
    st["consec_loss_deltas"] = int(consec)
    st["cooldown_until"] = cooldown_until
    st["ts"] = now.isoformat().replace("+00:00", "Z")
    _write_state(st)

    summary = {
        "ts": st["ts"],
        "date": session_date,
        "stand_down": bool(stand_down),
        "reason": reason,
        "equity_est": float(equity),
        "start_equity_est": float(start_equity),
        "pnl_today_est": float(pnl_today),
        "peak_equity_est": float(st["peak_equity"]),
        "drawdown_from_peak_est": float(equity - float(st["peak_equity"])),
        "drawdown_pct_from_peak": float(dd_pct),
        "open_positions_est": int(open_positions),
        "exposure_frac_est": float(exposure_frac),
        "delta_realized_est": float(delta_realized),
        "cooldown_until": cooldown_until,
    }

    try:
        if stand_down:
            log(f"[risk_rails] 🛑 stand_down: {reason}")
    except Exception:
        pass

    return summary


if __name__ == "__main__":  # pragma: no cover
    print(assess_and_update_risk_rails())

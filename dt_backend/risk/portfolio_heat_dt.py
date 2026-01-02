# dt_backend/risk/portfolio_heat_dt.py — v1.0 (Phase 5.5)
"""Portfolio heat manager (Phase 5.5).

Goal
----
Stop the system from taking multiple highly-correlated positions by using
simple, explainable caps.

This v1 implementation is intentionally conservative:
- Uses only whatever metadata exists in rolling (sector) and dt_metrics.json
- If sector is missing, bucket into "UNKNOWN"
- Applies:
  • max positions per sector
  • optional global exposure throttle (positions_value_est / equity_est)

Integration point
-----------------
Called from the policy engine after trade candidates are scored and
max_positions is enforced. It can turn trade_gate off and append a reason.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return int(float(raw)) if raw else int(default)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


def _sector_of_node(node: Dict[str, Any]) -> str:
    # Try a few common places. Keep it cheap.
    for key in ("sector", "Sector"):
        v = node.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    prof = node.get("profile")
    if isinstance(prof, dict):
        v = prof.get("sector") or prof.get("Sector")
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    ctx = node.get("context_dt")
    if isinstance(ctx, dict):
        v = ctx.get("sector")
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    return "UNKNOWN"


def apply_portfolio_heat_gates(
    rolling: Dict[str, Any],
    *,
    out_key: str = "policy_dt",
    max_per_sector: int | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply sector/exposure gates.

    Returns (rolling, summary).
    """
    if not isinstance(rolling, dict) or not rolling:
        return rolling, {"ok": False, "reason": "rolling_empty"}

    max_per_sector = max_per_sector if max_per_sector is not None else _env_int("DT_HEAT_MAX_PER_SECTOR", 2)
    max_exposure = _env_float("DT_HEAT_MAX_EXPOSURE_FRAC", 0.80)

    g = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
    metrics = g.get("dt_metrics") if isinstance(g.get("dt_metrics"), dict) else {}

    # Determine current exposure (best-effort)
    equity = 0.0
    pos_val = 0.0
    try:
        bots = metrics.get("bots") if isinstance(metrics, dict) else None
        if isinstance(bots, dict):
            for _, b in bots.items():
                if isinstance(b, dict):
                    equity += float(b.get("equity_est") or 0.0)
                    pos_val += float(b.get("positions_value_est") or 0.0)
    except Exception:
        equity = 0.0
        pos_val = 0.0

    exposure = (pos_val / equity) if equity > 0 else 0.0

    # Count open positions by sector (approximate).
    # We treat any existing OPEN positions as "occupied slots". If you want
    # a tighter mapping to actual portfolio, extend this to read dt_positions.json.
    sector_open: Dict[str, int] = {}
    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        pos = node.get("position_dt")
        # position_dt is optional; fallback: if last execution was BUY/SELL and size>0 assume open.
        is_open = False
        if isinstance(pos, dict):
            is_open = str(pos.get("status") or "").upper() in {"OPEN", "PARTIAL"}
        if not is_open:
            ex = node.get("execution_dt")
            if isinstance(ex, dict):
                is_open = str(ex.get("side") or "").upper() in {"BUY", "SELL"} and float(ex.get("size") or 0.0) > 0
        if not is_open:
            continue

        sec = _sector_of_node(node)
        sector_open[sec] = sector_open.get(sec, 0) + 1

    throttled = 0
    sector_blocked: Dict[str, int] = {}

    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        p = node.get(out_key)
        if not isinstance(p, dict):
            continue
        if not bool(p.get("trade_gate")):
            continue

        sec = _sector_of_node(node)

        # Sector cap
        if max_per_sector is not None and sector_open.get(sec, 0) >= max_per_sector:
            p["trade_gate"] = False
            prev = str(p.get("reason") or "").strip()
            suffix = f"heat=sector_cap({sec}:{max_per_sector})"
            p["reason"] = (prev + "; " + suffix).strip("; ") if prev else suffix
            p["reason_code"] = "PORTFOLIO_HEAT_SECTOR"
            throttled += 1
            sector_blocked[sec] = sector_blocked.get(sec, 0) + 1
            node[out_key] = p
            rolling[sym] = node
            continue

        # Exposure throttle (global)
        if exposure >= max_exposure:
            p["trade_gate"] = False
            prev = str(p.get("reason") or "").strip()
            suffix = f"heat=exposure({exposure:.2f}>{max_exposure:.2f})"
            p["reason"] = (prev + "; " + suffix).strip("; ") if prev else suffix
            p["reason_code"] = "PORTFOLIO_HEAT_EXPOSURE"
            throttled += 1
            node[out_key] = p
            rolling[sym] = node
            continue

    summary = {
        "ok": True,
        "throttled": throttled,
        "sector_open": sector_open,
        "sector_blocked": sector_blocked,
        "exposure": float(exposure),
        "max_exposure": float(max_exposure),
        "max_per_sector": int(max_per_sector) if max_per_sector is not None else None,
    }

    # mirror into global for visibility
    try:
        g = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
        g["heat_dt"] = summary
        rolling["_GLOBAL_DT"] = g
    except Exception:
        pass

    return rolling, summary

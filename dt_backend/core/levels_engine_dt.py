"""dt_backend/core/levels_engine_dt.py — v1.0 (Phase 2.5)

Levels engine (human-esque context).

We compute a handful of intraday-relevant reference levels per symbol:
  • Premarket high/low
  • Opening Range (OR) 5m / 15m
  • Session VWAP (from features_dt when available)
  • Prior day high/low/close (best-effort if daily bars exist)

This module is best-effort and should never raise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _ny_tz():
    if ZoneInfo is not None:
        return ZoneInfo("America/New_York")
    return timezone.utc


def _parse_iso(ts_raw: Any) -> Optional[datetime]:
    if not ts_raw:
        return None
    try:
        s = str(ts_raw).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _bars(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    b = node.get("bars_intraday") or []
    return b if isinstance(b, list) else []


def _bars_5m(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    b = node.get("bars_intraday_5m") or []
    return b if isinstance(b, list) else []


def _to_ohlc(bars: List[Dict[str, Any]]) -> Tuple[List[datetime], List[float], List[float]]:
    ts: List[datetime] = []
    highs: List[float] = []
    lows: List[float] = []
    for raw in bars:
        if not isinstance(raw, dict):
            continue
        dt = _parse_iso(raw.get("ts") or raw.get("t"))
        if dt is None:
            continue
        try:
            h = float(raw.get("h"))
            l = float(raw.get("l"))
        except Exception:
            continue
        ts.append(dt)
        highs.append(h)
        lows.append(l)
    return ts, highs, lows


def _opening_range(ts: List[datetime], highs: List[float], lows: List[float], *, minutes: int, now_utc: datetime) -> Tuple[float, float]:
    if not ts:
        return 0.0, 0.0
    ny = _ny_tz()
    now_ny = now_utc.astimezone(ny)
    open_ny = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    end_ny = open_ny.replace(minute=open_ny.minute + int(minutes))
    open_utc = open_ny.astimezone(timezone.utc)
    end_utc = end_ny.astimezone(timezone.utc)

    or_h: Optional[float] = None
    or_l: Optional[float] = None
    for i, t in enumerate(ts):
        if t < open_utc or t >= end_utc:
            continue
        or_h = highs[i] if or_h is None else max(or_h, highs[i])
        or_l = lows[i] if or_l is None else min(or_l, lows[i])
    return float(or_h or 0.0), float(or_l or 0.0)


def _premarket_hilo(ts: List[datetime], highs: List[float], lows: List[float], *, now_utc: datetime) -> Tuple[float, float]:
    if not ts:
        return 0.0, 0.0
    ny = _ny_tz()
    now_ny = now_utc.astimezone(ny)
    open_ny = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    open_utc = open_ny.astimezone(timezone.utc)

    ph: Optional[float] = None
    pl: Optional[float] = None
    # Premarket = any bars earlier than open, same NY calendar day
    day = now_ny.date()
    for i, t in enumerate(ts):
        t_ny = t.astimezone(ny)
        if t_ny.date() != day:
            continue
        if t >= open_utc:
            continue
        ph = highs[i] if ph is None else max(ph, highs[i])
        pl = lows[i] if pl is None else min(pl, lows[i])
    return float(ph or 0.0), float(pl or 0.0)


def compute_levels_for_symbol(sym: str, node: Dict[str, Any], *, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """Best-effort levels extraction for one symbol."""
    now_utc = now_utc or datetime.now(timezone.utc)

    # Prefer 1m for OR precision, otherwise 5m.
    b1 = _bars(node)
    b5 = _bars_5m(node)
    use = b1 if len(b1) >= 30 else b5

    ts, highs, lows = _to_ohlc(use)

    pm_h, pm_l = _premarket_hilo(ts, highs, lows, now_utc=now_utc)
    or5_h, or5_l = _opening_range(ts, highs, lows, minutes=5, now_utc=now_utc)
    or15_h, or15_l = _opening_range(ts, highs, lows, minutes=15, now_utc=now_utc)

    # VWAP (prefer features_dt)
    vwap = 0.0
    try:
        feat = node.get("features_dt") or {}
        if isinstance(feat, dict):
            vwap = float(feat.get("vwap") or 0.0)
    except Exception:
        vwap = 0.0

    # Prior day levels (best-effort if daily bars exist)
    prior_h = prior_l = prior_c = 0.0
    daily = node.get("bars_daily") or node.get("daily_bars") or []
    if isinstance(daily, list) and daily:
        last = daily[-1]
        if isinstance(last, dict):
            try:
                prior_h = float(last.get("h") or last.get("high") or 0.0)
                prior_l = float(last.get("l") or last.get("low") or 0.0)
                prior_c = float(last.get("c") or last.get("close") or 0.0)
            except Exception:
                pass

    return {
        "ts": now_utc.isoformat().replace("+00:00", "Z"),
        "premarket_high": float(pm_h),
        "premarket_low": float(pm_l),
        "prior_high": float(prior_h),
        "prior_low": float(prior_l),
        "prior_close": float(prior_c),
        "or5_high": float(or5_h),
        "or5_low": float(or5_l),
        "or15_high": float(or15_h),
        "or15_low": float(or15_l),
        "vwap": float(vwap),
    }


def update_levels_in_rolling(rolling: Dict[str, Any], *, max_symbols: int = 300) -> Dict[str, Any]:
    """Compute levels_dt for many symbols and return stats."""
    if not isinstance(rolling, dict) or not rolling:
        return {"symbols": 0, "updated": 0}

    now_utc = datetime.now(timezone.utc)
    syms = sorted([s for s in rolling.keys() if isinstance(s, str) and not s.startswith("_")])
    if max_symbols > 0:
        syms = syms[: int(max_symbols)]

    updated = 0
    for sym in syms:
        node = rolling.get(sym)
        if not isinstance(node, dict):
            continue
        node["levels_dt"] = compute_levels_for_symbol(sym, node, now_utc=now_utc)
        rolling[sym] = node
        updated += 1
    return {"symbols": len(syms), "updated": int(updated)}

"""dt_backend/core/levels_engine_dt.py — v1.1 (Phase 2.5)

Levels engine (human-esque context).

We compute a handful of intraday-relevant reference levels per symbol:
  • Premarket high/low
  • Opening Range (OR) 5m / 15m (best-effort, uses 1m if available)
  • Session VWAP (from features_dt when available)
  • Prior day high/low/close (best-effort, avoids "today partial bar" traps)

This module is best-effort and should never raise.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
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


def _bars(node: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    b = node.get(key) or []
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

    if len(ts) >= 2:
        idx = sorted(range(len(ts)), key=lambda i: ts[i])
        ts = [ts[i] for i in idx]
        highs = [highs[i] for i in idx]
        lows = [lows[i] for i in idx]

    return ts, highs, lows


def _market_open_close_utc(now_utc: datetime) -> Tuple[datetime, datetime]:
    ny = _ny_tz()
    now_ny = now_utc.astimezone(ny)
    open_ny = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_ny = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_ny.astimezone(timezone.utc), close_ny.astimezone(timezone.utc)


def _opening_range(ts, highs, lows, *, minutes: int, now_utc: datetime) -> Tuple[float, float]:
    if not ts:
        return 0.0, 0.0

    open_utc, _ = _market_open_close_utc(now_utc)
    end_utc = open_utc + timedelta(minutes=max(1, int(minutes)))

    or_h = None
    or_l = None
    for i, t in enumerate(ts):
        if t < open_utc or t >= end_utc:
            continue
        or_h = highs[i] if or_h is None else max(or_h, highs[i])
        or_l = lows[i] if or_l is None else min(or_l, lows[i])

    return float(or_h or 0.0), float(or_l or 0.0)


def _premarket_hilo(ts, highs, lows, *, now_utc: datetime) -> Tuple[float, float]:
    if not ts:
        return 0.0, 0.0

    ny = _ny_tz()
    now_ny = now_utc.astimezone(ny)
    day = now_ny.date()
    open_utc, _ = _market_open_close_utc(now_utc)

    ph = None
    pl = None
    for i, t in enumerate(ts):
        try:
            t_ny = t.astimezone(ny)
        except Exception:
            continue
        if t_ny.date() != day or t >= open_utc:
            continue
        ph = highs[i] if ph is None else max(ph, highs[i])
        pl = lows[i] if pl is None else min(pl, lows[i])

    return float(ph or 0.0), float(pl or 0.0)


def _prior_day_from_daily(node: Dict[str, Any], *, now_utc: datetime) -> Tuple[float, float, float]:
    ny = _ny_tz()
    today_ny = now_utc.astimezone(ny).date()

    daily = node.get("bars_daily") or node.get("daily_bars") or []
    if not isinstance(daily, list):
        return 0.0, 0.0, 0.0

    best_dt = None
    best_h = best_l = best_c = 0.0

    for raw in daily:
        if not isinstance(raw, dict):
            continue
        dt = _parse_iso(raw.get("ts") or raw.get("t") or raw.get("date"))
        if dt is None:
            continue
        if dt.astimezone(ny).date() >= today_ny:
            continue
        try:
            h = float(raw.get("h") or raw.get("high") or 0.0)
            l = float(raw.get("l") or raw.get("low") or 0.0)
            c = float(raw.get("c") or raw.get("close") or 0.0)
        except Exception:
            continue
        if best_dt is None or dt > best_dt:
            best_dt = dt
            best_h, best_l, best_c = h, l, c

    return float(best_h), float(best_l), float(best_c)


def compute_levels_for_symbol(sym: str, node: Dict[str, Any], *, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    now_utc = now_utc or datetime.now(timezone.utc)

    b1 = _bars(node, "bars_intraday")
    b5 = _bars(node, "bars_intraday_5m")
    use = b1 if len(b1) >= 5 else b5

    ts, highs, lows = _to_ohlc(use)

    pm_h, pm_l = _premarket_hilo(ts, highs, lows, now_utc=now_utc)
    or5_h, or5_l = _opening_range(ts, highs, lows, minutes=5, now_utc=now_utc)
    or15_h, or15_l = _opening_range(ts, highs, lows, minutes=15, now_utc=now_utc)

    vwap = 0.0
    feat = node.get("features_dt") or {}
    if isinstance(feat, dict):
        try:
            vwap = float(feat.get("vwap") or 0.0)
        except Exception:
            vwap = 0.0

    prior_h, prior_l, prior_c = _prior_day_from_daily(node, now_utc=now_utc)

    return {
        "ts": now_utc.isoformat().replace("+00:00", "Z"),
        "premarket_high": pm_h,
        "premarket_low": pm_l,
        "prior_high": prior_h,
        "prior_low": prior_l,
        "prior_close": prior_c,
        "or5_high": or5_h,
        "or5_low": or5_l,
        "or15_high": or15_h,
        "or15_low": or15_l,
        "vwap": vwap,
    }


def update_levels_in_rolling(rolling: Dict[str, Any], *, max_symbols: int = 300) -> Dict[str, Any]:
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
        try:
            node["levels_dt"] = compute_levels_for_symbol(sym, node, now_utc=now_utc)
            rolling[sym] = node
            updated += 1
        except Exception:
            continue

    return {"symbols": len(syms), "updated": int(updated)}

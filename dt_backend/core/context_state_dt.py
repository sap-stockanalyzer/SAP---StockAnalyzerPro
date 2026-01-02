# dt_backend/core/context_state_dt.py — v1.1
"""
Builds intraday context features for each symbol in the rolling cache.

Writes:
    rolling[sym]["context_dt"] = {
        intraday_return, intraday_range, intraday_vol, last_price,
        intraday_trend, vol_bucket, has_intraday_data, ts
    }

Best-effort, safe defaults. No heavy deps.
"""

from __future__ import annotations

import math
from datetime import datetime, date, timezone, timezone
from typing import Any, Dict, List, Optional

from .data_pipeline_dt import _read_rolling, save_rolling, log, ensure_symbol_node

try:
    # Prefer NY trading-day boundary instead of local machine timezone.
    from utils.time_utils import now_ny  # type: ignore
except Exception:  # pragma: no cover
    now_ny = None  # type: ignore


def _parse_ts(ts_raw: Any) -> Optional[datetime]:
    if ts_raw is None:
        return None
    try:
        if isinstance(ts_raw, (int, float)):
            return datetime.utcfromtimestamp(float(ts_raw))
        txt = str(ts_raw)
        try:
            return datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except Exception:
            return datetime.strptime(txt.split(".")[0], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _extract_today_bars(node: Dict[str, Any], *, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
    src = node.get("bars_intraday") or []
    if not isinstance(src, list):
        return []

    # In live mode, default to today's NY trading date.
    # In replay/backtest mode, callers pass target_date explicitly.
    if target_date is None:
        target_date = now_ny().date() if callable(now_ny) else date.today()

    out: List[Dict[str, Any]] = []

    for raw_bar in src:
        if not isinstance(raw_bar, dict):
            continue

        ts = _parse_ts(raw_bar.get("ts") or raw_bar.get("t") or raw_bar.get("timestamp"))
        if ts is None or ts.date() != target_date:
            continue

        price = raw_bar.get("c") if raw_bar.get("c") is not None else raw_bar.get("close", raw_bar.get("price"))
        price_f = _safe_float(price)
        if price_f is None:
            continue

        out.append({"ts": ts, "price": price_f})

    out.sort(key=lambda b: b["ts"])
    return out


def _pct(a: float, b: float) -> float:
    try:
        if b == 0.0 or not (math.isfinite(a) and math.isfinite(b)):
            return 0.0
        return (a / b) - 1.0
    except Exception:
        return 0.0


def _intraday_stats(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(bars) < 3:
        return {}

    prices = [b["price"] for b in bars]
    first = prices[0]
    last = prices[-1]
    high = max(prices)
    low = min(prices)

    intraday_return = _pct(last, first)
    intraday_range = _pct(high, low) if low > 0 else 0.0

    rets: List[float] = []
    for i in range(1, len(prices)):
        rets.append(_pct(prices[i], prices[i - 1]))

    if len(rets) >= 2:
        mu = sum(rets) / len(rets)
        var = sum((x - mu) ** 2 for x in rets) / max(len(rets) - 1, 1)
        intraday_vol = math.sqrt(max(var, 0.0))
    else:
        intraday_vol = 0.0

    return {
        "intraday_return": float(intraday_return),
        "intraday_range": float(intraday_range),
        "intraday_vol": float(intraday_vol),
        "last_price": float(last),
    }


def _trend_label(r: float, strong: float = 0.01, mild: float = 0.003) -> str:
    if r >= strong:
        return "strong_bull"
    if r >= mild:
        return "bull"
    if r <= -strong:
        return "strong_bear"
    if r <= -mild:
        return "bear"
    return "flat"


def _vol_bucket(vol: float) -> str:
    if vol >= 0.02:
        return "high"
    if vol >= 0.007:
        return "medium"
    return "low"


def build_intraday_context(*, target_date: Optional[date | str] = None, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    rolling = _read_rolling()
    if not rolling:
        log("[context_dt] ⚠️ rolling empty.")
        return {"symbols": 0, "updated": 0}

    updated = 0
    for sym in list(rolling.keys()):
        if str(sym).startswith("_"):
            continue

        node = ensure_symbol_node(rolling, sym)
        # Normalize target_date (replay/backtest may pass YYYY-MM-DD)
        td = target_date
        if isinstance(td, str):
            try:
                td = datetime.fromisoformat(td).date()
            except Exception:
                td = None
        bars_today = _extract_today_bars(node, target_date=td if isinstance(td, date) else None)
        stats = _intraday_stats(bars_today)
        if not stats:
            continue

        ctx = node.get("context_dt") or {}
        if not isinstance(ctx, dict):
            ctx = {}

        ctx.update(stats)
        ctx["intraday_trend"] = _trend_label(stats["intraday_return"])
        ctx["vol_bucket"] = _vol_bucket(stats["intraday_vol"])
        ctx["has_intraday_data"] = True
        ctx["ts"] = (now_utc or datetime.now(timezone.utc)).isoformat(timespec="seconds").replace("+00:00", "Z")

        node["context_dt"] = ctx
        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"[context_dt] ✅ updated {updated} symbols.")
    return {"symbols": len(rolling), "updated": updated}
# dt_backend/core/context_state_dt.py — v1.2 (CANDIDATE UNIVERSE + SAFE)
"""
Builds intraday context features for each symbol in the rolling cache.

Writes:
    rolling[sym]["context_dt"] = {
        intraday_return, intraday_range, intraday_vol, last_price,
        intraday_trend, vol_bucket, has_intraday_data, ts
    }

Also writes (Phase 1+ / fast-lane support):
    rolling["_GLOBAL_DT"]["candidate_universe_dt"] = {
        "symbols": [...],
        "n": int,
        "ts": ISO8601 UTC,
        "method": "context_score_v1",
        "weights": {...},
        "notes": str
    }

Best-effort, safe defaults. No heavy deps.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple

from .data_pipeline_dt import _read_rolling, save_rolling, log, ensure_symbol_node

try:
    from utils.time_utils import now_ny  # type: ignore
except Exception:  # pragma: no cover
    now_ny = None  # type: ignore


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


def _utc_now_iso(now_utc: Optional[datetime] = None) -> str:
    return (now_utc or datetime.now(timezone.utc)).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_ts(ts_raw: Any) -> Optional[datetime]:
    if ts_raw is None:
        return None
    try:
        if isinstance(ts_raw, (int, float)):
            return datetime.utcfromtimestamp(float(ts_raw)).replace(tzinfo=timezone.utc)
        txt = str(ts_raw).strip()
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
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


def _pct(a: float, b: float) -> float:
    try:
        if b == 0.0 or not (math.isfinite(a) and math.isfinite(b)):
            return 0.0
        return (a / b) - 1.0
    except Exception:
        return 0.0


def _extract_today_bars(node: Dict[str, Any], *, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
    src = node.get("bars_intraday") or []
    if not isinstance(src, list):
        return []

    if target_date is None:
        target_date = now_ny().date() if callable(now_ny) else date.today()

    out: List[Dict[str, Any]] = []
    for raw in src:
        if not isinstance(raw, dict):
            continue
        ts = _parse_ts(raw.get("ts") or raw.get("t") or raw.get("timestamp"))
        if ts is None or ts.date() != target_date:
            continue
        price = raw.get("c") if raw.get("c") is not None else raw.get("close", raw.get("price"))
        price_f = _safe_float(price)
        if price_f is None:
            continue
        out.append({"ts": ts, "price": float(price_f)})

    out.sort(key=lambda b: b["ts"])
    return out


def _intraday_stats(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(bars) < 3:
        return {}

    prices = [b["price"] for b in bars]
    first, last = prices[0], prices[-1]
    high, low = max(prices), min(prices)

    intraday_return = _pct(last, first)
    intraday_range = _pct(high, low) if low > 0 else 0.0

    rets = [_pct(prices[i], prices[i - 1]) for i in range(1, len(prices))]
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


def _candidate_score(ctx: Dict[str, Any], *, w_ret: float, w_vol: float, w_rng: float) -> float:
    r = float(ctx.get("intraday_return") or 0.0)
    v = float(ctx.get("intraday_vol") or 0.0)
    rg = float(ctx.get("intraday_range") or 0.0)
    return (w_ret * abs(r)) + (w_vol * v) + (w_rng * abs(rg))


def _write_candidate_universe(rolling: Dict[str, Any], scored: List[Tuple[str, float]], *, now_utc: Optional[datetime] = None) -> None:
    n = max(50, min(5000, _env_int("DT_CANDIDATE_UNIVERSE_N", 800)))

    w_ret = _env_float("DT_CAND_W_RET", 0.55)
    w_vol = _env_float("DT_CAND_W_VOL", 0.30)
    w_rng = _env_float("DT_CAND_W_RANGE", 0.15)

    rescored = []
    for sym, _ in scored:
        node = rolling.get(sym)
        if not isinstance(node, dict):
            continue
        ctx = node.get("context_dt")
        if not isinstance(ctx, dict):
            continue
        rescored.append((sym, _candidate_score(ctx, w_ret=w_ret, w_vol=w_vol, w_rng=w_rng)))

    rescored.sort(key=lambda t: t[1], reverse=True)
    syms = [s for s, _ in rescored[:n]]
    
    # NEW: Filter out recently-traded symbols to encourage rotation
    try:
        rotation_enabled = str(os.getenv("DT_UNIVERSE_ROTATION", "1") or "1").strip().lower() in {"1", "true", "yes", "y"}
        
        if rotation_enabled:
            exclusion_set = set()
            
            # Exclude currently open positions
            try:
                from dt_backend.services.position_manager_dt import read_positions_state
                pos_state = read_positions_state()
                open_syms = [s for s, ps in pos_state.items() 
                            if isinstance(ps, dict) and ps.get("status") == "OPEN"]
                exclusion_set.update(open_syms)
            except Exception:
                pass
            
            # Exclude recently exited (cooldown period)
            cooldown_hours = int(os.getenv("DT_UNIVERSE_COOLDOWN_HOURS", "2") or "2")
            if cooldown_hours > 0:
                try:
                    from dt_backend.services.position_manager_dt import read_positions_state
                    from dt_backend.core.time_override_dt import now_utc as get_now_utc
                    from datetime import datetime, timedelta
                    
                    pos_state = read_positions_state()
                    cutoff = (now_utc or get_now_utc()) - timedelta(hours=cooldown_hours)
                    
                    for sym, ps in pos_state.items():
                        if not isinstance(ps, dict):
                            continue
                        last_exit = ps.get("last_exit_ts")
                        if last_exit:
                            try:
                                ts = datetime.fromisoformat(str(last_exit).replace("Z", "+00:00"))
                                if ts > cutoff:
                                    exclusion_set.add(sym)
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # Filter candidates
            original_count = len(syms)
            syms = [s for s in syms if s not in exclusion_set]
            filtered_count = original_count - len(syms)
            
            if filtered_count > 0:
                log(f"[context_dt] 🔄 Filtered {filtered_count} recently-traded symbols for universe rotation")
                
    except Exception as e:
        log(f"[context_dt] ⚠️ universe rotation filter failed: {e}")

    g = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
    g["candidate_universe_dt"] = {
        "symbols": syms,
        "n": int(len(syms)),
        "ts": _utc_now_iso(now_utc),
        "method": "context_score_v1",
        "weights": {"w_ret": w_ret, "w_vol": w_vol, "w_range": w_rng},
        "notes": "Ranked by abs(intraday_return), intraday_vol, intraday_range.",
    }
    rolling["_GLOBAL_DT"] = g


def _fetch_vix_level() -> float:
    """Fetch current VIX level from market data.
    
    Returns 0.0 if unavailable (safe default).
    """
    try:
        # Try to get VIX from broker API or rolling cache
        # For now, we'll try to read from rolling cache first
        # If broker_api is available, we could fetch it there
        
        # Attempt 1: Check if VIX is in rolling cache
        rolling = _read_rolling()
        if rolling and isinstance(rolling, dict):
            vix_node = rolling.get("VIX") or rolling.get("^VIX")
            if isinstance(vix_node, dict):
                # Try to get last price from intraday bars
                bars = _extract_today_bars(vix_node)
                if bars and len(bars) > 0:
                    return float(bars[-1].get("price", 0.0))
                
                # Fallback to stored context
                ctx = vix_node.get("context_dt")
                if isinstance(ctx, dict):
                    last_price = ctx.get("last_price")
                    if last_price:
                        return float(last_price)
        
        # Attempt 2: Use broker API if available
        try:
            from dt_backend.engines.broker_api import BrokerAPI  # type: ignore
            b = BrokerAPI()
            if hasattr(b, "get_latest_quote"):
                quote = b.get_latest_quote("VIX")
                if isinstance(quote, dict):
                    price = quote.get("price") or quote.get("last") or quote.get("close")
                    if price:
                        return float(price)
        except Exception:
            pass
        
        # Default: return 0.0 (safe default, won't trigger spike)
        return 0.0
    except Exception:
        return 0.0


def build_intraday_context(
    *,
    target_date: Optional[date | str] = None,
    now_utc: Optional[datetime] = None,
    symbols: Optional[List[str]] = None,
    max_symbols: Optional[int] = None,
    build_candidates: bool = True,
) -> Dict[str, Any]:
    rolling = _read_rolling()
    if not rolling:
        log("[context_dt] ⚠️ rolling empty.")
        return {"symbols": 0, "updated": 0}

    if isinstance(target_date, str):
        try:
            target_date = datetime.fromisoformat(target_date).date()
        except Exception:
            target_date = None

    keys = [k for k in rolling.keys() if isinstance(k, str) and not k.startswith("_")]
    keys.sort()

    if symbols:
        wanted = {str(s).upper() for s in symbols}
        keys = [k for k in keys if k.upper() in wanted]

    if max_symbols is not None:
        keys = keys[: max(0, int(max_symbols))]

    now_utc = now_utc or datetime.now(timezone.utc)

    updated = 0
    scored: List[Tuple[str, float]] = []

    w_ret = _env_float("DT_CAND_W_RET", 0.55)
    w_vol = _env_float("DT_CAND_W_VOL", 0.30)
    w_rng = _env_float("DT_CAND_W_RANGE", 0.15)

    for sym in keys:
        node = ensure_symbol_node(rolling, sym)
        bars = _extract_today_bars(node, target_date=target_date)
        stats = _intraday_stats(bars)
        if not stats:
            continue

        ctx = node.get("context_dt") if isinstance(node.get("context_dt"), dict) else {}
        ctx.update(stats)
        ctx["intraday_trend"] = _trend_label(stats["intraday_return"])
        ctx["vol_bucket"] = _vol_bucket(stats["intraday_vol"])
        ctx["has_intraday_data"] = True
        ctx["ts"] = _utc_now_iso(now_utc)

        ctx["_cand_score"] = _candidate_score(ctx, w_ret=w_ret, w_vol=w_vol, w_rng=w_rng)

        node["context_dt"] = ctx
        rolling[sym] = node
        updated += 1

        if build_candidates and stats.get("last_price", 0.0) > 0.5:
            scored.append((sym.upper(), float(ctx.get("_cand_score") or 0.0)))

    if build_candidates:
        _write_candidate_universe(rolling, scored, now_utc=now_utc)
    
    # Add VIX level to global context (NEW in Phase 3)
    vix_level = _fetch_vix_level()
    vix_threshold = _env_float("DT_VIX_SPIKE_THRESHOLD", 35.0)
    gdt = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
    gdt["vix_level"] = float(vix_level)
    gdt["vix_spike"] = bool(vix_level >= vix_threshold)
    gdt["vix_threshold"] = float(vix_threshold)
    gdt["vix_ts"] = _utc_now_iso(now_utc)
    rolling["_GLOBAL_DT"] = gdt

    save_rolling(rolling)
    log(f"[context_dt] ✅ updated {updated} symbols. VIX={vix_level:.2f}")
    return {"symbols": len(keys), "updated": updated, "vix_level": vix_level}

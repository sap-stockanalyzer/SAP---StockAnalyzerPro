# dt_backend/engines/feature_engineering.py — v3.1
"""Intraday feature engineering for AION dt_backend.

Phase 1: "the bot's senses"
---------------------------
This module turns raw intraday bars into a compact feature snapshot per symbol.

Writes:
    rolling[sym]["features_dt"] = {...}

Feature goals
-------------
• One consistent view for all bots (same fields, same units)
• Fast + safe (best-effort, never crashes the loop)
• Uses 5Min bars by default for stability, but can fall back to 1Min

Configured by env:
    DT_FEATURE_TF = "5Min" (default) or "1Min"
    DT_FEATURES_MIN_INTERVAL = seconds (optional) skip recompute if too recent
    DT_MARKET_PROXIES = "SPY,QQQ" (default)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from dt_backend.core.data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node, log
from dt_backend.engines.indicators import (
    atr,
    bollinger_width,
    ema,
    keltner_width,
    lin_slope,
    pct_change,
    realized_vol,
    rsi,
    sma,
    stddev,
)


def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw != "" else float(default)
    except Exception:
        return float(default)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_ts(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        txt = str(s).strip()
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _ny_tz():
    if ZoneInfo is not None:
        return ZoneInfo("America/New_York")
    return timezone.utc


def _session_bounds_utc(now_utc: datetime) -> Tuple[datetime, datetime]:
    """Return today's NY session open/close in UTC (09:30–16:00 NY).

    Note: no holiday calendar here; this is purely intraday feature slicing.
    """
    ny = _ny_tz()
    now_ny = now_utc.astimezone(ny)
    open_ny = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_ny = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_ny.astimezone(timezone.utc), close_ny.astimezone(timezone.utc)


def _extract_bars(node: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    bars = node.get(key) or []
    return bars if isinstance(bars, list) else []


def _ohlcv_from_bars(bars: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[datetime]]:
    o: List[float] = []
    h: List[float] = []
    l: List[float] = []
    c: List[float] = []
    v: List[float] = []
    vw: List[float] = []
    ts: List[datetime] = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        dt = _parse_iso_ts(b.get("ts") or b.get("t"))
        if dt is None:
            continue
        try:
            oo = float(b.get("o"))
            hh = float(b.get("h"))
            ll = float(b.get("l"))
            cc = float(b.get("c"))
        except Exception:
            continue
        try:
            vv = float(b.get("v") or 0.0)
        except Exception:
            vv = 0.0
        try:
            vww = float(b.get("vw")) if b.get("vw") is not None else float("nan")
        except Exception:
            vww = float("nan")

        ts.append(dt)
        o.append(oo)
        h.append(hh)
        l.append(ll)
        c.append(cc)
        v.append(vv)
        vw.append(vww)
    return o, h, l, c, v, vw, ts


def _session_vwap_series(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> List[float]:
    """Session VWAP series.

    We compute cumulative VWAP using typical price (H+L+C)/3.
    This is more useful than per-bar vwap for 'reversion to VWAP' logic.
    """

    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n <= 0:
        return []
    out: List[float] = []
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        v = float(volumes[i] or 0.0)
        tp = (float(highs[i]) + float(lows[i]) + float(closes[i])) / 3.0
        if v > 0:
            cum_pv += tp * v
            cum_v += v
        out.append((cum_pv / cum_v) if cum_v > 0 else tp)
    return out


def _opening_range(ts: List[datetime], highs: List[float], lows: List[float], minutes: int, *, open_utc: datetime) -> Tuple[float, float]:
    if not ts or not highs or not lows:
        return 0.0, 0.0
    end_utc = open_utc + timedelta(minutes=max(1, int(minutes)))

    or_high = None
    or_low = None
    for i, t in enumerate(ts):
        if t < open_utc or t >= end_utc:
            continue
        hh = highs[i]
        ll = lows[i]
        or_high = hh if or_high is None else max(or_high, hh)
        or_low = ll if or_low is None else min(or_low, ll)

    return float(or_high or 0.0), float(or_low or 0.0)


def _trend_structure(closes: List[float], highs: List[float], lows: List[float]) -> Dict[str, float]:
    if len(closes) < 25:
        return {"ma_slope": 0.0, "hh": 0.0, "hl": 0.0, "trend_score": 0.0}

    sma_20 = sma(closes, 20)
    sma_20_prev = sma(closes[:-1], 20) if len(closes) > 21 else sma_20
    ma_slope = pct_change(sma_20, sma_20_prev)

    # HH/HL heuristic over last 10 bars
    win = min(10, len(highs) - 1)
    prior_high = max(highs[-(win + 1) : -1]) if win > 1 else highs[-2]
    prior_low = min(lows[-(win + 1) : -1]) if win > 1 else lows[-2]
    hh = 1.0 if highs[-1] > prior_high else 0.0
    hl = 1.0 if lows[-1] > prior_low else 0.0
    trend_score = (1.0 if ma_slope > 0 else -1.0) * (0.5 + 0.25 * hh + 0.25 * hl)
    return {
        "ma_slope": float(ma_slope),
        "hh": float(hh),
        "hl": float(hl),
        "trend_score": float(trend_score),
    }


def _market_proxy_features(rolling: Dict[str, Any], proxies: List[str], tf_key: str) -> Dict[str, float]:
    """Compute a small 'tape' snapshot from market proxy symbols.

    Output keys are prefixed by proxy symbol (e.g. mkt_spy_*).
    We also add a few aggregated keys (mkt_vol, mkt_trend).
    """

    bars_key = "bars_intraday_5m" if tf_key == "5Min" else "bars_intraday"

    out: Dict[str, float] = {}
    vols: List[float] = []
    trends: List[float] = []

    for sym in proxies:
        node = rolling.get(sym)
        if not isinstance(node, dict):
            continue

        bars = _extract_bars(node, bars_key)
        o, h, l, c, v, vw, ts = _ohlcv_from_bars(bars)
        if len(c) < 25:
            continue

        vwap_s = _session_vwap_series(h, l, c, v)
        rets = [pct_change(c[i], c[i - 1]) for i in range(1, len(c))]
        trend = _trend_structure(c, h, l)

        prefix = f"mkt_{sym.lower()}_"
        out[prefix + "last"] = float(c[-1])
        out[prefix + "vwap_dist"] = float(pct_change(c[-1], vwap_s[-1] if vwap_s else c[-1]))
        out[prefix + "ma_slope"] = float(trend.get("ma_slope") or 0.0)
        out[prefix + "vol"] = float(realized_vol(rets))
        out[prefix + "ret_20"] = float(pct_change(c[-1], c[-21])) if len(c) >= 21 else 0.0
        out[prefix + "trend_score"] = float(trend.get("trend_score") or 0.0)

        vols.append(float(realized_vol(rets)))
        trends.append(float(trend.get("trend_score") or 0.0))

    # Aggregates (helps bots treat proxies as a single market context)
    if vols:
        out["mkt_vol"] = float(sum(vols) / float(len(vols)))
    else:
        out["mkt_vol"] = 0.0

    if trends:
        out["mkt_trend"] = float(sum(trends) / float(len(trends)))
    else:
        out["mkt_trend"] = 0.0

    return out


def _feature_snapshot_for_symbol(sym: str, node: Dict[str, Any], *, rolling: Dict[str, Any], tf_key: str, mkt: Dict[str, float], now_utc: datetime) -> Dict[str, Any]:
    primary_key = "bars_intraday_5m" if tf_key == "5Min" else "bars_intraday"
    fallback_key = "bars_intraday" if primary_key == "bars_intraday_5m" else "bars_intraday_5m"

    bars = _extract_bars(node, primary_key)
    if len(bars) < 15:
        bars = _extract_bars(node, fallback_key)
        if len(bars) < 15:
            return {}
        used_tf = "1Min" if fallback_key == "bars_intraday" else "5Min"
    else:
        used_tf = tf_key

    o, h, l, c, v, vw, ts = _ohlcv_from_bars(bars)
    if len(c) < 15:
        return {}

    rets = [pct_change(c[i], c[i - 1]) for i in range(1, len(c))]
    rv = realized_vol(rets)

    vwap_s = _session_vwap_series(h, l, c, v)
    vwap_last = vwap_s[-1]
    vwap_dist = pct_change(c[-1], vwap_last)
    vwap_slope = lin_slope(vwap_s, window=min(20, len(vwap_s)))

    a14 = atr(h, l, c, window=14)

    # Opening range uses 1m bars if we have enough
    one_min_bars = _extract_bars(node, "bars_intraday")
    if one_min_bars:
        o1, h1, l1, c1, v1, vw1, ts1 = _ohlcv_from_bars(one_min_bars)
    else:
        o1, h1, l1, c1, v1, vw1, ts1 = ([], [], [], [], [], [], [])

    use_ts = ts1 if len(ts1) >= 30 else ts
    use_h = h1 if len(h1) >= 30 else h
    use_l = l1 if len(l1) >= 30 else l
    or5_h, or5_l = _opening_range(use_ts, use_h, use_l, 5, open_utc=open_utc)
    or15_h, or15_l = _opening_range(use_ts, use_h, use_l, 15, open_utc=open_utc)

    trend = _trend_structure(c, h, l)

    bb_w = bollinger_width(c, window=20, n_std=2.0)
    kc_w = keltner_width(c, h, l, c, window=20, atr_mult=1.5)
    squeeze_ratio = (bb_w / kc_w) if kc_w > 0 else 0.0
    squeeze_on = 1.0 if (kc_w > 0 and bb_w > 0 and bb_w < kc_w) else 0.0

    vol_win = min(20, len(v))
    vol_avg = (sum(v[-vol_win:]) / float(vol_win)) if vol_win > 0 else 0.0
    rel_vol = (v[-1] / vol_avg) if vol_avg > 0 else 0.0

    sma_20 = sma(c, 20)
    ema_9 = ema(c, 9)
    rsi_14 = rsi(c, 14)
    sd_20 = stddev(c[-20:]) if len(c) >= 20 else 0.0

    last_price = c[-1]
    open_price = c[0]

    feat: Dict[str, Any] = {
        "ts": now_utc.isoformat(timespec="seconds").replace("+00:00","Z"),
        "tf": used_tf,
        "last_price": float(last_price),
        "pct_chg_from_open": float(pct_change(last_price, open_price)),

        # VWAP
        "vwap": float(vwap_last),
        "vwap_dist": float(vwap_dist),
        "vwap_slope": float(vwap_slope),

        # Vol/ATR
        "atr_14": float(a14),
        "realized_vol": float(rv),

        # Opening range
        "or5_high": float(or5_h),
        "or5_low": float(or5_l),
        "or15_high": float(or15_h),
        "or15_low": float(or15_l),
        "or5_break": float(1.0 if (or5_h and last_price > or5_h) else (-1.0 if (or5_l and last_price < or5_l) else 0.0)),
        "or15_break": float(1.0 if (or15_h and last_price > or15_h) else (-1.0 if (or15_l and last_price < or15_l) else 0.0)),

        # Trend structure
        "ma_slope": float(trend.get("ma_slope") or 0.0),
        "hh": float(trend.get("hh") or 0.0),
        "hl": float(trend.get("hl") or 0.0),
        "trend_score": float(trend.get("trend_score") or 0.0),

        # Squeeze
        "bb_width": float(bb_w),
        "kc_width": float(kc_w),
        "squeeze_ratio": float(squeeze_ratio),
        "squeeze_on": float(squeeze_on),

        # Volume
        "rel_volume": float(rel_vol),

        # Common indicators
        "sma_20": float(sma_20),
        "ema_9": float(ema_9),
        "rsi_14": float(rsi_14),
        "sd_20": float(sd_20),
        "sma20_dist": float(pct_change(last_price, sma_20)) if sma_20 else 0.0,

        # Market proxy features
        **mkt,
    }

    # Merge intraday context if present (avoid collisions)
    ctx = node.get("context_dt") or {}
    if isinstance(ctx, dict):
        for k, v_ in ctx.items():
            if k in feat:
                continue
            feat[k] = v_

    return feat


def build_intraday_features(max_symbols: int | None = None, *, now_utc: datetime | None = None, ignore_min_interval: bool = False) -> Dict[str, Any]:
    """Compute features_dt for each symbol in rolling."""
    rolling = _read_rolling() or {}
    if not isinstance(rolling, dict) or not rolling:
        log("[dt_features] ⚠️ rolling empty, nothing to do.")
        return {"symbols": 0, "updated": 0}

    tf_key = (os.getenv("DT_FEATURE_TF", "5Min") or "5Min").strip()

    now_utc = now_utc or datetime.now(timezone.utc)

    tf_key = "1Min" if tf_key.lower().startswith("1") else "5Min"

    min_int = 0.0 if ignore_min_interval else _env_float("DT_FEATURES_MIN_INTERVAL", 0.0)
    now_t = time.time()

    proxies = [s.strip().upper() for s in (os.getenv("DT_MARKET_PROXIES", "SPY,QQQ").split(",")) if s.strip()]
    if not proxies:
        proxies = ["SPY", "QQQ"]
    mkt = _market_proxy_features(rolling, proxies, tf_key)

    items = [(sym, node) for sym, node in rolling.items() if isinstance(sym, str) and not sym.startswith("_")]
    items.sort(key=lambda kv: kv[0])
    if max_symbols is not None:
        items = items[: max(0, int(max_symbols))]

    updated = 0
    skipped = 0
    for sym, node_raw in items:
        if not isinstance(node_raw, dict):
            continue
        node = ensure_symbol_node(rolling, sym)

        if min_int > 0:
            meta = node.get("_features_meta")
            if isinstance(meta, dict):
                try:
                    last_t = float(meta.get("last_t") or 0.0)
                except Exception:
                    last_t = 0.0
                if last_t > 0 and (now_t - last_t) < min_int:
                    skipped += 1
                    continue

        feat = _feature_snapshot_for_symbol(sym, node, rolling=rolling, tf_key=tf_key, mkt=mkt, now_utc=now_utc)
        if not feat:
            continue

        node["features_dt"] = feat
        node["_features_meta"] = {"last_t": now_t, "tf": feat.get("tf")}
        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"[dt_features] ✅ updated features_dt for {updated} symbols (skipped={skipped}) tf={tf_key}.")
    return {"symbols": len(items), "updated": updated, "skipped": skipped, "tf": tf_key}

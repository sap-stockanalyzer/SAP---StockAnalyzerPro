
# dt_backend/engines/indicators.py — v2.0
"""
Low-level intraday indicator utilities for AION dt_backend.

Design:
  • Pure-Python, dependency-light
  • Operate on basic sequences / lists
  • Defensive: never raise in normal use; return safe defaults

These helpers are used by feature_engineering to build per-symbol
feature snapshots for intraday models.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence
import math


def _to_list(x: Iterable[float]) -> List[float]:
    return [float(v) for v in x]


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(values: Sequence[float], window: int) -> float:
    """
    Simple moving average of the last `window` observations.

    Returns 0.0 if not enough data.
    """
    vals = _to_list(values)
    if window <= 0 or len(vals) < window:
        return 0.0
    chunk = vals[-window:]
    return sum(chunk) / float(window)


def ema(values: Sequence[float], window: int) -> float:
    """
    Exponential moving average with standard smoothing factor.

    If not enough data, returns simple mean of available values.
    """
    vals = _to_list(values)
    if window <= 0 or not vals:
        return 0.0
    if len(vals) < window:
        return sum(vals) / float(len(vals))
    alpha = 2.0 / (window + 1.0)
    ema_val = vals[0]
    for v in vals[1:]:
        ema_val = alpha * v + (1.0 - alpha) * ema_val
    return ema_val


# ---------------------------------------------------------------------------
# Volatility & returns
# ---------------------------------------------------------------------------

def pct_change(a: float, b: float) -> float:
    """Return (a / b - 1) with safety checks."""
    try:
        if b == 0.0 or not (math.isfinite(a) and math.isfinite(b)):
            return 0.0
        return (a / b) - 1.0
    except Exception:
        return 0.0


def realized_vol(returns: Sequence[float]) -> float:
    """
    Naive realized volatility (standard deviation of returns).

    For intraday usage we don't annualize; this is a raw measure
    of choppiness inside the session.
    """
    vals = _to_list(returns)
    n = len(vals)
    if n < 2:
        return 0.0
    mu = sum(vals) / float(n)
    var = sum((x - mu) ** 2 for x in vals) / float(max(n - 1, 1))
    if var < 0.0:
        var = 0.0
    return math.sqrt(var)


def stddev(values: Sequence[float]) -> float:
    """Standard deviation (sample) with safe defaults."""
    vals = _to_list(values)
    n = len(vals)
    if n < 2:
        return 0.0
    mu = sum(vals) / float(n)
    var = sum((x - mu) ** 2 for x in vals) / float(max(n - 1, 1))
    return math.sqrt(max(0.0, var))


def true_range(high: float, low: float, prev_close: float) -> float:
    """True range for a single bar."""
    try:
        return max(
            float(high) - float(low),
            abs(float(high) - float(prev_close)),
            abs(float(low) - float(prev_close)),
        )
    except Exception:
        return 0.0


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], window: int = 14) -> float:
    """Average True Range (simple average of TR over window)."""
    try:
        hs = _to_list(highs)
        ls = _to_list(lows)
        cs = _to_list(closes)
    except Exception:
        return 0.0
    n = min(len(hs), len(ls), len(cs))
    if n < 2 or window <= 0:
        return 0.0
    trs: List[float] = []
    for i in range(1, n):
        trs.append(true_range(hs[i], ls[i], cs[i - 1]))
    if len(trs) < window:
        window = len(trs)
    if window <= 0:
        return 0.0
    chunk = trs[-window:]
    return sum(chunk) / float(window)


def lin_slope(values: Sequence[float], window: int = 10) -> float:
    """Simple linear-regression slope over the last `window` points."""
    vals = _to_list(values)
    if window <= 1 or len(vals) < window:
        return 0.0
    y = vals[-window:]
    n = float(window)
    sx = (n - 1) * n / 2.0
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sy = sum(y)
    sxy = sum(i * y[i] for i in range(window))
    denom = (n * sxx - sx * sx)
    if denom == 0.0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def bollinger_width(values: Sequence[float], window: int = 20, n_std: float = 2.0) -> float:
    vals = _to_list(values)
    if window <= 1 or len(vals) < window:
        return 0.0
    chunk = vals[-window:]
    mid = sum(chunk) / float(window)
    sd = stddev(chunk)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return max(0.0, upper - lower)


def keltner_width(values: Sequence[float], highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], window: int = 20, atr_mult: float = 1.5) -> float:
    vals = _to_list(values)
    if window <= 1 or len(vals) < window:
        return 0.0
    mid = ema(vals, window)
    a = atr(highs, lows, closes, window)
    upper = mid + atr_mult * a
    lower = mid - atr_mult * a
    return max(0.0, upper - lower)


def session_vwap(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], volumes: Sequence[float]) -> float:
    """Session VWAP (volume-weighted average price) computed from bar data.

    We use the bar 'typical price' (H+L+C)/3.
    Returns 0.0 if volume is missing or sums to zero.
    """
    try:
        hs = _to_list(highs)
        ls = _to_list(lows)
        cs = _to_list(closes)
        vs = _to_list(volumes)
    except Exception:
        return 0.0
    n = min(len(hs), len(ls), len(cs), len(vs))
    if n <= 0:
        return 0.0
    num = 0.0
    den = 0.0
    for i in range(n):
        v = vs[i]
        if v <= 0:
            continue
        tp = (hs[i] + ls[i] + cs[i]) / 3.0
        num += tp * v
        den += v
    if den <= 0.0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index) — exported as rsi_14 via feature_engineering
# ---------------------------------------------------------------------------

def rsi(values: Sequence[float], window: int = 14) -> float:
    """
    Classic Wilder RSI on closing prices.

    Returns value in [0, 100]. If not enough data, returns 50.
    """
    vals = _to_list(values)
    if window <= 0 or len(vals) <= window:
        return 50.0

    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(vals)):
        diff = vals[i] - vals[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)

    if len(gains) < window:
        return 50.0

    avg_gain = sum(gains[:window]) / float(window)
    avg_loss = sum(losses[:window]) / float(window)

    # Wilder smoothing
    for i in range(window, len(gains)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / float(window)
        avg_loss = (avg_loss * (window - 1) + losses[i]) / float(window)

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return max(0.0, min(100.0, rsi_val))

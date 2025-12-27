# backend/services/macro_fetcher.py — v1.1
"""
macro_fetcher.py — v1.1 (Do-not-overwrite-on-failure + batch yfinance + atomic writes)

Fixes:
  ✅ Batch-fetch tickers with one yfinance call (less rate-limit pain)
  ✅ If fetch returns junk (zeros/empty), DO NOT overwrite existing snapshots
  ✅ Atomic writes (tmp -> replace)
  ✅ No duplicate SPY fetch for breadth
  ✅ Keeps spy_daily_pct (percent) + spy_pct_decimal (decimal) clear
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import yfinance as yf

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import log, safe_float


# ------------------------------------------------------------
# Atomic write helper
# ------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------
# Batch yfinance helper
# ------------------------------------------------------------

def _extract_close_series(df: Any, symbol: str) -> Any:
    """
    df can be:
      - regular columns: ["Open","High","Low","Close"...] (single ticker)
      - MultiIndex columns: top level = ticker, second level = OHLCV (multi ticker)
    Return a "Close" series-like object or None.
    """
    try:
        if df is None or getattr(df, "empty", True):
            return None

        cols = getattr(df, "columns", None)
        if cols is None:
            return None

        # Multi-ticker: df[symbol]["Close"]
        if hasattr(cols, "levels") and len(cols.levels) >= 2:
            if symbol in df.columns.get_level_values(0):
                sub = df[symbol]
                if isinstance(sub, dict):
                    return None
                if "Close" in sub.columns:
                    return sub["Close"]
            return None

        # Single ticker: df["Close"]
        if "Close" in df.columns:
            return df["Close"]
        return None
    except Exception:
        return None


def _last2_close_pct(close_series: Any) -> Tuple[float, float]:
    """
    Return (close, pct) where pct is percent change vs previous close.
    """
    try:
        s = close_series.dropna()
        if len(s) < 2:
            return 0.0, 0.0
        last = float(s.iloc[-1])
        prev = float(s.iloc[-2])
        if prev <= 0:
            return float(last if last > 0 else 0.0), 0.0
        pct = ((last - prev) / prev) * 100.0
        return float(last), float(pct)
    except Exception:
        return 0.0, 0.0


def _yf_last2_batch(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    One yfinance call for many tickers. Returns:
      { "SPY": {"close":..., "pct":...}, ... }
    """
    out: Dict[str, Dict[str, float]] = {s: {"close": 0.0, "pct": 0.0} for s in symbols}
    syms = [s for s in symbols if str(s).strip()]
    if not syms:
        return out

    try:
        df = yf.download(
            tickers=" ".join(syms),
            period="5d",
            interval="1d",
            progress=False,
            group_by="ticker",
            threads=True,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return out

        # For each symbol, extract close series
        for s in syms:
            close_series = _extract_close_series(df, s)
            if close_series is None:
                continue
            close, pct = _last2_close_pct(close_series)
            out[s] = {"close": float(close), "pct": float(pct)}

        return out

    except Exception as e:
        log(f"[macro_fetcher] ⚠️ yfinance batch fetch failed: {e}")
        return out


# ------------------------------------------------------------
# Sanity gate
# ------------------------------------------------------------

def _macro_looks_sane(m: Dict[str, Any]) -> bool:
    """
    We consider macro sane if we have at least *some* real signal.
    Most important: SPY + VIX should not be zero.
    """
    try:
        vix = abs(safe_float(m.get("vix_close", 0.0)))
        spy = abs(safe_float(m.get("spy_close", 0.0)))
        spy_dec = abs(safe_float(m.get("spy_pct_decimal", 0.0)))
        breadth = abs(safe_float(m.get("breadth_proxy", 0.0)))

        # Require real closes for core indicators
        if vix <= 0.0 or spy <= 0.0:
            return False

        # Require at least a tiny bit of non-flatness OR a real VIX level
        if vix >= 8.0:  # VIX is basically never below ~8 historically
            return True

        return (spy_dec > 0.0001) or (breadth > 0.0001)
    except Exception:
        return False


# ------------------------------------------------------------
# Risk-off (same logic as your v1.0, kept stable)
# ------------------------------------------------------------

def _risk_off_score(vix_close: float, spy_pct_dec: float, dxy_pct_dec: float) -> float:
    vix_component = min(max((vix_close - 15.0) / 25.0, 0.0), 1.0)      # 15..40 mapped
    spy_component = min(max((-spy_pct_dec) / 0.03, 0.0), 1.0)          # -3% maps to 1
    dxy_component = min(max((dxy_pct_dec) / 0.01, 0.0), 1.0)           # +1% maps to 1
    return float(min(1.0, 0.45 * vix_component + 0.45 * spy_component + 0.10 * dxy_component))


# ------------------------------------------------------------
# Main builder
# ------------------------------------------------------------

def build_macro_features() -> Dict[str, Any]:
    log("🌐 Fetching macro signals (VIX, SPY, QQQ, TNX, DXY, GLD, USO)…")

    tickers = ["^VIX", "SPY", "QQQ", "^TNX", "DX-Y.NYB", "GLD", "USO"]
    data = _yf_last2_batch(tickers)

    vix_close, vix_pct = safe_float(data["^VIX"]["close"]), safe_float(data["^VIX"]["pct"])
    spy_close, spy_pct = safe_float(data["SPY"]["close"]), safe_float(data["SPY"]["pct"])
    qqq_close, qqq_pct = safe_float(data["QQQ"]["close"]), safe_float(data["QQQ"]["pct"])
    tnx_close, tnx_pct = safe_float(data["^TNX"]["close"]), safe_float(data["^TNX"]["pct"])
    dxy_close, dxy_pct = safe_float(data["DX-Y.NYB"]["close"]), safe_float(data["DX-Y.NYB"]["pct"])
    gld_close, gld_pct = safe_float(data["GLD"]["close"]), safe_float(data["GLD"]["pct"])
    uso_close, uso_pct = safe_float(data["USO"]["close"]), safe_float(data["USO"]["pct"])

    spy_pct_dec = spy_pct / 100.0
    dxy_pct_dec = dxy_pct / 100.0

    # breadth_proxy: keep your existing proxy style, but do not refetch SPY
    breadth_proxy = float(spy_pct_dec)

    volatility = float(max(0.0, min(0.10, float(vix_close) / 100.0)))
    risk_off = _risk_off_score(float(vix_close), float(spy_pct_dec), float(dxy_pct_dec))

    now_iso_utc = datetime.now(timezone.utc).isoformat()
    now_iso_local = datetime.now(TIMEZONE).isoformat()

    macro_state: Dict[str, Any] = {
        "vix_close": float(vix_close),
        "vix_daily_pct": float(vix_pct),

        "spy_close": float(spy_close),
        "spy_daily_pct": float(spy_pct),          # percent
        "spy_pct_decimal": float(spy_pct_dec),    # decimal

        "qqq_close": float(qqq_close),
        "qqq_daily_pct": float(qqq_pct),

        "tnx_close": float(tnx_close),
        "tnx_daily_pct": float(tnx_pct),

        "dxy_close": float(dxy_close),
        "dxy_daily_pct": float(dxy_pct),
        "dxy_pct_decimal": float(dxy_pct_dec),

        "gld_close": float(gld_close),
        "gld_daily_pct": float(gld_pct),

        "uso_close": float(uso_close),
        "uso_daily_pct": float(uso_pct),

        "breadth_proxy": float(breadth_proxy),

        # stable downstream keys
        "volatility": float(volatility),
        "risk_off": float(risk_off),

        # timestamps
        "generated_at": now_iso_local,
        "updated_at": now_iso_utc,
    }

    # ----------------------------
    # Critical: Do NOT overwrite on failure
    # ----------------------------
    if not _macro_looks_sane(macro_state):
        log("[macro_fetcher] ⚠️ Macro fetch looks invalid (zeros/empty). Keeping last snapshot (no overwrite).")
        return {"status": "skipped", "reason": "macro_not_sane", "macro_state": macro_state}

    # Save canonical macro_state.json
    out_path = PATHS.get("macro_state")
    if isinstance(out_path, Path):
        try:
            _atomic_write_json(out_path, macro_state)
            log(f"[macro_fetcher] 📈 macro_state.json updated → {out_path}")
        except Exception as e:
            log(f"[macro_fetcher] ⚠️ Failed to write macro_state.json: {e}")

    # Also write market_state-compatible snapshot for regime_detector fallbacks
    try:
        ml_root = PATHS.get("ml_data", Path("ml_data"))
        market_state_path = Path(ml_root) / "market_state.json"
        payload = dict(macro_state)
        payload.setdefault("regime_hint", "neutral")
        _atomic_write_json(market_state_path, payload)
        log(f"[macro_fetcher] 🧭 market_state.json (macro snapshot) updated → {market_state_path}")
    except Exception as e:
        log(f"[macro_fetcher] ⚠️ Failed to write market_state.json macro snapshot: {e}")

    return {"status": "ok", **macro_state}


if __name__ == "__main__":
    res = build_macro_features()
    print(json.dumps(res, indent=2))

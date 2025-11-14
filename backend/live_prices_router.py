"""
live_prices_router.py — FastAPI router serving /live-prices using StockAnalysis live snapshot fetch.
Supports both batch (top N) and on-demand symbol queries.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from fastapi import APIRouter, Query
from typing import Optional
from backend.data_pipeline import _fetch_from_stockanalysis, _read_rolling, log
import requests
import yfinance as yf

BASE_URL = "https://stockanalysis.com/api/screener/s/i/"  # StockAnalysis API base

# ==========================================================
# --- Intraday Bar Fetcher (Yahoo Fallback) ---------------
# ==========================================================
def get_intraday_bars_bulk(symbols, interval="1m", lookback_minutes=390):
    """
    Fetch intraday bars via Yahoo Finance fallback.
    Returns {symbol: pd.DataFrame[timestamp, open, high, low, close, volume]}.
    """
    out = {}
    end = datetime.utcnow()
    start = end - timedelta(minutes=lookback_minutes)

    # --- FIX: filter invalid placeholders ---
    valid_syms = [
        s for s in symbols
        if isinstance(s, str)
        and s.isalpha()
        and len(s) <= 6
        and s not in ("TIMESTAMP", "SYMBOLS", "BARS", "PRICES")
    ]

    for sym in valid_syms:
        try:
            df = yf.download(
                sym,
                interval=interval,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False
            )
            if df is None or df.empty:
                continue
            df = df.reset_index().rename(
                columns={
                    "Datetime": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            out[sym] = df
        except Exception as e:
            log(f"[live_prices_router] ⚠️ Failed to fetch {sym}: {e}")
            continue

    log(f"[live_prices_router] 🕒 Intraday bars built for {len(out)} symbols.")
    return out

# ==========================================================
# --- Live Prices Fetcher (StockAnalysis) ------------------
# ==========================================================
def _fetch_batch_from_stockanalysis(symbols: list[str]) -> dict:
    """Batch fetch live snapshot data for multiple tickers using StockAnalysis screener API (/s/i/)."""
    try:
        r = requests.get(BASE_URL, timeout=20)
        if r.status_code != 200:
            log(f"⚠️ StockAnalysis batch fetch returned {r.status_code}")
            return {}

        j = r.json()
        data = (j.get("data") or {}).get("data", [])
        if not data:
            log("⚠️ StockAnalysis returned no data (empty dataset).")
            return {}

        out = {}
        for item in data:
            sym = str(item.get("s", "")).upper()
            if not sym:
                continue
            out[sym] = {
                "symbol": sym,
                "name": item.get("n"),
                "price": item.get("price"),
                "change": item.get("change"),
                "industry": item.get("industry"),
                "volume": item.get("volume"),
                "marketCap": item.get("marketCap"),
                "pe_ratio": item.get("peRatio"),
            }

        log(f"💹 Successfully parsed {len(out)} tickers from StockAnalysis /s/i/")
        return out

    except Exception as e:
        log(f"⚠️ Batch fetch failed: {e}")
        return {}

# ==========================================================
# --- Sync Helper for Backend + DT Jobs --------------------
# ==========================================================
def fetch_live_prices(symbols: Optional[list[str]] = None, limit: int = 50) -> dict:
    """
    Sync batch fetch for backend + DT jobs using StockAnalysis screener /s/i/.
    Returns a dict keyed by symbol: { "AAPL": {"price": ..., "volume": ...}, ... }.
    """
    rolling = _read_rolling() or {}

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
    else:
        symbol_list = list(rolling.keys())[:limit]

    # --- FIX: ensure valid tickers only ---
    symbol_list = [
        s for s in symbol_list
        if isinstance(s, str)
        and s.isalpha()
        and len(s) <= 6
        and s not in ("TIMESTAMP", "SYMBOLS", "BARS", "PRICES")
    ]

    if not symbol_list:
        log("⚠️ No valid symbols to fetch.")
        return {}

    results = _fetch_batch_from_stockanalysis(symbol_list)
    log(f"💹 Batch fetched {len(results)} tickers from StockAnalysis /s/i/")
    return results

# ==========================================================
# --- API Router Endpoint ---------------------------------
# ==========================================================
router = APIRouter()

@router.get("/live-prices")
async def get_live_prices(
    symbols: Optional[str] = Query(
        None,
        description="Comma-separated list of symbols, e.g. AAPL,MSFT,TSLA. "
                    "If omitted, the first 50 tickers in Rolling are used."
    ),
    limit: int = Query(50, description="Number of tickers to fetch if no symbols provided."),
):
    """
    Returns live price snapshots for selected or default symbols.
    Pulls directly from StockAnalysis API via data_pipeline._fetch_from_stockanalysis.
    """
    rolling = _read_rolling() or {}

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        symbol_list = list(rolling.keys())[:limit]

    # --- FIX: filter invalid placeholders ---
    symbol_list = [
        s for s in symbol_list
        if isinstance(s, str)
        and s.isalpha()
        and len(s) <= 6
        and s not in ("TIMESTAMP", "SYMBOLS", "BARS", "PRICES")
    ]

    results = []
    for sym in symbol_list:
        try:
            snap = _fetch_from_stockanalysis(sym)
            if snap:
                results.append({
                    "symbol": sym,
                    "name": snap.get("name"),
                    "price": snap.get("close") or snap.get("price"),
                    "volume": snap.get("volume"),
                    "marketCap": snap.get("marketCap"),
                    "pe_ratio": snap.get("pe_ratio"),
                    "pb_ratio": snap.get("pb_ratio"),
                    "ps_ratio": snap.get("ps_ratio"),
                    "sector": snap.get("sector"),
                })
        except Exception as e:
            log(f"⚠️ live price fetch failed for {sym}: {e}")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(results),
        "symbols_requested": symbol_list,
        "prices": results,
    }

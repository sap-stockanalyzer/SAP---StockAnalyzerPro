"""
macro_fetcher.py — v2.1 (Unified Config + Sector ETF Breadth)
Author: AION Analytics / StockAnalyzerPro

Purpose:
- Compute macro breadth & sector-level ETF returns.
- Outputs both Parquet and JSON macro_features files.
- Unified via config.py PATHS for consistent folder layout.
"""

from __future__ import annotations
import os, json
import pandas as pd
from .config import PATHS  # ✅ unified path import

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
MACRO_DIR = PATHS["macro"]
STOCK_CACHE_DAILY = PATHS["stock_cache"] / "daily"

os.makedirs(MACRO_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Fetch sector ETF prices
# ---------------------------------------------------------------------
def fetch_sector_etfs() -> pd.DataFrame:
    """Load daily sector ETF data from stock_cache/daily and compute returns."""
    etfs = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
    rows = []
    for sym in etfs:
        p = STOCK_CACHE_DAILY / f"{sym}.json"
        if p.exists():
            df = pd.read_json(p)
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = sym
            rows.append(df[["date", "close", "symbol"]])

    if not rows:
        return pd.DataFrame()

    cat = pd.concat(rows).pivot(index="date", columns="symbol", values="close").sort_index()
    ret = cat.pct_change()
    ret.columns = [f"ret_{c}" for c in ret.columns]
    out = pd.concat([cat, ret], axis=1).reset_index()
    return out

# ---------------------------------------------------------------------
# Macro feature builder
# ---------------------------------------------------------------------
def build_macro_features() -> str:
    """Compute ETF return breadth and save to both Parquet and JSON in macro_cache."""
    etf = fetch_sector_etfs()
    out_path = MACRO_DIR / "macro_features.parquet"

    if etf.empty:
        pd.DataFrame().to_parquet(out_path)
        with open(MACRO_DIR / "macro_features.json", "w", encoding="utf-8") as f:
            json.dump({}, f)
        return str(out_path)

    ret_cols = [c for c in etf.columns if c.startswith("ret_")]
    etf["breadth_pos"] = (etf[ret_cols] > 0).sum(axis=1)

    etf.to_parquet(out_path, index=False)
    etf.to_json(MACRO_DIR / "macro_features.json", orient="records", date_format="iso")
    return str(out_path)

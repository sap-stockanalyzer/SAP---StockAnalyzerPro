"""
metrics_fetcher.py — v2.2 (Rolling-Aware + Key Normalization + Unified Config)
Author: AION Analytics / StockAnalyzerPro

Purpose:
- Fetch key quantitative metrics directly from StockAnalysis only for missing fields.
- Enrich each ticker node inside rolling.json.gz with fresh metrics.
- Skip redundant API calls when rolling already holds valid data.
- Normalize all metric keys to snake_case across the Rolling cache.
- Self-heal missing tickers via ensure_symbol_fields().
"""

import os, json, requests, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from .data_pipeline import (
    _read_rolling,
    _atomic_write_json_gz,
    RollingLock,
    ensure_symbol_fields,
    log,
)
from .config import PATHS  # ✅ unified config import

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SA_BASE = "https://stockanalysis.com/api/screener"
ROLLING_PATH = PATHS["rolling"]  # ✅ replaced hardcoded path

SMART_METRICS = [
    "ch1w","ch1m","ch3m","ch6m","ch1y","chYTD",
    "ma50","ma200","rsi","beta",
    "peRatio","pbRatio","psRatio","pegRatio","fcfYield",
    "earningsYield","marketCap","roa","roe","roic",
    "profitMargin","operatingMargin","grossMargin",
    "revenueGrowth","epsGrowth","debtEquity","debtEbitda"
]

# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------
NORMALIZE_KEYS = {
    "peRatio": "pe_ratio", "pbRatio": "pb_ratio", "psRatio": "ps_ratio",
    "pegRatio": "peg_ratio", "fcfYield": "fcf_yield", "earningsYield": "earnings_yield",
    "debtEquity": "debt_equity", "debtEbitda": "debt_ebitda",
    "revenueGrowth": "revenue_growth", "epsGrowth": "eps_growth",
    "profitMargin": "profit_margin", "operatingMargin": "operating_margin",
    "grossMargin": "gross_margin", "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio", "marketCap": "marketCap"
}

def normalize_keys(node: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metric keys from camelCase → snake_case for consistency."""
    if not isinstance(node, dict):
        return node
    for old, new in NORMALIZE_KEYS.items():
        if old in node and new not in node:
            node[new] = node.pop(old)
    return node


# ---------------------------------------------------------------------
# StockAnalysis fetcher
# ---------------------------------------------------------------------
def _fetch_metric(metric: str) -> Dict[str, Any]:
    """Fetch a single metric table from StockAnalysis (symbol → value)."""
    try:
        r = requests.get(f"{SA_BASE}/s/d/{metric}", timeout=10)
        if r.status_code != 200:
            return {}
        js = r.json()
        rows = js.get("data", {}).get("data", [])
        return {str(row[0]).upper(): row[1] for row in rows if isinstance(row, list) and len(row) >= 2}
    except Exception as e:
        log(f"⚠️ Metric fetch failed for {metric}: {e}")
        return {}


# ---------------------------------------------------------------------
# Main Enrichment Routine
# ---------------------------------------------------------------------
def build_latest_metrics(max_workers: int = 8) -> Dict[str, Any]:
    """
    Incremental metric enrichment:
    - Reads rolling.json.gz
    - Checks which metrics are missing or None
    - Fetches only those metrics from StockAnalysis
    - Updates rolling cache safely with normalized snake_case keys
    """
    start = time.time()
    log("[metrics_fetcher] 🚀 Starting incremental metrics update...")

    rolling = _read_rolling()
    if not rolling:
        log("⚠️ Rolling cache empty — cannot build metrics.")
        return {}

    # 🧩 Determine missing metrics
    missing_metrics = []
    for metric in SMART_METRICS:
        have_count = sum(
            1 for n in rolling.values()
            if isinstance(n, dict) and n.get(metric) not in (None, "", 0)
        )
        coverage = have_count / max(len(rolling), 1)
        if coverage < 0.95:
            missing_metrics.append(metric)

    if not missing_metrics:
        log("[metrics_fetcher] ℹ️ All metrics already present — skipping StockAnalysis fetch.")
        return {"updated": 0, "duration": 0.0}

    log(f"[metrics_fetcher] 🧩 Missing or incomplete metrics: {len(missing_metrics)} fields → {missing_metrics}")

    # 🧠 Fetch only missing metrics in parallel
    merged: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_metric, m): m for m in missing_metrics}
        for fut in as_completed(futures):
            metric = futures[fut]
            res = fut.result() or {}
            for sym, val in res.items():
                merged.setdefault(sym, {})[metric] = val

    # 🧩 Normalize + update Rolling
    updated = 0
    for sym, metrics in merged.items():
        sym = sym.upper()
        node = rolling.get(sym)
        if not node:
            node = ensure_symbol_fields(sym)
        if not node:
            continue

        node = normalize_keys(node)
        metrics = normalize_keys(metrics)

        changed = False
        for k, v in metrics.items():
            norm_k = NORMALIZE_KEYS.get(k, k)
            if norm_k not in node or node.get(norm_k) in (None, "", 0):
                node[norm_k] = v
                changed = True

        if changed:
            rolling[sym] = node
            updated += 1

    # ✅ Safe atomic save
    if updated:
        with RollingLock():
            _atomic_write_json_gz(ROLLING_PATH, rolling)

    dur = time.time() - start
    log(f"[metrics_fetcher] ✅ Incremental metrics update complete — {updated} tickers updated in {dur:.1f}s.")
    return {"updated": updated, "duration": dur}

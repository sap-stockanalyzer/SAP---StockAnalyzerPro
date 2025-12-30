"""
metrics_fetcher.py — v3.1
Aligned with new backend/core + nightly_job v4.0
--------------------------------------------------

Goal:
    Fetch / refresh StockAnalysis "metrics" for all Rolling symbols.

StockAnalysis provides 200+ indicators; we only fetch a curated set that:
    • is stable across all symbols
    • is used by your ML models
    • overlaps with your fundamentals / backfill bundle

Inputs:
    /s/d/<metric>
        Example: https://stockanalysis.com/api/screener/s/d/rsi

Outputs:
    rolling[sym]["metrics"] = {
        "<metric>": value,
        ...
    }

This module:
    • Uses batch fetching for efficiency
    • Normalizes all keys
    • Merges cleanly into Rolling
    • Never erases existing rolling data
    • Supports in-memory merge (so backfill can call it BEFORE rolling is saved)
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import requests

from backend.core.data_pipeline import (
    _read_rolling,
    save_rolling,
    safe_float,
    log,
)
from backend.core.config import PATHS


# ==============================================================================
# CONFIG
# ==============================================================================

SA_BASE = "https://stockanalysis.com/api/screener"

# Best-practice metrics for technical + valuation layers
METRIC_LIST = [
    "rsi",             # → rsi_14
    "ma20", "ma50", "ma200",
    "beta",
    "pbRatio", "psRatio", "pegRatio",
    "ch1w", "ch1m", "ch3m", "ch6m", "ch1y", "chYTD",
    "volatility",
]

# Normalize camelCase → snake_case
NORMALIZE = {
    "pbRatio": "pb_ratio",
    "psRatio": "ps_ratio",
    "pegRatio": "peg_ratio",
    "rsi": "rsi_14",
}


# ==============================================================================
# HELPERS
# ==============================================================================

def _rolling_path_str() -> str:
    p = PATHS.get("rolling") or PATHS.get("rolling_body")
    try:
        return str(p)
    except Exception:
        return "<rolling>"


def _sa_get_metric_table(metric: str) -> Dict[str, Any]:
    """Fetch /s/d/<metric> and return {SYM: value} map."""
    url = f"{SA_BASE}/s/d/{metric}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return {}
        js = r.json()
        rows = (js or {}).get("data", {}).get("data", [])

        out: Dict[str, Any] = {}
        for row in rows:
            if isinstance(row, list) and len(row) >= 2:
                sym = str(row[0]).upper()
                val = row[1]
            else:
                # tolerate dict rows
                try:
                    sym = (row.get("symbol") or row.get("s") or "").upper()
                    val = row.get(metric)
                except Exception:
                    continue

            if sym:
                out[sym] = val

        return out

    except Exception as e:
        log(f"⚠️ Metric fetch failed for '{metric}': {e}")
        return {}


def _normalize_key(k: str) -> str:
    return NORMALIZE.get(k, k)


# ==============================================================================
# MAIN METRICS REFRESH
# ==============================================================================

def build_latest_metrics(rolling: Optional[Dict[str, Any]] = None, *, persist: bool = True) -> Dict[str, Any]:
    """
    Fetch all metric tables and merge into rolling[sym]["metrics"].

    - If `rolling` is None, loads canonical rolling via _read_rolling().
    - If `persist` is True, saves via save_rolling() at the end.

    Called from:
        nightly_job.py — BEFORE model training
        backfill_history.py — AFTER history rebuild (in-memory merge, persist=False)
    """
    in_memory = rolling is not None

    if rolling is None:
        rolling = _read_rolling()

    if not rolling:
        log(f"⚠️ No rolling cache at {_rolling_path_str()} — skipping metrics fetch.")
        return {"status": "no_rolling", "in_memory": bool(in_memory)}

    log("📊 Fetching latest StockAnalysis metrics…")

    # Step 1 — fetch each table
    metric_tables: Dict[str, Dict[str, Any]] = {}
    for metric in METRIC_LIST:
        metric_tables[metric] = _sa_get_metric_table(metric)

    updated = 0
    total_symbols = 0

    # Step 2 — merge metrics into Rolling
    # NOTE: keys should already be uppercase, but we normalize defensively.
    for sym, node in list(rolling.items()):
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue

        total_symbols += 1
        sym_u = sym.upper()

        metrics = node.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        changed = False
        for metric in METRIC_LIST:
            tbl = metric_tables.get(metric, {})
            if sym_u not in tbl:
                continue

            new_key = _normalize_key(metric)
            metrics[new_key] = safe_float(tbl[sym_u])
            changed = True

        if changed:
            node["metrics"] = metrics
            rolling[sym_u] = node
            updated += 1

    if persist:
        save_rolling(rolling)

    log(f"✅ Metrics updated for {updated}/{total_symbols} symbols.")
    return {
        "status": "ok",
        "updated": int(updated),
        "total_symbols": int(total_symbols),
        "persisted": bool(persist),
        "in_memory": bool(in_memory),
    }


# ==============================================================================
# Nightly Job Compatibility Wrapper
# ==============================================================================

def build_metrics(rolling=None, *, persist: bool = True):
    """
    Wrapper used by nightly_job/backfill.
    If a rolling dict is provided, merges into it; otherwise loads from disk.
    """
    return build_latest_metrics(rolling=rolling, persist=persist)

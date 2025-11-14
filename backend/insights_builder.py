"""
insights_builder.py — v2.6 (Unified Config + Read-Only Optimized Top-50 Insights)
Author: AION Analytics / StockAnalyzerPro

Purpose:
- Consumes rolling cache already enriched with ai_model predictions
- No re-prediction or model loading (pure aggregation)
- Builds and writes Top-50 JSONs per horizon:
    PATHS["insights"]/top50_1w.json
    PATHS["insights"]/top50_2w.json
    PATHS["insights"]/top50_4w.json
    PATHS["insights"]/top50_52w.json
- Normalizes all keys (snake_case) for Rolling consistency.
- Adds tqdm progress bar + runtime summary.
"""

from __future__ import annotations
import os, json
from typing import Dict, Any, List
from tqdm import tqdm
from time import time
import orjson
from .config import PATHS  # ✅ unified config import
from .data_pipeline import _read_rolling, log

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
INSIGHTS_DIR = PATHS["insights"]           # ✅ unified path
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
HORIZONS = ["1w", "2w", "4w", "52w"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _row_for(sym: str, p: Dict[str, Any], horizon: str, current_price: float) -> Dict[str, Any]:
    """Formats one ticker’s prediction row."""
    return {
        "ticker": sym,
        "currentPrice": round(float(current_price), 4),
        "predictedPrice": float(p.get("predictedPrice", 0.0)),
        "expectedReturnPct": round(
            (p.get("predictedPrice", 0.0) - current_price) / current_price * 100.0, 4
        ),
        "confidence": float(p.get("confidence", 0.0)),
        "score": float(p.get("score", 0.0)),
        "rankingScore": float(p.get("rankingScore", 0.0)),
        "horizon": horizon,
    }

# ---------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------
def build_daily_insights(limit: int = 50) -> Dict[str, Any]:
    """
    Builds and saves Top-50 boards per horizon from existing predictions in Rolling.
    Returns a summary dict with counts per horizon and output paths.
    """
    start_time = time()
    rolling = _read_rolling() or {}
    syms = list(rolling.keys())
    summary: Dict[str, Any] = {}
    boards: Dict[str, List[Dict[str, Any]]] = {h: [] for h in HORIZONS}

    log(f"[insights_builder] 🚀 Building Top-{limit} Insights from existing predictions ({len(syms):,} symbols)...")

    for sym in tqdm(syms, desc="Aggregating predictions", unit="ticker"):
        node = rolling.get(sym) or {}
        preds = node.get("predictions", {})
        if not preds:
            continue

        current_price = node.get("close") or node.get("price")
        try:
            current_price = float(current_price)
        except Exception:
            continue

        for h in HORIZONS:
            p = preds.get(h)
            if not p:
                continue
            boards[h].append(_row_for(sym, p, h, current_price))

    # -----------------------------------------------------------------
    # Build and write Top-50 per horizon
    # -----------------------------------------------------------------
    outputs = {}
    for h in HORIZONS:
        rows = boards[h]
        rows.sort(key=lambda r: (r["rankingScore"], r["expectedReturnPct"]), reverse=True)
        top = rows[:limit]
        for i, r in enumerate(top, 1):
            r["rank"] = i

        out_path = INSIGHTS_DIR / f"top50_{h}.json"
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(top, option=orjson.OPT_INDENT_2))
        outputs[h] = {"count": len(top), "path": str(out_path)}
        log(f"✅ Wrote Top-{len(top)} ({h}) → {out_path}")

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------
    dur = time() - start_time
    total_symbols = len(syms)
    total_preds = sum(len(v) for v in boards.values())
    print("\n────────────────────────────────────────────────────────")
    print(f"✅ Insights build complete for {total_symbols:,} symbols.")
    print(f"📊 Total predictions processed: {total_preds:,}")
    for h, meta in outputs.items():
        print(f"   • {h}: {meta['count']:>3} tickers → {meta['path']}")
    print(f"⏱️ Completed in {dur:.1f}s (read-only aggregation).")
    print(f"📁 Saved outputs → {INSIGHTS_DIR}/top50_*.json")
    print("────────────────────────────────────────────────────────\n")

    summary["outputs"] = outputs
    return summary


# ---------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    build_daily_insights(limit=50)

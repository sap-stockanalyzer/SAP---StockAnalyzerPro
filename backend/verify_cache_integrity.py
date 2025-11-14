
"""
verify_cache_integrity.py — Validates rolling cache integrity and coverage.
"""
import os, json, gzip, statistics as stats
from typing import Dict, Any

from data_pipeline import ROLLING_PATH, log

if not os.path.exists(ROLLING_PATH):
    log("❌ rolling.json.gz not found.")
    raise SystemExit(1)

with gzip.open(ROLLING_PATH, "rt", encoding="utf-8") as f:
    data: Dict[str, Any] = json.load(f)

n = len(data)
lengths = [(sym, len(node.get("history") or [])) for sym, node in data.items()]
lens = [l for _, l in lengths]
short = [sym for sym, L in lengths if L < 60]

log(f"📊 Tickers: {n}")
if lens:
    log(f"📈 Avg days: {sum(lens)/len(lens):.1f}, Median: {stats.median(lens):.1f}, Min: {min(lens)}, Max: {max(lens)}")
log(f"⚠️ Short histories (<60d): {len(short)}")
if short[:10]:
    log(f"  e.g., {short[:10]}")
log("✅ Integrity check complete.")

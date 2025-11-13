# backfill_intraday_yf.py
import yfinance as yf
import json, gzip, datetime, os
from tqdm import tqdm

ROLLING_PATH = r"data/stock_cache/master/rolling.json.gz"
OUT_PATH = r"data_dt/rolling_intraday.json.gz"

# Load your symbol universe
print("📥 Loading symbol list from rolling.json.gz ...")
with gzip.open(ROLLING_PATH, "rt", encoding="utf-8") as f:
    rolling = json.load(f)
symbols = list(rolling.keys())[:500]  # limit for test, increase if stable
print(f"✅ Loaded {len(symbols)} symbols.")

# ---------------------------------------------------------------------
# Fetch today's bars (1m) for each
# ---------------------------------------------------------------------
import datetime
from tqdm import tqdm
import yfinance as yf

def _coerce_float(x):
    """Safe float conversion from Series, scalar, or None."""
    try:
        v = x.iloc[0] if hasattr(x, "iloc") else x
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            return 0.0
        return float(v)
    except Exception:
        return 0.0

def _coerce_int(x):
    """Safe int conversion from Series, scalar, or None."""
    try:
        v = x.iloc[0] if hasattr(x, "iloc") else x
        if v is None or (isinstance(v, float) and (v != v)):
            return 0
        return int(v)
    except Exception:
        return 0

today = datetime.date.today().strftime("%Y-%m-%d")
bars_out = {}

for sym in tqdm(symbols, desc="Fetching"):
    try:
        df = yf.download(
            sym,
            interval="1m",
            period="1d",
            auto_adjust=True,
            progress=False
        )
        if df.empty:
            continue

        bars = []
        for ts, row in df.iterrows():
            bars.append({
                "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "o": round(_coerce_float(row["Open"]), 2),
                "h": round(_coerce_float(row["High"]), 2),
                "l": round(_coerce_float(row["Low"]), 2),
                "c": round(_coerce_float(row["Close"]), 2),
                "v": _coerce_int(row["Volume"]),
            })
        if bars:
            bars_out[sym] = bars
    except Exception as e:
        print(f"⚠️ {sym}: {e}")

# Save in the same structure as your Alpaca file
payload = {
    "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "symbols": len(bars_out),
    "bars": bars_out,
}
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
    json.dump(payload, f)

print(f"💾 Saved {len(bars_out)} symbols → {OUT_PATH}")
print("✅ Done.")

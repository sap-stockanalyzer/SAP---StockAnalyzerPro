"""
rank_fetch_scheduler.py — AION Intraday Rank-Aware Fetch Scheduler
------------------------------------------------------------------
Reads latest rank file, prioritizes higher-ranked tickers,
fetches intraday bars from Alpaca (live) or YFinance (off-hours),
and maintains /data_dt/rolling_intraday.json.gz in this format:

{
  "timestamp": "2025-11-08T21:45:00Z",
  "symbols": 5526,
  "bars": {
    "AAPL": [
      { "t": "2025-11-08T21:39:00Z", "o": 188.10, "h": 188.18, "l": 187.95, "c": 188.05, "v": 412562 },
      ...
    ]
  }
}
"""
import os, time, gzip, json, asyncio, requests, pytz, yfinance as yf
from datetime import datetime, timezone, timedelta
from dt_backend.config_dt import FETCH_SPEED_FACTOR, DT_PATHS
from backend.data_pipeline import log
from dotenv import load_dotenv
from pathlib import Path
from utils.progress_bar import progress_bar

# Load environment variables from backend/.env
load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")

# ------------------------------------------------------------
# Alpaca setup
# ------------------------------------------------------------
ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY or "",
    "APCA-API-SECRET-KEY": ALPACA_SECRET or "",
}
ALPACA_URL = "https://data.alpaca.markets/v2/stocks/bars"

DATA_PATH = DT_PATHS["dtrolling"]
RANK_FILE = DT_PATHS["dtsignals"] / "prediction_rank_fetch.json.gz"
os.makedirs(DATA_PATH.parent, exist_ok=True)

if not ALPACA_KEY or not ALPACA_SECRET:
    log("⚠️ Alpaca API keys not set — live intraday fetch disabled.")

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
HISTORICAL_BACKFILL = True
HISTORICAL_LOOKBACK_DAYS = 1
LIVE_LIMIT = 5
BACKFILL_LIMIT = 390

# ------------------------------------------------------------
# Market Hours Detection
# ------------------------------------------------------------
def is_market_open():
    """Return True if US markets are currently open (Mon–Fri, 9:30–16:00 ET)."""
    now = datetime.now(pytz.timezone("America/New_York"))
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_ranks(path: str | os.PathLike = RANK_FILE) -> dict:
    path = os.fspath(path)
    if not os.path.exists(path):
        log(f"⚠️ Rank file missing at {path}")
        return {"owned": [], "ranks": []}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"⚠️ Failed to load ranks: {e}")
        return {"owned": [], "ranks": []}

def load_rolling() -> dict:
    if not os.path.exists(DATA_PATH):
        return {"timestamp": None, "symbols": 0, "bars": {}}
    try:
        with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"⚠️ Could not read rolling intraday: {e}")
        return {"timestamp": None, "symbols": 0, "bars": {}}

def save_rolling(rolling: dict):
    try:
        rolling["timestamp"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        rolling["symbols"] = len(rolling.get("bars", {}))
        with gzip.open(DATA_PATH, "wt", encoding="utf-8") as f:
            json.dump(rolling, f, ensure_ascii=False, indent=2)
        log(f"💾 Saved rolling_intraday.json.gz ({rolling['symbols']} symbols)")
    except Exception as e:
        log(f"⚠️ Failed to save rolling intraday: {e}")

# ------------------------------------------------------------
# Fetch Alpaca bars (batch ≤200)
# ------------------------------------------------------------
async def fetch_alpaca_bars_batch(symbols, timeframe="1Min",
                                  start=None, end=None, limit=390):
    if not symbols:
        return {}
    out = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i+200]
        params = {
            "symbols": ",".join(chunk),
            "timeframe": timeframe,
            "limit": limit,
            "feed": "iex"   # free feed
        }
        if start and end:
            params["start"] = start
            params["end"] = end
        try:
            r = requests.get(ALPACA_URL, headers=HEADERS, params=params, timeout=20)
            if r.status_code != 200:
                log(f"⚠️ Alpaca {r.status_code}: {r.text[:200]}")
                continue
            bars = r.json().get("bars", {})
            out.update(bars)
            await asyncio.sleep(0.3)
        except Exception as e:
            log(f"⚠️ Alpaca fetch failed: {e}")
    return out

# ------------------------------------------------------------
# Fetch YFinance bars (fallback for off-hours)
# ------------------------------------------------------------
def _yahoo_symbol(sym: str) -> str:
    # BRK.A -> BRK-A, BF.B -> BF-B, LEN.B -> LEN-B, etc.
    return sym.replace(".", "-")

def fetch_yfinance_bars_batch(symbols, period="1d", interval="1m"):
    if not symbols:
        return {}
    out = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i+200]
        try:
            tickers = yf.download(
                tickers=" ".join(chunk),
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            for sym in chunk:
                try:
                    df = tickers[sym].dropna()
                    out[sym] = [
                        {
                            "t": ts.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                            "o": float(row["Open"]),
                            "h": float(row["High"]),
                            "l": float(row["Low"]),
                            "c": float(row["Close"]),
                            "v": int(row["Volume"]),
                        }
                        for ts, row in df.iterrows()
                    ]
                except Exception:
                    continue
        except Exception as e:
            log(f"⚠️ YFinance fetch failed: {e}")
    return out

# ------------------------------------------------------------
# Merge bars
# ------------------------------------------------------------
def merge_into_rolling(rolling: dict, new_data: dict, max_bars: int = 390):
    store = rolling.setdefault("bars", {})
    for sym, bars in new_data.items():
        clean = []
        for b in bars:
            try:
                clean.append({
                    "t": b.get("t"),
                    "o": b.get("o"),
                    "h": b.get("h"),
                    "l": b.get("l"),
                    "c": b.get("c"),
                    "v": b.get("v"),
                })
            except Exception:
                continue
        if not clean:
            continue
        existing = store.get(sym, [])
        merged = existing + clean
        merged = {x["t"]: x for x in merged if x.get("t")}
        merged = [merged[t] for t in sorted(merged.keys())]
        store[sym] = merged[-max_bars:]
    rolling["symbols"] = len(store)
    return rolling

# ------------------------------------------------------------
# Historical backfill
# ------------------------------------------------------------
async def run_historical_backfill(ranks):
    if not ranks:
        log("⚠️ No ranked symbols for historical backfill.")
        return
    log(f"⏳ Starting historical backfill for {len(ranks)} symbols...")
    symbols = [r["symbol"] for r in ranks]
    end = datetime.utcnow().replace(tzinfo=timezone.utc)
    start = end - timedelta(days=HISTORICAL_LOOKBACK_DAYS)
    start_str, end_str = start.isoformat(), end.isoformat()

    out_data = {}
    for sym in progress_bar(symbols, desc="Backfilling", unit="ticker", total=len(symbols)):
        data = await fetch_alpaca_bars_batch([sym], "1Min",
                                             start=start_str, end=end_str,
                                             limit=BACKFILL_LIMIT)
        out_data.update(data)

    log(f"✅ Completed Alpaca backfill for {len(out_data)} / {len(symbols)} symbols.")
    rolling = load_rolling()
    rolling = merge_into_rolling(rolling, out_data)
    save_rolling(rolling)
    log(f"✅ Historical backfill complete ({len(out_data)} symbols).")

# ------------------------------------------------------------
# Weighted rank-based scheduler
# ------------------------------------------------------------
async def rank_fetch_scheduler():
    log(f"⚡ Rank Fetch Scheduler started — speed factor {FETCH_SPEED_FACTOR}×")
    base_intervals = {4: 30, 2: 60, 1: 120, 0.5: 240}
    bands = [(500, 4), (1500, 2), (3000, 1), (5526, 0.5)]
    cycle = 0

    ranks_data = load_ranks()
    ranks = ranks_data.get("ranks", [])

    if not ranks:
        log("⚙️ No rank file found — seeding ranks from StockAnalysis universe...")
        try:
            r = requests.get("https://stockanalysis.com/api/screener/s/i/", timeout=15)
            if r.status_code == 200:
                data = r.json().get("data", {}).get("data", [])
                ranks = [
                    {"symbol": x["s"], "rank": i + 1, "predicted": 0, "confidence": 0.5}
                    for i, x in enumerate(data)
                ]
                log(f"✅ Seeded {len(ranks)} tickers from StockAnalysis for backfill.")
            else:
                log(f"⚠️ StockAnalysis seed fetch failed: {r.status_code}")
        except Exception as e:
            log(f"⚠️ Failed to seed ranks: {e}")

    if HISTORICAL_BACKFILL:
        await run_historical_backfill(ranks)

    while True:
        cycle += 1
        owned = set(ranks_data.get("owned", []))
        t0 = time.time()
        total_fetched = {}

        for cutoff, weight in bands:
            batch_syms = [r["symbol"] for r in ranks if r["rank"] <= cutoff]
            batch_syms = list(set(batch_syms) | owned)
            interval = base_intervals[weight] / FETCH_SPEED_FACTOR
            log(f"[Cycle {cycle}] 🌀 Fetch ≤{cutoff} (w={weight}) interval={interval:.1f}s")

            if is_market_open():
                log("📈 Market open — using Alpaca (IEX feed).")
                new_bars = await fetch_alpaca_bars_batch(batch_syms, limit=LIVE_LIMIT)
            else:
                log("🌙 Market closed — using YFinance for updates.")
                new_bars = fetch_yfinance_bars_batch(batch_syms, period="5d", interval="5m")

            total_fetched.update(new_bars)
            await asyncio.sleep(interval / 10)

        if total_fetched:
            rolling = load_rolling()
            rolling = merge_into_rolling(rolling, total_fetched)
            save_rolling(rolling)

        elapsed = time.time() - t0
        log(f"[Cycle {cycle}] ✅ Completed rank refresh in {elapsed:.1f}s "
            f"({len(total_fetched)} symbols updated)")
        await asyncio.sleep(1)

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(rank_fetch_scheduler())
    except KeyboardInterrupt:
        log("🛑 Rank Fetch Scheduler stopped by user.")

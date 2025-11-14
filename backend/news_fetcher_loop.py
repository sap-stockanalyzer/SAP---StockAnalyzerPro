"""Hourly news collector (Perigon API) — Rolling + Raw + Intel version."""
from __future__ import annotations
import os, json, time, gzip, datetime, hashlib, schedule, requests
from typing import Dict, List
from dotenv import load_dotenv
from .data_pipeline import log
from . import vector_index
from .config import PATHS  # ✅ unified config import
from pathlib import Path

# ---------------------------------------------------------------------
# Environment and Config
# ---------------------------------------------------------------------
load_dotenv()

# Use centralized paths from config.py
RAW_DIR = PATHS["news"]  # ✅ keep in same folder that news_intel expects
MASTER_PATH = PATHS["news"] / "master_news.json.gz"

PROVIDER = os.getenv("AION_NEWS_PROVIDER", "perigon")
PERIGON_KEY = os.getenv("PERIGON_KEY", "")

# ---------------------------------------------------------------------
# File-level lock to prevent concurrent writes to news_raw_*.json
# ---------------------------------------------------------------------

RAW_LOCK = PATHS["news"] / "news_raw.lock"

class NewsRawLock:
    """Lightweight lock file to serialize writes to news_raw_YYYYMMDD.json."""
    def __enter__(self):
        # Wait until no one else is writing
        while RAW_LOCK.exists():
            time.sleep(0.5)
        RAW_LOCK.write_text("locked")
        return self

    def __exit__(self, *args):
        try:
            RAW_LOCK.unlink(missing_ok=True)
        except Exception:
            pass

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _today_path() -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d")
    return str(RAW_DIR / f"news_raw_{ts}.json")

def _hash_url(url: str) -> str:
    return hashlib.md5((url or "").encode("utf-8")).hexdigest()

def _load_today() -> Dict[str, Dict]:
    path = _today_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_today(doc: Dict[str, Dict]):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(_today_path(), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    log(f"[news_fetcher] 💾 Saved daily raw → {_today_path()}")

def _load_master() -> Dict[str, Dict]:
    if not MASTER_PATH.exists():
        return {}
    try:
        with gzip.open(MASTER_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_master(data: Dict[str, Dict]):
    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(MASTER_PATH) + ".tmp"
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, MASTER_PATH)
    log(f"[news_fetcher] 💾 Master news archive updated → {MASTER_PATH}")

# ---------------------------------------------------------------------
# Fetcher (Perigon API)
# ---------------------------------------------------------------------
def _fetch_perigon() -> List[Dict]:
    """Fetches latest business and market headlines from Perigon."""
    if not PERIGON_KEY:
        log("[news_fetcher] ⚠️ PERIGON_KEY not set — skipping fetch.")
        return []
    try:
        params = {
            "category": "business,markets,technology",
            "language": "en",
            "country": "us",
            "pageSize": 100,
            "from": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        }
        headers = {"x-api-key": PERIGON_KEY}
        r = requests.get("https://api.goperigon.com/v1/all", params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            log(f"[news_fetcher] ⚠️ Perigon returned {r.status_code}: {r.text[:200]}")
            return []

        js = r.json()
        articles = js.get("articles", []) or []
        out = []
        for a in articles:
            out.append({
                "title": a.get("title"),
                "summary": a.get("description") or a.get("snippet") or "",
                "url": a.get("url"),
                "published_at": a.get("publishedAt"),
                "source": (a.get("source") or {}).get("name", ""),
                "tickers": a.get("tickers") or [],
                "sentiment": (a.get("sentiment") or {}).get("score"),
            })
        return out

    except Exception as e:
        log(f"[news_fetcher] ⚠️ Perigon fetch failed: {e}")
        return []

# ---------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------
def run():
    start = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    today_store = _load_today()
    master = _load_master()

    added = 0
    articles = _fetch_perigon() if PROVIDER == "perigon" else []

    for a in articles:
        url = a.get("url") or ""
        hid = _hash_url(url)
        if not url or hid in today_store:
            continue

        today_store[hid] = a
        master[hid] = a
        text = f"{a.get('title', '')} {a.get('summary', '')}"
        vector_index.add(doc_id=hid, text=text, meta={"url": url, "source": a.get("source")})
        added += 1

    with NewsRawLock():
        _save_today(today_store)
        _save_master(master)

    # ✅ Auto-run News Intelligence after fetch
    try:
        from backend.news_intel import run_news_intel
        run_news_intel()
        log("[news_fetcher] 🧠 News Intelligence updated after fetch.")
    except Exception as e:
        log(f"[news_fetcher] ⚠️ News Intelligence run failed: {e}")

    dur = time.time() - start
    log(f"[news_fetcher] ✅ Added {added} new / total {len(today_store)} today | Master: {len(master)} | {dur:.1f}s")

# ---------------------------------------------------------------------
# Scheduler (runs hourly)
# ---------------------------------------------------------------------
def _run_loop():
    log("[news_fetcher_loop] 📰 Starting hourly Perigon loop.")
    schedule.every().hour.do(run)  # ✅ fixed to hourly
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    _run_loop()

import os, time, re, asyncio, aiohttp, json, threading, schedule
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import PATHS  # ✅ unified path reference

# ============== API keys ==============
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d3ttfo9r01qvr0dkh0lgd3ttfo9r01qvr0dkh0m0")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "7f6db2001f8b4db0b0a6d24486ed3b46")
RSS2JSON_KEY = os.getenv("RSS2JSON_KEY", "b0dxju7gbwgmwq6djqpfr1xcym3kfdcswhtevpyr")

# ============== Cache Directories (from config) ==============
NEWS_CACHE_DIR = PATHS["news"]
os.makedirs(NEWS_CACHE_DIR, exist_ok=True)

MASTER_NEWS_FILE = NEWS_CACHE_DIR / "news_cache.json"
NEWS_CACHE: List[Dict[str, Any]] = []  # global cache (list of articles)

# Track last fetch (optional)
LAST_FETCHED: Dict[str, str] = {}

# ============== Ticker Map (persistent master) ==============
TICKER_MAP: Dict[str, str] = {}

try:
    stocks_file = PATHS["stock_cache"] / "master.json"
    if stocks_file.exists():
        with open(stocks_file, "r", encoding="utf-8") as f:
            stocks_data = json.load(f)

        for ticker, name in stocks_data.items():
            if isinstance(name, str) and name.strip():
                TICKER_MAP[ticker.upper()] = name.strip().lower()

        print(f"✅ Loaded {len(TICKER_MAP)} tickers from master.json for news filtering")
    else:
        print(f"⚠️ master.json not found at: {stocks_file}")
        TICKER_MAP = {}
except Exception as e:
    print(f"⚠️ Failed to load ticker map from master.json: {e}")
    TICKER_MAP = {}

# ---------------------------------------------------------------------
# File-level lock to prevent concurrent writes to news_raw_*.json
# ---------------------------------------------------------------------
from pathlib import Path
import time

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

# ============== Filtering Rules ==============
FINANCE_KEYWORDS = [
    "earnings", "guidance", "revenue", "profit", "merger", "ipo", "dividend",
    "sec", "upgrade", "downgrade", "fed", "interest rate", "inflation",
    "tariff", "trade war", "sanction", "gdp", "unemployment", "treasury",
    "yields", "bond", "oil", "gold", "crypto", "regulation", "market"
]

TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")


def _iso(dt: Optional[str]) -> str:
    if not dt:
        return ""
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00")).isoformat()
    except Exception:
        return dt


def tag_tickers(article: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join([article.get("title", ""), article.get("summary", "")]).lower()
    matched = []

    for t in TICKER_MAP.keys():
        if re.search(rf"\b{re.escape(t.lower())}\b", text):
            matched.append(t)

    for t, name in TICKER_MAP.items():
        if name and name in text:
            matched.append(t)

    for m in TICKER_PATTERN.findall(article.get("title", "") + " " + article.get("summary", "")):
        if m in TICKER_MAP:
            matched.append(m)

    article["tickers"] = sorted(list(set(matched)))
    return article


def categorize_article(article: Dict[str, Any]) -> str:
    text = " ".join([article.get("title", ""), article.get("summary", "")]).lower()
    if article.get("tickers"):
        return "stock"

    macro_terms = [
        "fed", "interest rate", "inflation", "tariff", "trade war", "sanction",
        "gdp", "unemployment", "treasury", "yields", "bond", "congress",
        "government", "white house"
    ]
    if any(term in text for term in macro_terms):
        return "macro"

    sector_terms = [
        "tech sector", "semiconductor", "chipmaker", "banking", "oil", "energy",
        "crypto", "healthcare", "pharma", "retail", "automotive"
    ]
    if any(term in text for term in sector_terms):
        return "sector"

    return "other"


def is_relevant(article: Dict[str, Any]) -> bool:
    """
    Heuristics that returns True for finance/market/company-relevant items.
    """
    title = article.get("title", "")
    summary = article.get("summary", "")
    text = f"{title} {summary}".lower()

    if TICKER_PATTERN.search(title) or TICKER_PATTERN.search(summary):
        return True
    for name in TICKER_MAP.values():
        if name and name in text:
            return True

    if any(kw in text for kw in FINANCE_KEYWORDS):
        return True

    if len(summary) < 40 and len(title) < 15:
        return False
    junk_words = ["football", "basketball", "hollywood", "celebrity", "music", "recipe"]
    if any(j in text for j in junk_words):
        return False

    return True


def _normalize_article(
    *,
    source: str,
    url: str,
    title: str,
    summary: str,
    published_at: str,
    image: Optional[str] = None,
    author: Optional[str] = None,
    id_override: Optional[str] = None,
) -> Dict[str, Any]:
    art = {
        "id": id_override or url,
        "source": source,
        "url": url,
        "title": title or "",
        "summary": summary or "",
        "publishedAt": _iso(published_at),
        "image": image,
        "author": author,
    }
    tag_tickers(art)
    art["type"] = categorize_article(art)
    return art


# ============== Fetchers ==============
async def fetch_finnhub_news(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    try:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": "general", "token": FINNHUB_API_KEY}
        async with session.get(url, params=params, timeout=20) as r:
            if r.status != 200:
                print(f"⚠️ Finnhub HTTP {r.status}")
                return []
            data = await r.json()

        results = []
        for item in data or []:
            url = item.get("url")
            if not url:
                continue
            art = _normalize_article(
                source="Finnhub",
                url=url,
                title=item.get("headline", ""),
                summary=item.get("summary", ""),
                published_at=item.get("datetime") and datetime.utcfromtimestamp(item["datetime"]).isoformat(),
                image=item.get("image"),
                author=item.get("source"),
            )
            if is_relevant(art):
                results.append(art)
        return results
    except Exception as e:
        print(f"⚠️ Finnhub fetch error: {e}")
        return []


async def fetch_newsapi(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {"category": "business", "country": "us", "apiKey": NEWSAPI_KEY, "pageSize": 50}
        async with session.get(url, params=params, timeout=20) as r:
            if r.status != 200:
                print(f"⚠️ NewsAPI HTTP {r.status}")
                return []
            data = await r.json()

        articles = data.get("articles") or []
        results = []
        for a in articles:
            link = a.get("url")
            if not link:
                continue
            art = _normalize_article(
                source=a.get("source", {}).get("name", "NewsAPI"),
                url=link,
                title=a.get("title", ""),
                summary=a.get("description", "") or "",
                published_at=a.get("publishedAt") or "",
                image=a.get("urlToImage"),
                author=a.get("author"),
            )
            if is_relevant(art):
                results.append(art)
        return results
    except Exception as e:
        print(f"⚠️ NewsAPI fetch error: {e}")
        return []


async def fetch_rss2json(session: aiohttp.ClientSession, feed_url: str) -> List[Dict[str, Any]]:
    try:
        api = "https://api.rss2json.com/v1/api.json"
        params = {"rss_url": feed_url, "api_key": RSS2JSON_KEY, "count": 50}
        async with session.get(api, params=params, timeout=25) as r:
            if r.status != 200:
                print(f"⚠️ RSS2JSON HTTP {r.status} for {feed_url}")
                return []
            data = await r.json()

        items = data.get("items") or []
        feed_title = (data.get("feed") or {}).get("title", "RSS")
        results = []
        for it in items:
            link = it.get("link")
            if not link:
                continue
            content = it.get("content", "") or it.get("description", "") or ""
            art = _normalize_article(
                source=feed_title,
                url=link,
                title=it.get("title", ""),
                summary=content,
                published_at=it.get("pubDate") or "",
                image=it.get("thumbnail") or None,
                author=it.get("author"),
            )
            if is_relevant(art):
                results.append(art)
        return results
    except Exception as e:
        print(f"⚠️ RSS2JSON fetch error for {feed_url}: {e}")
        return []


# ============== Update Loop ==============
async def update_news():
    global NEWS_CACHE
    async with aiohttp.ClientSession() as session:
        articles: List[Dict[str, Any]] = []

        finnhub = await fetch_finnhub_news(session) or []
        newsapi = await fetch_newsapi(session) or []

        articles.extend(finnhub)
        articles.extend(newsapi)

        rss_feeds = [
            "https://seekingalpha.com/feed.xml",
            "https://finance.yahoo.com/rss/topstories",
            "https://www.cnbc.com/id/15839135/device/rss/rss.html",
            "https://www.cnbc.com/id/10000664/device/rss/rss.html",
            "https://www.cnbc.com/id/44877279/device/rss/rss.html",
            "https://www.investing.com/rss/news.rss",
            "https://www.fool.co.uk/feed/",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "https://www.nasdaq.com/feed/rssoutbound?category=Stocks",
            "https://www.forbes.com/business/feed/",
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
        ]

        for feed in rss_feeds:
            rss_items = await fetch_rss2json(session, feed) or []
            articles.extend(rss_items)

        if articles:
            seen = set()
            unique_articles: List[Dict[str, Any]] = []
            for art in (articles + (NEWS_CACHE or [])):
                uid = art.get("id") or art.get("url")
                if not uid or uid in seen:
                    continue
                seen.add(uid)
                unique_articles.append(art)

            NEWS_CACHE = sorted(unique_articles, key=lambda x: x.get("publishedAt", ""), reverse=True)

            # ✅ Protect the write with the lock
            from .news_fetcher_loop import NewsRawLock  # or define it at top
            with NewsRawLock():
                save_cache()

            print(f"💾 Saved {len(NEWS_CACHE)} total articles (added {len(articles)} new this cycle)")

# ============== Save / Load / Loop ==============
def save_cache():
    safe = NEWS_CACHE or []
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    daily_file = NEWS_CACHE_DIR / f"{date_str}.json"
    try:
        with open(daily_file, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)
        with open(MASTER_NEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save news cache: {e}")


def load_cache():
    global NEWS_CACHE
    if MASTER_NEWS_FILE.exists():
        try:
            with open(MASTER_NEWS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Corrupted master news cache. Resetting.")
            NEWS_CACHE = []
            save_cache()
            return
        except Exception as e:
            print(f"⚠️ Failed to read master news cache: {e}")
            NEWS_CACHE = []
            return

        if isinstance(data, list):
            retro: List[Dict[str, Any]] = []
            seen = set()
            for a in data:
                uid = a.get("id") or a.get("url")
                if not uid or uid in seen:
                    continue
                seen.add(uid)
                tagged = tag_tickers(a)
                tagged["type"] = categorize_article(tagged)
                retro.append(tagged)
            NEWS_CACHE = sorted(retro, key=lambda x: x.get("publishedAt", ""), reverse=True)
            save_cache()
            print(f"✅ Retro-tagged & deduped {len(NEWS_CACHE)} cached articles")
        else:
            NEWS_CACHE = []
            save_cache()
            print("⚠️ Master news cache not a list; reset")
    else:
        NEWS_CACHE = []
        print("✅ No existing news cache found")


def start_news_loop():
    async def _loop():
        while True:
            try:
                await update_news()
            except Exception as e:
                print(f"⚠️ News loop error: {e}")
            await asyncio.sleep(300)  # every 5 minutes

    t = threading.Thread(target=lambda: asyncio.run(_loop()), daemon=True)
    t.start()
    print("📰 News background loop started")


if __name__ == "__main__":
    load_cache()
    start_news_loop()
    asyncio.run(update_news())

"""
social_sentiment_fetcher.py — v1.0
Author: AION Analytics / StockAnalyzerPro

Purpose:
    Collects Reddit posts from selected subreddits, performs FinBERT sentiment analysis,
    and writes a summarized per-ticker sentiment snapshot for daily enrichment.

Output:
    PATHS["news"]/social_sentiment_YYYYMMDD.json
"""

from __future__ import annotations
import os, json, datetime, time, re
from typing import Dict, Any, List
from backend.config import PATHS
from backend.data_pipeline import log
from backend.vector_index import add as add_to_index

# Optional: FinBERT model (requires transformers)
try:
    from transformers import pipeline
    _finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
except Exception as e:
    _finbert = None
    log(f"[social_sentiment] ⚠️ FinBERT unavailable ({e}); using keyword fallback.")

# Optional: Reddit API (PRAW)
try:
    import praw
    _reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="aion-analytics-bot"
    )
except Exception as e:
    _reddit = None
    log(f"[social_sentiment] ⚠️ Reddit API unavailable ({e}); skipping live fetch.")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
OUT_DIR = PATHS["news"]
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / f"social_sentiment_{datetime.date.today()}.json"

DEFAULT_SUBS = ["stocks", "investing", "wallstreetbets", "stockmarket"]
POST_LIMIT = int(os.getenv("SOCIAL_POST_LIMIT", "150"))

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
_TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")

def _extract_tickers(text: str) -> List[str]:
    """Rough ticker extractor from post title + body."""
    found = _TICKER_PATTERN.findall(text.upper())
    # Filter out common English words or false positives
    blacklist = {"A", "I", "IT", "OR", "FOR", "ALL", "ON", "BY", "AND", "THE", "ARE"}
    return sorted({t for t in found if t not in blacklist})[:10]

def _score_text(text: str) -> Dict[str, float]:
    """Returns FinBERT sentiment score or fallback heuristic."""
    if _finbert is not None:
        try:
            res = _finbert(text[:512])[0]
            label = res.get("label", "").lower()
            score = float(res.get("score", 0.0))
            if "positive" in label:
                return {"sentiment": 1 * score}
            elif "negative" in label:
                return {"sentiment": -1 * score}
            else:
                return {"sentiment": 0.0}
        except Exception:
            pass
    # fallback: simple word count sentiment
    pos = sum(w in text.lower() for w in ["buy", "bull", "call", "up", "long"])
    neg = sum(w in text.lower() for w in ["sell", "bear", "short", "down"])
    return {"sentiment": float(pos - neg)}

# ---------------------------------------------------------------------
# CORE FETCH + AGGREGATE
# ---------------------------------------------------------------------
def fetch_reddit_posts(subs: List[str] = DEFAULT_SUBS, limit: int = POST_LIMIT) -> List[Dict[str, Any]]:
    if not _reddit:
        log("[social_sentiment] ⚠️ Reddit API unavailable — returning empty list.")
        return []

    posts = []
    for sub in subs:
        try:
            for post in _reddit.subreddit(sub).new(limit=limit):
                text = f"{post.title} {post.selftext or ''}"
                tickers = _extract_tickers(text)
                if not tickers:
                    continue
                sent = _score_text(text)
                posts.append({
                    "sub": sub,
                    "title": post.title,
                    "tickers": tickers,
                    "sentiment": sent["sentiment"],
                    "upvotes": int(post.score or 0),
                    "created_utc": datetime.datetime.utcfromtimestamp(post.created_utc).isoformat() + "Z",
                    "url": f"https://reddit.com{post.permalink}",
                })
        except Exception as e:
            log(f"[social_sentiment] ⚠️ Subreddit fetch failed ({sub}): {e}")
        time.sleep(1.0)
    log(f"[social_sentiment] 📰 Collected {len(posts)} posts across {len(subs)} subreddits.")
    return posts

def aggregate_per_ticker(posts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregates sentiment and buzz per ticker."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in posts:
        s = float(p.get("sentiment", 0.0))
        up = int(p.get("upvotes", 0))
        for t in p.get("tickers", []):
            node = out.setdefault(t, {"mentions": 0, "avg_sentiment": 0.0, "buzz": 0})
            node["mentions"] += 1
            node["avg_sentiment"] += s
            node["buzz"] += max(1, up)
    for t, v in out.items():
        v["avg_sentiment"] = round(v["avg_sentiment"] / max(v["mentions"], 1), 4)
    return out

def save_daily_snapshot(per_ticker: Dict[str, Dict[str, Any]]):
    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(per_ticker, f, ensure_ascii=False, indent=2)
        log(f"[social_sentiment] 💾 Saved sentiment snapshot → {OUT_PATH}")
        # Add vector embeddings for semantic recall
        for t, row in list(per_ticker.items())[:200]:
            txt = f"{t} social sentiment {row['avg_sentiment']:+.3f}"
            add_to_index(f"reddit_{t}_{datetime.date.today()}", txt, {"ticker": t, "sentiment": row["avg_sentiment"]})
    except Exception as e:
        log(f"[social_sentiment] ⚠️ Failed to save snapshot: {e}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def run_social_sentiment():
    posts = fetch_reddit_posts()
    if not posts:
        log("[social_sentiment] ⚠️ No posts fetched.")
        return
    agg = aggregate_per_ticker(posts)
    save_daily_snapshot(agg)

if __name__ == "__main__":
    run_social_sentiment()

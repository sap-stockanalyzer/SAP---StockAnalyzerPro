"""
social_sentiment_fetcher.py — v3.0
AION Analytics — Social Market Intelligence Layer
"""

from __future__ import annotations

import os
import re
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import requests

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import (
    _read_rolling,
    save_rolling,
    safe_float,
    log,
)

# =====================================================================
# PATHS
# =====================================================================

CACHE_FILE = PATHS["social_intel"]
TICKER_REGEX = re.compile(r"\b[A-Z]{2,6}\b")

# Twitter is intentionally disabled by design (user constraint: NO TWITTER).
# We keep a placeholder variable for backwards compatibility.
TWITTER_BEARER = ""

# Reddit OAuth credentials (required)
REDDIT_CLIENT_ID = (os.getenv("REDDIT_CLIENT_ID", "") or "").strip()
REDDIT_CLIENT_SECRET = (os.getenv("REDDIT_CLIENT_SECRET", "") or "").strip()
# Backwards compat: some older envs used REDDIT_SECRET
if not REDDIT_CLIENT_SECRET:
    REDDIT_CLIENT_SECRET = (os.getenv("REDDIT_SECRET", "") or "").strip()
REDDIT_USER_AGENT = (os.getenv("REDDIT_USER_AGENT", "aion/1.0 (wallstreetbets sentiment)") or "").strip()


# =====================================================================
# Env helpers
# =====================================================================

def _env_int(name: str, default: int) -> int:
    """Read integer environment variable with safe fallback."""
    try:
        v = str(os.getenv(name, "")).strip()
        return int(float(v)) if v else int(default)
    except Exception:
        return int(default)


# =====================================================================
# Heuristics for sentiment
# =====================================================================

POS_WORDS = [
    "moon", "bull", "bullish", "rocket", "soaring", "profit",
    "gain", "pump", "call", "calls", "green", "run",
]

NEG_WORDS = [
    "bagholder", "dump", "bearish", "red", "crash", "puts",
    "collapse", "panic", "down", "loss", "bleeding",
]

def score_sentiment(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    raw = (pos - neg) / (pos + neg)
    return max(-1.0, min(1.0, raw))


def extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    tickers = [m.group(0) for m in TICKER_REGEX.finditer(text)]
    blacklist = {"YOLO", "DD", "CEO", "GDP", "USA", "FED", "IMO", "OTM"}
    return [t for t in set(tickers) if t not in blacklist]


# =====================================================================
# Reddit Fetcher (OAuth, /r/wallstreetbets only)
# =====================================================================

_REDDIT_TOKEN_CACHE: dict = {"access_token": "", "expires_at": 0.0}


def _reddit_oauth_token() -> str:
    """Return a cached OAuth token for Reddit (client credentials).

    Uses env vars:
      - REDDIT_CLIENT_ID
      - REDDIT_CLIENT_SECRET
      - REDDIT_USER_AGENT

    Notes:
      * This uses the *app-only* client_credentials flow.
      * Works for reading public subreddit content.
    """
    import time

    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return ""

    now = time.time()
    tok = str(_REDDIT_TOKEN_CACHE.get("access_token") or "")
    exp = float(_REDDIT_TOKEN_CACHE.get("expires_at") or 0.0)
    if tok and now < exp:
        return tok

    try:
        auth = (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        headers = {"User-Agent": REDDIT_USER_AGENT or "aion/1.0"}
        data = {"grant_type": "client_credentials"}
        r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            headers=headers,
            data=data,
            timeout=20,
        )
        if r.status_code != 200:
            log(f"[social] reddit oauth failed status={r.status_code}")
            return ""
        j = r.json() if r.content else {}
        tok = str(j.get("access_token") or "")
        expires_in = float(j.get("expires_in") or 0.0)
        if tok and expires_in > 0:
            # shave 10s for safety
            _REDDIT_TOKEN_CACHE["access_token"] = tok
            _REDDIT_TOKEN_CACHE["expires_at"] = now + max(0.0, expires_in - 10.0)
        return tok
    except Exception as e:
        log(f"[social] reddit oauth exception: {e}")
        return ""


def _reddit_get_json(path: str, params: dict | None = None) -> Any:
    """GET JSON from Reddit OAuth API."""
    tok = _reddit_oauth_token()
    if not tok:
        return None
    headers = {
        "Authorization": f"bearer {tok}",
        "User-Agent": REDDIT_USER_AGENT or "aion/1.0",
    }
    url = "https://oauth.reddit.com" + path
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=20)
        if r.status_code != 200:
            log(f"[social] reddit GET failed {path} status={r.status_code}")
            return None
        return r.json() if r.content else None
    except Exception as e:
        log(f"[social] reddit GET exception {path}: {e}")
        return None


def _fetch_reddit_wsb_posts(*, sort: str = "hot", limit: int = 100, pages: int = 2, min_score: int = 5) -> list[dict]:
    """Fetch WSB posts (optionally paged).

    sort: hot | new | top | rising
    """
    sort = (sort or "hot").strip().lower()
    if sort not in {"hot", "new", "top", "rising"}:
        sort = "hot"

    out: list[dict] = []
    after = None
    for _ in range(max(1, int(pages))):
        params = {"limit": int(limit)}
        if sort == "top":
            # last 24h catches the current tape vibe without becoming archaeology
            params["t"] = "day"
        if after:
            params["after"] = after

        j = _reddit_get_json(f"/r/wallstreetbets/{sort}", params=params)
        data = j.get("data") if isinstance(j, dict) else None
        children = data.get("children") if isinstance(data, dict) else None
        if not isinstance(children, list) or not children:
            break

        for ch in children:
            d = ch.get("data") if isinstance(ch, dict) else None
            if not isinstance(d, dict):
                continue
            score = int(_safe_float(d.get("score"), 0))
            if score < int(min_score):
                continue
            # Keep text fields small-ish and deterministic
            title = str(d.get("title") or "")[:400]
            body = str(d.get("selftext") or "")[:2000]
            out.append(
                {
                    "id": str(d.get("id") or ""),
                    "name": str(d.get("name") or ""),
                    "created_utc": _safe_float(d.get("created_utc"), 0.0),
                    "score": score,
                    "num_comments": int(_safe_float(d.get("num_comments"), 0)),
                    "author": str(d.get("author") or ""),
                    "permalink": str(d.get("permalink") or ""),
                    "title": title,
                    "text": body,
                    "url": str(d.get("url") or ""),
                    "source": "reddit",
                    "subreddit": "wallstreetbets",
                    "kind": "post",
                }
            )

        after = str(data.get("after") or "") if isinstance(data, dict) else ""
        if not after:
            break

    # Dedup by id
    seen = set()
    deduped: list[dict] = []
    for p in out:
        pid = str(p.get("id") or "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        deduped.append(p)
    return deduped


def _fetch_reddit_wsb_comments(post_fullname: str, *, limit: int = 30, min_score: int = 2) -> list[dict]:
    """Fetch a small slice of top-level comments for a given post fullname (t3_xxx)."""
    if not post_fullname:
        return []
    params = {
        "link_id": post_fullname,
        "limit": int(limit),
        "sort": "top",
        "raw_json": 1,
    }
    j = _reddit_get_json("/r/wallstreetbets/comments", params=params)
    data = j.get("data") if isinstance(j, dict) else None
    children = data.get("children") if isinstance(data, dict) else None
    if not isinstance(children, list):
        return []

    out: list[dict] = []
    for ch in children:
        d = ch.get("data") if isinstance(ch, dict) else None
        if not isinstance(d, dict):
            continue
        score = int(_safe_float(d.get("score"), 0))
        if score < int(min_score):
            continue
        body = str(d.get("body") or "")[:1500]
        if not body:
            continue
        out.append(
            {
                "id": str(d.get("id") or ""),
                "name": str(d.get("name") or ""),
                "created_utc": _safe_float(d.get("created_utc"), 0.0),
                "score": score,
                "author": str(d.get("author") or ""),
                "body": body,
                "source": "reddit",
                "subreddit": "wallstreetbets",
                "kind": "comment",
                "link_id": post_fullname,
            }
        )

    return out


def _fetch_reddit() -> List[Dict[str, Any]]:
    """Compatibility wrapper: returns a unified list of WSB items (posts + optional comments)."""
    # Knobs (env)
    sort = os.getenv("AION_WSB_SORT", "hot")
    pages = int(_safe_float(os.getenv("AION_WSB_PAGES", "2"), 2))
    max_posts = int(_safe_float(os.getenv("AION_WSB_MAX_POSTS", "120"), 120))
    min_post_score = int(_safe_float(os.getenv("AION_WSB_MIN_POST_SCORE", "5"), 5))

    include_comments = str(os.getenv("AION_WSB_INCLUDE_COMMENTS", "1") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    max_comments_per_post = int(_safe_float(os.getenv("AION_WSB_MAX_COMMENTS_PER_POST", "25"), 25))
    min_comment_score = int(_safe_float(os.getenv("AION_WSB_MIN_COMMENT_SCORE", "2"), 2))

    posts = _fetch_reddit_wsb_posts(sort=sort, limit=min(100, max_posts), pages=pages, min_score=min_post_score)
    posts = posts[: max(0, int(max_posts))]

    items: list[dict] = list(posts)
    if include_comments:
        # Only pull comments for the most relevant posts (by score)
        top = sorted(posts, key=lambda x: int(_safe_float(x.get("score"), 0)), reverse=True)
        top = top[: max(1, min(25, len(top)))]
        for p in top:
            fullname = str(p.get("name") or "")  # e.g. t3_abc123
            items.extend(_fetch_reddit_wsb_comments(fullname, limit=max_comments_per_post, min_score=min_comment_score))

    return items


def _fetch_twitter() -> List[Dict[str, Any]]:
    '''Twitter is intentionally disabled (NO TWITTER).'''
    return []


# =====================================================================
# Fallback FinViz
# =====================================================================

def _fallback_sources() -> List[Dict[str, Any]]:
    try:
        url = "https://finviz.com/api/news.ashx"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        js = r.json()
    except Exception:
        return []

    out = []
    for itm in js:
        text = itm.get("title") or ""
        tickers = extract_tickers(text)
        sent = score_sentiment(text)

        out.append({
            "source": "finviz",
            "text": text,
            "tickers": tickers,
            "sentiment": sent,
            "buzz": 1,
            "timestamp": itm.get("date"),
        })
    return out


# =====================================================================
# SOCIAL INTEL CORE
# =====================================================================

def build_social_sentiment() -> Dict[str, Any]:
    """Fetch, score, and persist social sentiment (WSB-only)."""

    # Caching: avoid hammering Reddit and keep artifacts stable.
    cache_min = _env_int("AION_WSB_CACHE_MINUTES", 3)
    force = os.getenv("AION_WSB_FORCE", "0").strip().lower() in {"1","true","yes","y","on"}
    try:
        if (not force) and CACHE_FILE.exists() and cache_min > 0:
            age_s = (datetime.now(timezone.utc) - datetime.fromtimestamp(CACHE_FILE.stat().st_mtime, tz=timezone.utc)).total_seconds()
            if age_s < cache_min * 60:
                cached = read_json(CACHE_FILE, {})
                if isinstance(cached, dict) and cached.get("status") == "ok":
                    log(f"[social] 💤 using cached social_intel (age={age_s:.0f}s)")
                    return cached
    except Exception:
        pass

    log("💬 Fetching social sentiment (Reddit /r/wallstreetbets)…")

    posts = _fetch_reddit()
    sources_used = ["reddit_wsb"] if posts else []

    if not posts:
        out = {"status": "empty", "sources": [], "updated": 0, "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
        try:
            atomic_write_json(CACHE_FILE, out)
        except Exception:
            pass
        log("[social] ⚠️ No WSB posts/comments fetched.")
        return out

    # =====================================================================
    # Novelty (FIXED timezone issue)
    # =====================================================================

    def novelty(ts: Any) -> float:
        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts).replace(tzinfo=TIMEZONE)
            else:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=TIMEZONE)
        except Exception:
            return 0.0

        age_hours = (datetime.now(TIMEZONE) - dt).total_seconds() / 3600
        return max(0.0, min(1.0, math.exp(-age_hours / 12)))

    # =====================================================================
    # Aggregate per-symbol
    # =====================================================================

    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for p in posts:
        tickers = p.get("tickers") or []
        for sym in tickers:
            sym_u = sym.upper()
            clusters.setdefault(sym_u, []).append(p)

    rolling = _read_rolling()
    if not rolling:
        log("[social] ⚠️ No rolling.json.gz — cannot store symbol intel.")
        return {"status": "no_rolling"}

    updated = 0

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue

        sym_u = sym.upper()
        plist = clusters.get(sym_u) or []

        if not plist:
            node["social"] = {
                "sentiment": 0.0,
                "buzz": 0,
                "novelty": 0.0,
                "heat_score": 0.0,
                "last_updated": datetime.now(TIMEZONE).isoformat(),
            }
            rolling[sym] = node
            updated += 1
            continue

        sentiments = [p["sentiment"] for p in plist]
        buzzes = [safe_float(p.get("buzz", 1)) for p in plist]
        novs = [novelty(p.get("timestamp")) for p in plist]

        avg_sent = statistics.mean(sentiments) if sentiments else 0.0
        total_buzz = sum(buzzes)
        avg_nov = statistics.mean(novs) if novs else 0.0
        heat = avg_sent * math.log1p(total_buzz) * (1 + avg_nov)

        node["social"] = {
            "sentiment": float(avg_sent),
            "buzz": int(total_buzz),
            "novelty": float(avg_nov),
            "heat_score": float(round(heat, 4)),
            "last_updated": datetime.now(TIMEZONE).isoformat(),
        }
        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"[social] Updated social sentiment for {updated} symbols.")

    # =====================================================================
    # Global social_intel.json
    # =====================================================================

    try:
        market_sent = statistics.mean([p["sentiment"] for p in posts]) if posts else 0.0
        buzz_index = sum(p.get("buzz", 1) for p in posts)

        trending = sorted(
            clusters.items(),
            key=lambda kv: sum(p.get("buzz", 1) for p in kv[1]),
            reverse=True
        )[:20]

        intel = {
            "timestamp": datetime.now(TIMEZONE).isoformat(),
            "market_social_sentiment": market_sent,
            "buzz_index": buzz_index,
            "sources_used": sources_used,
            "top_trending_tickers": [
                {"symbol": sym, "buzz": sum(p.get("buzz", 1) for p in plist)}
                for sym, plist in trending
            ],
        }

        CACHE_FILE.write_text(json.dumps(intel, indent=2), encoding="utf-8")
        log("[social] 🧠 Updated social_intel.json")

    except Exception as e:
        log(f"[social] ⚠️ Failed writing social_intel.json: {e}")

    return {"status": "ok", "updated": updated, "sources": sources_used}

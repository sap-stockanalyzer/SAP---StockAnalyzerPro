# backend/services/news_fetcher.py
"""
News Fetcher v1.1 — Marketaux market-wide collector (NO per-ticker loops)

What this file DOES:
  ✅ Makes a small number of Marketaux calls (market-wide)
  ✅ Normalizes raw articles into a stable internal schema
  ✅ Writes raw batches to disk (temporary fallback) OR hands off to news_cache if present
  ✅ Returns a compact run summary for logging/monitoring
  ✅ NEW: Reads DAILY call budgets from env and auto-splits by time-of-day (Option A)
  ✅ NEW: Weekend "batch once per day" protection (prevents triple spending)

What this file does NOT do:
  ❌ No per-symbol fetching
  ❌ No "fetch universe" fan-out
  ❌ No ML feature building
  ❌ No sentiment/buzz summaries (that belongs in news_intel / news_brain)

Design principle:
  News fetching should be cheap, stable, and deterministic-ish.
  Nightly job should NOT depend on live API calls.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests

from backend.core.config import PATHS, TIMEZONE
from utils.logger import log, warn, error
from utils.time_utils import ts

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# -------------------------------------------------------------------
# Marketaux config
# -------------------------------------------------------------------

MARKETAUX_BASE = "https://api.marketaux.com/v1/news/all"

DEFAULT_LANGUAGE = "en"

# Conservative defaults (free tier friendly)
DEFAULT_TIMEOUT_SECS = 12
DEFAULT_MAX_ARTICLES_PER_CALL = 50  # depends on Marketaux plan; keep conservative

# Env var name for key
MARKETAUX_API_KEY_ENV = "MARKETAUX_API_KEY"


# -------------------------------------------------------------------
# Env helpers + budgets (Step 2)
# -------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


# Total DAILY caps (your free-tier budget knobs)
DAILY_CALLS_HIGH_IMPACT = _env_int("NEWS_CALLS_HIGH_IMPACT", 55)
DAILY_CALLS_TOP_MOVERS  = _env_int("NEWS_CALLS_TOP_MOVERS", 30)
DAILY_CALLS_LATEST      = _env_int("NEWS_CALLS_LATEST_MARKET", 15)

# Optional runtime knobs
ENV_LIMIT_PER_CALL = _env_int("NEWS_LIMIT_PER_CALL", DEFAULT_MAX_ARTICLES_PER_CALL)
ENV_SLEEP_BETWEEN_CALLS = _env_float("NEWS_SLEEP_BETWEEN_CALLS", 0.2)

# Tranche fractions (must sum to ~1.0)
FRAC_MORNING   = _env_float("NEWS_TRANCHE_MORNING_FRAC", 0.40)
FRAC_MIDDAY    = _env_float("NEWS_TRANCHE_MIDDAY_FRAC", 0.35)
FRAC_AFTERNOON = _env_float("NEWS_TRANCHE_AFTERNOON_FRAC", 0.25)

# Weekend mode
WEEKEND_BATCH = str(os.getenv("NEWS_WEEKEND_BATCH", "true")).strip().lower() in ("1", "true", "yes", "y", "on")


# -------------------------------------------------------------------
# Paths (raw fallback cache + weekend marker)
# -------------------------------------------------------------------

def _root() -> Path:
    return Path(PATHS.get("root") or Path("."))


def _news_root() -> Path:
    base = PATHS.get("news")
    if not base:
        base = (PATHS.get("data_root") or _root() / "data") / "raw" / "news"
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _raw_dir() -> Path:
    d = _news_root() / "raw_market"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _daily_run_tag() -> str:
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d")


def _weekend_marker_path(day_tag: str) -> Path:
    # Stored under raw_market/<YYYY-MM-DD>/.weekend_batch_done
    out_dir = _raw_dir() / day_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / ".weekend_batch_done"


def _weekend_batch_already_ran(day_tag: str) -> bool:
    return _weekend_marker_path(day_tag).exists()


def _mark_weekend_batch_done(day_tag: str) -> None:
    p = _weekend_marker_path(day_tag)
    try:
        p.write_text(ts(), encoding="utf-8")
    except Exception:
        pass


# -------------------------------------------------------------------
# API key / auth
# -------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    key = os.getenv(MARKETAUX_API_KEY_ENV, "").strip()
    if key:
        return key
    warn("[news_fetcher] MARKETAUX_API_KEY not set; news fetch is disabled.")
    return None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _iso_utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _fmt_marketaux_ts(dt: datetime) -> str:
    """
    Marketaux expects timestamps formatted as YYYY-MM-DDTHH:MM:SS
    (no timezone suffix, no milliseconds).
    """
    return dt.replace(microsecond=0, tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")


def _parse_dt(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        ss = str(s).replace("Z", "+00:00")
        return datetime.fromisoformat(ss)
    except Exception:
        return None


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _stable_article_id(raw: Dict[str, Any]) -> str:
    """
    A stable ID for dedupe later.
    Prefer URL; fallback to title+source+published_at.
    """
    url = str(raw.get("url") or "").strip()
    if url:
        return _sha1(url.lower())

    title = str(raw.get("title") or raw.get("headline") or "").strip()
    source = str(raw.get("source") or raw.get("source_name") or "").strip()
    published = str(raw.get("published_at") or raw.get("published") or "").strip()
    return _sha1(f"{title}|{source}|{published}".lower())


def _extract_symbols(raw: Dict[str, Any]) -> List[str]:
    """
    Best-effort ticker/entity extraction.
    """
    syms: set[str] = set()

    entities = raw.get("entities")
    if isinstance(entities, list):
        for e in entities:
            if not isinstance(e, dict):
                continue
            sym = e.get("symbol") or e.get("ticker") or e.get("code")
            if sym and isinstance(sym, str):
                s = sym.upper().strip()
                if s and len(s) <= 10 and " " not in s:
                    syms.add(s)

    symbols = raw.get("symbols")
    if isinstance(symbols, list):
        for sym in symbols:
            if sym and isinstance(sym, str):
                s = sym.upper().strip()
                if s and len(s) <= 10 and " " not in s:
                    syms.add(s)

    return sorted(syms)


def _normalize_article(raw: Dict[str, Any], fetch_tag: str) -> Dict[str, Any]:
    """
    Normalize a Marketaux article into a stable internal schema.
    """
    published_at = raw.get("published_at") or raw.get("published")
    published_dt = _parse_dt(published_at)

    title = raw.get("title") or raw.get("headline") or ""
    source = raw.get("source") or raw.get("source_name") or ""

    sentiment = raw.get("sentiment") or raw.get("sentiment_score")
    relevance = raw.get("relevance_score") or raw.get("relevance")

    return {
        "article_id": _stable_article_id(raw),
        "fetch_tag": fetch_tag,
        "fetched_at": _iso_utc_now(),
        "published_at": published_dt.isoformat() if published_dt else None,

        "source": str(source) if source is not None else None,
        "headline": str(title) if title is not None else None,
        "description": raw.get("description") or "",
        "url": raw.get("url"),

        "sentiment_score": sentiment,
        "relevance_score": relevance,

        "symbols": _extract_symbols(raw),

        "raw": raw,
    }


def _write_raw_batch_fallback(fetch_tag: str, articles: List[Dict[str, Any]]) -> Optional[Path]:
    if not articles:
        return None

    day = _daily_run_tag()
    out_dir = _raw_dir() / day
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{fetch_tag}_{datetime.utcnow().strftime('%H%M%S')}.json"
    path = out_dir / fname

    payload = {
        "meta": {"fetch_tag": fetch_tag, "saved_at": ts(), "count": len(articles)},
        "articles": articles,
    }

    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    except Exception as e:
        error(f"[news_fetcher] Failed writing raw batch fallback: {path}", e)
        return None


def _handoff_to_cache_if_available(fetch_tag: str, articles: List[Dict[str, Any]]) -> bool:
    if not articles:
        return True
    try:
        from backend.services import news_cache  # type: ignore
        if hasattr(news_cache, "ingest_articles"):
            news_cache.ingest_articles(articles, source_tag=fetch_tag)
            return True
        return False
    except Exception:
        return False


# -------------------------------------------------------------------
# HTTP request
# -------------------------------------------------------------------

def _request_marketaux(params: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    key = _get_api_key()
    if not key:
        return 0, {}

    p = dict(params)
    p["api_token"] = key

    try:
        resp = requests.get(MARKETAUX_BASE, params=p, timeout=DEFAULT_TIMEOUT_SECS)
        code = int(resp.status_code)

        if code != 200:
            body = resp.text[:250] if resp.text else ""
            warn(f"[news_fetcher] Marketaux request failed: {code} {body}")
            return code, {}

        return code, resp.json() if resp.content else {}
    except Exception as e:
        warn(f"[news_fetcher] Marketaux request exception: {e}")
        return 0, {}


def _extract_articles(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") or payload.get("articles") or []
    if not isinstance(data, list):
        return []
    return [a for a in data if isinstance(a, dict)]


# -------------------------------------------------------------------
# Market-wide fetch modes
# -------------------------------------------------------------------

def fetch_latest_market_news(
    hours_back: int = 24,
    limit: int = DEFAULT_MAX_ARTICLES_PER_CALL,
    language: str = DEFAULT_LANGUAGE,
    page: int = 1,
) -> List[Dict[str, Any]]:
    fetch_tag = "latest_market"
    end = datetime.utcnow()
    start = end - timedelta(hours=max(1, int(hours_back)))

    params = {
        "language": language,
        "published_after": _fmt_marketaux_ts(start),
        "published_before": _fmt_marketaux_ts(end),
        "limit": int(limit),
        "page": int(page),
        "sort": "published_at",
    }

    code, payload = _request_marketaux(params)
    if code != 200:
        return []

    raw_articles = _extract_articles(payload)
    return [_normalize_article(a, fetch_tag=fetch_tag) for a in raw_articles]


def fetch_high_impact_news(
    hours_back: int = 48,
    limit: int = DEFAULT_MAX_ARTICLES_PER_CALL,
    language: str = DEFAULT_LANGUAGE,
    page: int = 1,
) -> List[Dict[str, Any]]:
    fetch_tag = "high_impact"
    end = datetime.utcnow()
    start = end - timedelta(hours=max(1, int(hours_back)))

    params = {
        "language": language,
        "published_after": _fmt_marketaux_ts(start),
        "published_before": _fmt_marketaux_ts(end),
        "limit": int(limit),
        "page": int(page),
        "sort": "relevance_score",
    }

    code, payload = _request_marketaux(params)
    if code != 200:
        return []

    raw_articles = _extract_articles(payload)
    return [_normalize_article(a, fetch_tag=fetch_tag) for a in raw_articles]


def fetch_top_movers_news(
    hours_back: int = 24,
    limit: int = DEFAULT_MAX_ARTICLES_PER_CALL,
    language: str = DEFAULT_LANGUAGE,
    page: int = 1,
) -> List[Dict[str, Any]]:
    fetch_tag = "top_movers"
    end = datetime.utcnow()
    start = end - timedelta(hours=max(1, int(hours_back)))

    params = {
        "language": language,
        "published_after": _fmt_marketaux_ts(start),
        "published_before": _fmt_marketaux_ts(end),
        "limit": int(limit),
        "page": int(page),
        "sort": "published_at",
    }

    code, payload = _request_marketaux(params)
    if code != 200:
        return []

    raw_articles = _extract_articles(payload)
    return [_normalize_article(a, fetch_tag=fetch_tag) for a in raw_articles]


# -------------------------------------------------------------------
# Step 3: auto tranche logic (time-of-day split)
# -------------------------------------------------------------------

def _compute_tranche_frac(now_local: datetime) -> float:
    """
    Option A: split daily budget by time-of-day.

    Morning:   < 11:00
    Midday:    11:00 - 13:59
    Afternoon: >= 14:00

    Weekend: if WEEKEND_BATCH enabled, use full daily budget once.
    """
    is_weekend = now_local.weekday() >= 5
    if is_weekend and WEEKEND_BATCH:
        return 1.0

    hour = int(now_local.hour)
    if hour < 11:
        return float(FRAC_MORNING)
    if hour < 14:
        return float(FRAC_MIDDAY)
    return float(FRAC_AFTERNOON)


def _scale_calls(daily_calls: int, frac: float) -> int:
    if daily_calls <= 0:
        return 0
    if frac <= 0:
        return 0
    # If you run once per window (morning/midday/afternoon), these add up to daily_calls.
    return max(0, int(round(daily_calls * frac)))


# -------------------------------------------------------------------
# Public entrypoint
# -------------------------------------------------------------------

def run_news_fetch(
    calls_high_impact: Optional[int] = None,
    calls_top_movers: Optional[int] = None,
    calls_latest_market: Optional[int] = None,
    limit_per_call: Optional[int] = None,
    sleep_between_calls: Optional[float] = None,
    hours_back_high_impact: int = 48,
    hours_back_top_movers: int = 24,
    hours_back_latest_market: int = 24,
) -> Dict[str, Any]:
    """
    Collector-style fetch run.

    If calls_* are None, we auto-compute using:
      - env daily budgets (Step 2)
      - time-of-day tranche split (Step 3)

    Weekend protection:
      - If WEEKEND_BATCH=true, we do full budget once per day and skip subsequent weekend runs.
    """
    now_local = datetime.now(TIMEZONE)
    day_tag = _daily_run_tag()

    # Weekend once-per-day guard
    is_weekend = now_local.weekday() >= 5
    if is_weekend and WEEKEND_BATCH and _weekend_batch_already_ran(day_tag):
        log(f"[news_fetcher] ⏭️ Weekend batch already ran for {day_tag}; skipping.")
        return {
            "status": "skipped",
            "reason": "weekend_batch_already_ran",
            "total_calls": 0,
            "total_articles": 0,
            "written_files": [],
            "cache_handoff": True,
            "per_mode": {},
        }

    tranche_frac = _compute_tranche_frac(now_local)

    # Auto compute call counts unless explicitly provided
    if calls_high_impact is None:
        calls_high_impact = _scale_calls(DAILY_CALLS_HIGH_IMPACT, tranche_frac)
    if calls_top_movers is None:
        calls_top_movers = _scale_calls(DAILY_CALLS_TOP_MOVERS, tranche_frac)
    if calls_latest_market is None:
        calls_latest_market = _scale_calls(DAILY_CALLS_LATEST, tranche_frac)

    if limit_per_call is None:
        limit_per_call = int(ENV_LIMIT_PER_CALL)
    if sleep_between_calls is None:
        sleep_between_calls = float(ENV_SLEEP_BETWEEN_CALLS)

    log(
        "[news_fetcher] 🚀 run_news_fetch (market-wide) "
        f"tranche_frac={tranche_frac:.2f} "
        f"high_impact={calls_high_impact}, movers={calls_top_movers}, latest={calls_latest_market}, "
        f"limit_per_call={limit_per_call}"
    )

    key = _get_api_key()
    if not key:
        return {
            "status": "disabled",
            "reason": "missing_api_key",
            "total_calls": 0,
            "total_articles": 0,
            "written_files": [],
            "cache_handoff": False,
            "per_mode": {},
        }

    written_files: List[str] = []
    total_articles = 0
    total_calls = 0
    cache_handoff_ok = True

    def _ingest(fetch_tag: str, articles: List[Dict[str, Any]]) -> int:
        nonlocal cache_handoff_ok
        if not articles:
            return 0
        ok = _handoff_to_cache_if_available(fetch_tag, articles)
        if not ok:
            cache_handoff_ok = False
            path = _write_raw_batch_fallback(fetch_tag, articles)
            if path:
                written_files.append(str(path))
        return len(articles)

    per_mode: Dict[str, Any] = {}

    # ---- High impact ----
    per_mode["high_impact"] = {"calls": 0, "articles": 0}
    for i in range(max(0, int(calls_high_impact))):
        arts = fetch_high_impact_news(hours_back=hours_back_high_impact, limit=int(limit_per_call), page=(i + 1))
        n = _ingest("high_impact", arts)
        per_mode["high_impact"]["calls"] += 1
        per_mode["high_impact"]["articles"] += n
        total_calls += 1
        total_articles += n
        if sleep_between_calls and sleep_between_calls > 0:
            time.sleep(float(sleep_between_calls))

    # ---- Top movers ----
    per_mode["top_movers"] = {"calls": 0, "articles": 0}
    for i in range(max(0, int(calls_top_movers))):
        arts = fetch_top_movers_news(hours_back=hours_back_top_movers, limit=int(limit_per_call), page=(i + 1))
        n = _ingest("top_movers", arts)
        per_mode["top_movers"]["calls"] += 1
        per_mode["top_movers"]["articles"] += n
        total_calls += 1
        total_articles += n
        if sleep_between_calls and sleep_between_calls > 0:
            time.sleep(float(sleep_between_calls))

    # ---- Latest market ----
    per_mode["latest_market"] = {"calls": 0, "articles": 0}
    for i in range(max(0, int(calls_latest_market))):
        arts = fetch_latest_market_news(hours_back=hours_back_latest_market, limit=int(limit_per_call), page=(i + 1))
        n = _ingest("latest_market", arts)
        per_mode["latest_market"]["calls"] += 1
        per_mode["latest_market"]["articles"] += n
        total_calls += 1
        total_articles += n
        if sleep_between_calls and sleep_between_calls > 0:
            time.sleep(float(sleep_between_calls))

    status = "ok" if total_calls > 0 else "empty"

    # Mark weekend batch done (only after successful run attempt)
    if is_weekend and WEEKEND_BATCH:
        _mark_weekend_batch_done(day_tag)

    log(
        f"[news_fetcher] ✅ run_news_fetch complete: calls={total_calls}, articles={total_articles}, "
        f"cache_handoff={'yes' if cache_handoff_ok else 'no (fallback raw writes)'}"
    )

    return {
        "status": status,
        "total_calls": int(total_calls),
        "total_articles": int(total_articles),
        "written_files": written_files,
        "cache_handoff": bool(cache_handoff_ok),
        "per_mode": per_mode,
        "meta": {
            "day": day_tag,
            "tranche_frac": float(tranche_frac),
            "weekend_batch": bool(WEEKEND_BATCH),
        },
        "note": "Market-wide collection. Nightly job should NOT run this; intraday scheduler should.",
    }


# -------------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------------

if __name__ == "__main__":
    # When run as: python -m backend.services.news_fetcher
    # we do an AUTO tranche run by default (Option A).
    out = run_news_fetch()
    print(json.dumps(out, indent=2))

"""
admin_keys.py (ROOT)

Single source of truth for secrets / API keys across backend + dt_backend.
- Reads from environment variables only.
- Do NOT hardcode secrets here.
"""

from __future__ import annotations

import os

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()

# Supabase
SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _env("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = _env("SUPABASE_ANON_KEY")
SUPABASE_BUCKET = _env("SUPABASE_BUCKET", "aion")

# Alpaca / trading
ALPACA_API_KEY_ID = _env("ALPACA_API_KEY_ID")
ALPACA_API_SECRET_KEY = _env("ALPACA_API_SECRET_KEY")
ALPACA_PAPER_BASE_URL = _env("ALPACA_PAPER_BASE_URL")

# Legacy aliases used in a few places
ALPACA_KEY = ALPACA_API_KEY_ID
ALPACA_SECRET = ALPACA_API_SECRET_KEY

# Macro / data / news providers (kept for parity with existing .env tooling)
FRED_API = _env("FRED_API")
PERIGON_KEY = _env("PERIGON_KEY")
MARKETAUX_API_KEY = _env("MARKETAUX_API_KEY")
FINNHUB_API_KEY = _env("FINNHUB_API_KEY")
NEWSAPI_KEY = _env("NEWSAPI_KEY")
RSS2JSON_KEY = _env("RSS2JSON_KEY")
TWITTER_BEARER = _env("TWITTER_BEARER")

# Reddit
REDDIT_CLIENT_ID = _env("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = _env("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = _env("REDDIT_USER_AGENT")

# Massive (if used)
MASSIVE_API = _env("MASSIVE_API")
MASSIVE_S3API = _env("MASSIVE_S3API")
MASSIVE_S3API_SECRET = _env("MASSIVE_S3API_SECRET")
MASSIVE_S3_ENDPOINT = _env("MASSIVE_S3_ENDPOINT")
MASSIVE_S3_BUCKET = _env("MASSIVE_S3_BUCKET")

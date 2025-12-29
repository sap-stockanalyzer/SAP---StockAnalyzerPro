"""
backend/core/config.py (SHIM)

This file is now a thin compatibility layer.

Canonical sources of truth live at repo root:
- config.py      (paths)
- settings.py    (knobs/timezone defaults)
- admin_keys.py  (secrets)

Do NOT add new configuration here.
"""

from __future__ import annotations

from config import ROOT, DATA_ROOT, PATHS, get_path  # type: ignore
from settings import TIMEZONE  # type: ignore
from admin_keys import (  # type: ignore
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    ALPACA_PAPER_BASE_URL,
    ALPACA_KEY,
    ALPACA_SECRET,
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_ANON_KEY,
    SUPABASE_BUCKET,
)

# Backward-compat aliases some modules expect
ALPACA_API_SECRET_KEY = ALPACA_API_SECRET_KEY
ALPACA_API_SECRET = ALPACA_API_SECRET_KEY

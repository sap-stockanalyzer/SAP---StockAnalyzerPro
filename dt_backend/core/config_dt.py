"""
dt_backend/core/config_dt.py (SHIM)

This file is now a thin compatibility layer.

Canonical sources of truth live at repo root:
- config.py      (paths)
- settings.py    (knobs/timezone defaults)
- admin_keys.py  (secrets)

Do NOT add new configuration here.
"""

from __future__ import annotations

from config import ROOT, DT_PATHS, ensure_dt_dirs, get_dt_path  # type: ignore
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

"""dt_backend/core/time_override_dt.py

Replay/backtest needs *deterministic time*.

In live trading, modules call datetime.now(timezone.utc) directly.
In replay/backtest, that breaks everything because bars are historical.

We solve it with a tiny convention:

    DT_NOW_UTC = ISO8601 timestamp (e.g. 2025-12-15T14:35:00Z)

If DT_NOW_UTC is set, the engine treats that as "now" for:
  - trade logs timestamps
  - broker fills timestamps (local sim)
  - bracket state timestamps
  - time-stops / scratch / EOD flatten logic

When DT_NOW_UTC is not set, behavior is unchanged (real time).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        txt = (s or "").strip()
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def now_utc() -> datetime:
    """Return the effective UTC 'now'.

    If DT_NOW_UTC is set, returns that timestamp; otherwise real now.
    """
    override = (os.getenv("DT_NOW_UTC", "") or "").strip()
    dt = _parse_iso(override) if override else None
    return dt or datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().isoformat().replace("+00:00", "Z")


# Back-compat aliases
def utc_iso() -> str:
    """Alias for now_utc_iso() used by older modules."""
    return now_utc_iso()

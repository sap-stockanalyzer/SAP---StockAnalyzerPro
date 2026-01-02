"""dt_backend/core/micro_regime_dt.py — v1.0 (Phase 2.5)

Time-of-day micro-regimes.

Humans trade differently at 09:31 than at 12:15. Most bots pretend time
doesn't exist and then act surprised when lunchtime chop eats them.

We expose a small, stable schema:
    {
      "label": "PREMARKET"|"OPEN"|"MID"|"LUNCH"|"AFTERNOON"|"POWER_HOUR"|"CLOSED",
      "allow_trading": bool,
      "ts": "...Z"
    }

Defaults
--------
  • Stand down during LUNCH unless DT_ALLOW_LUNCH_TRADES=1
  • Stand down when market is CLOSED/PREMARKET (this can be relaxed later)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ny_tz():
    if ZoneInfo is not None:
        return ZoneInfo("America/New_York")
    return timezone.utc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def compute_micro_regime(now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    now_utc = now_utc or datetime.now(timezone.utc)
    ny = _ny_tz()
    now = now_utc.astimezone(ny)

    # Session times (no holiday calendar here)
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if now < open_t:
        label = "PREMARKET"
    elif now >= close_t:
        label = "CLOSED"
    else:
        mins = (now - open_t).total_seconds() / 60.0
        if mins < 45:
            label = "OPEN"
        elif mins < 120:
            label = "MID"
        elif mins < 240:
            label = "LUNCH"
        elif mins < 330:
            label = "AFTERNOON"
        else:
            label = "POWER_HOUR"

    allow_lunch = _env_bool("DT_ALLOW_LUNCH_TRADES", False)
    allow_premarket = _env_bool("DT_ALLOW_PREMARKET_TRADES", False)

    allow = True
    if label in {"CLOSED"}:
        allow = False
    elif label == "PREMARKET":
        allow = bool(allow_premarket)
    elif label == "LUNCH":
        allow = bool(allow_lunch)

    return {"label": label, "allow_trading": bool(allow), "ts": _utc_iso()}

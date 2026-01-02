"""dt_backend/core/bars_fetch_state_dt.py

Small persistence layer for intraday bars fetch throttling.

We keep a single JSON state file under da_brains/intraday so that:
  • multiple processes don't re-fetch the same time window over and over
  • the live bars loop is naturally rate-limit friendly

This is *not* a perfect scheduler; it's a pragmatic storm-preventer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config_dt import DT_PATHS


def _state_path() -> Path:
    base = Path(DT_PATHS.get("da_brains") or Path("."))
    return base / "intraday" / ".dt_bars_fetch_state.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_state() -> Dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def write_state(state: Dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = dict(state or {})
    payload["_ts"] = _to_iso(_utc_now())
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        return


def get_last_end(timeframe: str) -> Optional[datetime]:
    st = read_state()
    key = f"last_end_{timeframe}"
    v = st.get(key)
    return _parse_iso(str(v)) if v else None


def set_last_end(timeframe: str, dt: datetime) -> None:
    st = read_state()
    key = f"last_end_{timeframe}"
    st[key] = _to_iso(dt)
    write_state(st)

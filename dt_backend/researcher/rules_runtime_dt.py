# dt_backend/researcher/rules_runtime_dt.py — Phase 9
"""Runtime application of researcher filter rules (gating only).

Rules are loaded from a JSON file path set via:
  DT_RESEARCHER_RULES_PATH=/abs/path/to/rules.json

If unset or unreadable, this module is a no-op.

Supported rule kinds (v1)
------------------------
* DISABLE_WHEN: disable bot when (REGIME==value) or (MICRO==value)
* MIN_FEATURE: require a feature >= min
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "data": None}


def _rules_path() -> Optional[Path]:
    p = (os.getenv("DT_RESEARCHER_RULES_PATH", "") or "").strip()
    if not p:
        return None
    return Path(p)


def load_rules() -> Dict[str, Any]:
    p = _rules_path()
    if not p or not p.exists():
        return {}
    try:
        mtime = p.stat().st_mtime
        if _CACHE.get("path") == str(p) and _CACHE.get("mtime") == mtime and isinstance(_CACHE.get("data"), dict):
            return _CACHE["data"]
        raw = json.loads(p.read_text(encoding="utf-8"))
        data = raw if isinstance(raw, dict) else {}
        _CACHE.update({"path": str(p), "mtime": mtime, "data": data})
        return data
    except Exception:
        return {}


def bot_allowed(
    *,
    bot: str,
    regime: str,
    micro: str,
    features_dt: Dict[str, Any] | None,
) -> Tuple[bool, str]:
    """Return (allowed, reason)."""

    data = load_rules()
    rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, list) or not rules:
        return True, ""

    b = str(bot).upper()
    reg = str(regime).upper()
    mic = str(micro).upper()
    feats = features_dt if isinstance(features_dt, dict) else {}

    for r in rules:
        if not isinstance(r, dict):
            continue
        if str(r.get("bot") or "").upper() not in {b, "*", "ALL"}:
            continue
        kind = str(r.get("kind") or "").upper()
        rid = str(r.get("id") or "rule")

        if kind == "DISABLE_WHEN":
            when = r.get("when") if isinstance(r.get("when"), dict) else {}
            dim = str(when.get("dim") or "").upper()
            val = str(when.get("value") or "").upper()
            if dim == "REGIME" and val and reg == val:
                return False, f"researcher:{rid}"
            if dim == "MICRO" and val and mic == val:
                return False, f"researcher:{rid}"

        if kind == "MIN_FEATURE":
            params = r.get("params") if isinstance(r.get("params"), dict) else {}
            feat = str(params.get("feature") or "")
            mn = params.get("min")
            try:
                mnf = float(mn)
            except Exception:
                mnf = None
            if feat and mnf is not None:
                try:
                    v = float(feats.get(feat) or 0.0)
                except Exception:
                    v = 0.0
                if v < mnf:
                    return False, f"researcher:{rid} {feat}<{mnf}"

    return True, ""

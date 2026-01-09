"""dt_backend.core.position_registry — Strategy-level position ownership registry.

Alpaca positions are account-level. If you run multiple strategies (DT vs Swing,
or multiple DT bots) in the same Alpaca account, they can unintentionally trade
against each other (e.g., DT selling Swing's long).

This module provides a tiny, boring, durable shared registry that tracks
*reserved* quantities per strategy. It is intentionally conservative:

- A strategy only increases its reserved qty when it *knows* it received a fill.
- A strategy only decreases its reserved qty when it *knows* it sold/covered.
- Periodic reconciliation compares reserved totals to broker truth (Alpaca)
  and flags mismatches.

Registry file (JSON)
--------------------
{
  "_meta": {"updated_at": "...Z", "schema": 1},
  "symbols": {
     "AAPL": {
        "reserved": {"DT": 5.0, "SWING": 10.0},
        "alpaca_qty": 15.0,
        "mismatch": false,
        "ts": "...Z"
     },
     ...
  }
}

The system uses this registry as a *safety blanket*:
- DT SELLs are clamped to DT-reserved qty (never touches Swing-reserved shares).
- If reserved_total > alpaca_qty (or alpaca_qty is missing) we can flag mismatch
  and optionally stand down.

No broker keys are required to read/write this registry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _default_path() -> Path:
    # Shared location across strategies (DT + backend swing) if they share PROJECT_ROOT.
    override = (os.getenv("AION_POSITION_REGISTRY_PATH", "") or "").strip()
    if override:
        return Path(override)

    # Prefer DT_TRUTH_DIR/da_brains layout when available.
    dt_truth = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if dt_truth:
        return Path(dt_truth) / "intraday" / "positions" / "position_registry.json"

    # Fall back to a stable path under da_brains if present.
    # We don't import DT_PATHS here to keep this module dependency-light.
    return Path("da_brains") / "shared" / "positions" / "position_registry.json"


@dataclass
class Registry:
    path: Path
    data: Dict[str, Any]


def load_registry(path: Optional[Path] = None) -> Registry:
    p = path or _default_path()
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                d.setdefault("_meta", {})
                d.setdefault("symbols", {})
                if not isinstance(d["symbols"], dict):
                    d["symbols"] = {}
                return Registry(path=p, data=d)
    except Exception:
        pass

    d2: Dict[str, Any] = {"_meta": {"schema": 1, "updated_at": _utc_iso()}, "symbols": {}}
    return Registry(path=p, data=d2)


def save_registry(reg: Registry) -> None:
    try:
        reg.path.parent.mkdir(parents=True, exist_ok=True)
        reg.data.setdefault("_meta", {})
        reg.data["_meta"]["schema"] = 1
        reg.data["_meta"]["updated_at"] = _utc_iso()
        tmp = reg.path.with_suffix(reg.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(reg.data, f, ensure_ascii=False, indent=2)
        tmp.replace(reg.path)
    except Exception:
        # Never crash trading because the registry can't be written.
        pass


def get_reserved_qty(reg: Registry, symbol: str, owner: str) -> float:
    sym = (symbol or "").upper().strip()
    owner = (owner or "").upper().strip() or "DT"
    node = reg.data.get("symbols", {}).get(sym)
    if not isinstance(node, dict):
        return 0.0
    reserved = node.get("reserved")
    if not isinstance(reserved, dict):
        return 0.0
    return _safe_float(reserved.get(owner), 0.0)


def set_reserved_qty(reg: Registry, symbol: str, owner: str, qty: float) -> None:
    sym = (symbol or "").upper().strip()
    if not sym:
        return
    owner = (owner or "").upper().strip() or "DT"

    reg.data.setdefault("symbols", {})
    symbols = reg.data["symbols"]
    node = symbols.get(sym)
    if not isinstance(node, dict):
        node = {}
        symbols[sym] = node

    reserved = node.get("reserved")
    if not isinstance(reserved, dict):
        reserved = {}
    reserved[owner] = float(max(0.0, qty))
    node["reserved"] = reserved
    node.setdefault("alpaca_qty", None)
    node["ts"] = _utc_iso()


def add_reserved_qty(reg: Registry, symbol: str, owner: str, delta: float) -> float:
    cur = get_reserved_qty(reg, symbol, owner)
    new = max(0.0, float(cur) + float(delta))
    set_reserved_qty(reg, symbol, owner, new)
    return new


def reserved_total(reg: Registry, symbol: str) -> float:
    sym = (symbol or "").upper().strip()
    node = reg.data.get("symbols", {}).get(sym)
    if not isinstance(node, dict):
        return 0.0
    reserved = node.get("reserved")
    if not isinstance(reserved, dict):
        return 0.0
    return float(sum(_safe_float(v, 0.0) for v in reserved.values()))


def set_alpaca_qty(reg: Registry, symbol: str, qty: Optional[float]) -> None:
    sym = (symbol or "").upper().strip()
    if not sym:
        return
    reg.data.setdefault("symbols", {})
    symbols = reg.data["symbols"]
    node = symbols.get(sym)
    if not isinstance(node, dict):
        node = {}
        symbols[sym] = node

    node["alpaca_qty"] = None if qty is None else float(qty)
    node["ts"] = _utc_iso()


def recompute_mismatch(reg: Registry, symbol: Optional[str] = None) -> None:
    syms = [symbol] if symbol else list((reg.data.get("symbols") or {}).keys())
    for s in syms:
        sym = str(s).upper().strip()
        node = reg.data.get("symbols", {}).get(sym)
        if not isinstance(node, dict):
            continue
        a = node.get("alpaca_qty")
        alp = None if a is None else _safe_float(a, 0.0)
        tot = reserved_total(reg, sym)
        mismatch = False
        if alp is None:
            mismatch = False  # unknown broker qty shouldn't auto-kill
        else:
            mismatch = tot - alp > 1e-9
        node["mismatch"] = bool(mismatch)
        node["reserved_total"] = float(tot)
        node["ts"] = _utc_iso()


def any_mismatch(reg: Registry) -> bool:
    syms = (reg.data.get("symbols") or {})
    if not isinstance(syms, dict):
        return False
    for _, node in syms.items():
        if isinstance(node, dict) and bool(node.get("mismatch")):
            return True
    return False


def reconcile_with_alpaca_positions(reg: Registry, alpaca_positions: Dict[str, float]) -> Dict[str, Any]:
    """Update alpaca_qty snapshot and recompute mismatches.

    alpaca_positions: {"AAPL": 15.0, ...}

    Returns a summary with mismatch_count.
    """
    seen = 0
    for sym, qty in alpaca_positions.items():
        set_alpaca_qty(reg, sym, float(qty))
        seen += 1

    # Also mark symbols that exist in registry but not in broker as zero.
    syms = (reg.data.get("symbols") or {})
    if isinstance(syms, dict):
        for sym in list(syms.keys()):
            if sym not in alpaca_positions:
                set_alpaca_qty(reg, sym, 0.0)

    recompute_mismatch(reg)
    mismatches = 0
    if isinstance(reg.data.get("symbols"), dict):
        for _, node in reg.data["symbols"].items():
            if isinstance(node, dict) and bool(node.get("mismatch")):
                mismatches += 1

    save_registry(reg)
    return {
        "status": "ok",
        "seen": int(seen),
        "mismatch_count": int(mismatches),
        "ts": _utc_iso(),
    }


def can_sell_qty(reg: Registry, symbol: str, owner: str) -> float:
    """Return how many shares a strategy is allowed to sell for a symbol.

    Conservative rule: a strategy may only sell what it has reserved.
    If alpaca_qty is known, clamp to broker qty too.
    """
    sym = (symbol or "").upper().strip()
    owner = (owner or "").upper().strip() or "DT"
    if not sym:
        return 0.0

    allowed = get_reserved_qty(reg, sym, owner)

    node = reg.data.get("symbols", {}).get(sym)
    if isinstance(node, dict):
        a = node.get("alpaca_qty")
        if a is not None:
            allowed = min(float(allowed), _safe_float(a, 0.0))

    return max(0.0, float(allowed))


def reserve_on_fill(reg: Registry, symbol: str, side: str, qty: float, owner: str) -> Registry:
    """Update reserved quantities after a known fill.

    BUY  -> increase owner's reserved qty
    SELL -> decrease owner's reserved qty
    """
    sym = (symbol or "").upper().strip()
    side = (side or "").upper().strip()
    owner = (owner or "").upper().strip() or "DT"
    q = _safe_float(qty, 0.0)
    if not sym or q <= 0:
        return reg

    if side == "BUY":
        add_reserved_qty(reg, sym, owner, +q)
    elif side == "SELL":
        add_reserved_qty(reg, sym, owner, -q)

    return reg

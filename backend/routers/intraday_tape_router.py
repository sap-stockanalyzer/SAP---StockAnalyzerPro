"""
Intraday Tape Router — AION Analytics

Purpose:
  Expose "what it bought/sold" (fills/orders) and "what it's thinking" (signals)
  as lightweight, file-backed endpoints for the UI.

Endpoints (mounted under /api/backend):
  GET /api/backend/intraday/fills/recent?limit=20
  GET /api/backend/intraday/fills?limit=20
  GET /api/backend/intraday/orders/recent?limit=20
  GET /api/backend/intraday/orders?limit=20
  GET /api/backend/intraday/signals/latest?limit=50
  GET /api/backend/intraday/signals?limit=50
  GET /api/backend/intraday/signals/top?limit=50

Alias (paper broker compatibility):
  GET /api/backend/broker/paper/fills?limit=20

How files are discovered:
  - If PATHS contains explicit keys, we use them:
      PATHS["intraday_fills_path"]
      PATHS["intraday_orders_path"]
      PATHS["intraday_signals_path"]
  - Otherwise we auto-discover under PATHS["ml_data_dt"] (default: "ml_data_dt")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query

from backend.core.config import PATHS, TIMEZONE


router = APIRouter(prefix="/api/backend", tags=["intraday-tape"])


# -----------------------------
# Helpers
# -----------------------------

def _iso_now() -> str:
    return datetime.now(TIMEZONE).isoformat()


def _read_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl(path: Path, limit: int = 1000) -> List[dict]:
    """Read newline-delimited JSON (best-effort)."""
    out: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(out) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _as_list(payload: Any, preferred_keys: List[str]) -> List[Any]:
    """Coerce payload into a list, trying known keys first."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in preferred_keys:
            v = payload.get(k)
            if isinstance(v, list):
                return v
        # Sometimes it’s dict keyed by ids -> row objects
        # Keep dict values if they look like records.
        vals = list(payload.values())
        if vals and all(isinstance(x, dict) for x in vals[: min(10, len(vals))]):
            return vals
    return []


def _mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _candidate_roots() -> List[Path]:
    base = Path(PATHS.get("ml_data_dt", "ml_data_dt"))
    # Common places your dt engine might write
    return [
        base,
        base / "sim_logs",
        base / "broker",
        base / "broker" / "paper",
        base / "paper",
        base / "paper_broker",
        base / "fills",
        base / "orders",
        base / "executions",
        base / "trades",
        base / "signals",
        base / "predictions",
        base / "artifacts",
        base / "logs",
        base / "cache",
    ]


def _find_latest_file(
    explicit_path: Optional[str],
    patterns: List[str],
) -> Optional[Path]:
    # Explicit beats discovery
    if explicit_path:
        try:
            p = Path(explicit_path)
            if p.exists() and p.is_file():
                return p
        except Exception:
            pass

    best: Optional[Path] = None
    best_m = 0.0

    for root in _candidate_roots():
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.rglob(pat):
                if not p.is_file():
                    continue
                mt = _mtime(p)
                if mt > best_m:
                    best_m = mt
                    best = p

    return best


def _slice_recent(items: List[Any], limit: int) -> List[Any]:
    if limit <= 0:
        return []
    # Most logs append over time; keep the last N if big.
    return items[-limit:] if len(items) > limit else items


def _sort_by_conf(items: List[dict]) -> List[dict]:
    def conf(x: dict) -> float:
        try:
            return float(x.get("confidence") or x.get("conf") or x.get("p") or 0.0)
        except Exception:
            return 0.0
    return sorted(items, key=conf, reverse=True)


def _load_list_from_file(path: Path, keys: List[str], limit: int) -> Tuple[List[Any], str]:
    """
    Returns (items, updated_at_iso).
    Supports .json and .jsonl.
    """
    updated_at = ""
    try:
        updated_at = datetime.fromtimestamp(_mtime(path), tz=TIMEZONE).isoformat()
    except Exception:
        updated_at = ""

    if path.suffix.lower() == ".jsonl":
        items = _read_jsonl(path, limit=max(limit * 5, 200))
        return _slice_recent(items, limit), updated_at

    payload = _read_json(path)
    items = _as_list(payload, keys)
    return _slice_recent(items, limit), updated_at


# -----------------------------
# Endpoints
# -----------------------------

@router.get("/intraday/fills/recent")
async def intraday_fills_recent(limit: int = Query(20, ge=1, le=500)) -> Dict[str, Any]:
    """
    Return recent paper/live fills (buy/sell executions).

    Output:
      { "updated_at": "...", "source": "...", "fills": [...] }
    """
    p = _find_latest_file(
        PATHS.get("intraday_fills_path"),
        patterns=[
            "*fills*.json",
            "*fill*.json",
            "*executions*.json",
            "*trades*.json",
            "*paper_fills*.json",
            "*broker_fills*.json",
            "*.jsonl",  # if you log fills as jsonl
        ],
    )
    if not p:
        return {"updated_at": None, "source": None, "fills": []}

    fills, updated_at = _load_list_from_file(p, keys=["fills", "items", "executions", "trades"], limit=limit)
    return {"updated_at": updated_at, "source": str(p), "fills": fills}


@router.get("/intraday/fills")
async def intraday_fills(limit: int = Query(20, ge=1, le=500)) -> Dict[str, Any]:
    return await intraday_fills_recent(limit=limit)


@router.get("/intraday/orders/recent")
async def intraday_orders_recent(limit: int = Query(20, ge=1, le=500)) -> Dict[str, Any]:
    """
    Return recent orders (submitted/canceled/filled).

    Output:
      { "updated_at": "...", "source": "...", "orders": [...] }
    """
    p = _find_latest_file(
        PATHS.get("intraday_orders_path"),
        patterns=[
            "*orders*.json",
            "*order*.json",
            "*paper_orders*.json",
            "*broker_orders*.json",
            "*.jsonl",
        ],
    )
    if not p:
        return {"updated_at": None, "source": None, "orders": []}

    orders, updated_at = _load_list_from_file(p, keys=["orders", "items"], limit=limit)
    return {"updated_at": updated_at, "source": str(p), "orders": orders}


@router.get("/intraday/orders")
async def intraday_orders(limit: int = Query(20, ge=1, le=500)) -> Dict[str, Any]:
    return await intraday_orders_recent(limit=limit)


@router.get("/intraday/signals/latest")
async def intraday_signals_latest(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    """
    Return latest intraday signals.

    Output:
      { "updated_at": "...", "source": "...", "signals": [...] }
    """
    p = _find_latest_file(
        PATHS.get("intraday_signals_path"),
        patterns=[
            "*signals*.json",
            "*dt_signals*.json",
            "*intraday_signals*.json",
            "*predictions*.json",        # fallback if you only write predictions
            "*intraday_predictions*.json",
            "*.jsonl",
        ],
    )
    if not p:
        return {"updated_at": None, "source": None, "signals": []}

    sigs, updated_at = _load_list_from_file(p, keys=["signals", "top", "items", "results", "predictions"], limit=limit)
    # keep shape stable for UI
    return {"updated_at": updated_at, "source": str(p), "signals": sigs}


@router.get("/intraday/signals")
async def intraday_signals(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    return await intraday_signals_latest(limit=limit)


@router.get("/intraday/signals/top")
async def intraday_signals_top(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    """
    Return top signals (sorted by confidence if present).
    """
    res = await intraday_signals_latest(limit=max(limit * 5, 100))
    sigs = res.get("signals") or []
    sigs2: List[dict] = [x for x in sigs if isinstance(x, dict)]
    sigs2 = _sort_by_conf(sigs2)
    return {"updated_at": res.get("updated_at"), "source": res.get("source"), "signals": sigs2[:limit]}


# Alias to satisfy older UI probes:
@router.get("/broker/paper/fills")
async def broker_paper_fills(limit: int = Query(20, ge=1, le=500)) -> Dict[str, Any]:
    """
    Alias for /intraday/fills/recent
    """
    res = await intraday_fills_recent(limit=limit)
    # Keep the key name "fills" to match your UI expectation.
    return res

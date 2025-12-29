from __future__ import annotations

from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Query

from dt_backend.core.data_pipeline_dt import _read_rolling, _rolling_path

router = APIRouter()


@router.get("/rolling/path")
def rolling_path() -> Dict[str, str]:
    return {"path": str(_rolling_path())}


@router.get("/rolling/summary")
def rolling_summary(
    include_meta: bool = Query(default=False, description="If false, omit keys starting with '_'"),
) -> Dict[str, Any]:
    """Lightweight summary to avoid shipping the full rolling blob."""
    rolling = _read_rolling() or {}
    if not include_meta:
        rolling = {k: v for k, v in rolling.items() if not str(k).startswith("_")}

    symbols = [str(k) for k in rolling.keys() if not str(k).startswith("_")]
    return {
        "symbols_total": len(symbols),
        "symbols": len(symbols),
        "symbols_sample": sorted(symbols)[:50],
        "has_meta_keys": any(str(k).startswith("_") for k in ( _read_rolling() or {}).keys()),
    }


@router.get("/rolling")
def get_rolling(
    symbol: Optional[str] = Query(default=None, description="If provided, return only this symbol node."),
    include_meta: bool = Query(default=False, description="If false, omit keys starting with '_'"),
    offset: int = Query(default=0, ge=0, description="Page offset into symbol list (only when symbol is not provided)."),
    limit: int = Query(default=200, ge=1, le=500, description="Max symbols to return (only when symbol is not provided)."),
    symbols_only: bool = Query(default=False, description="If true, return only the paged list of symbols (no data)."),
) -> Dict[str, Any]:
    """Paged rolling accessor.

    Important: we intentionally avoid returning the ENTIRE rolling object by default.
    Use `symbol=...` for a single node, or paginate with offset/limit.
    """
    rolling = _read_rolling() or {}

    # Single-symbol mode (safe + backwards compatible for callers that always pass symbol)
    if symbol:
        key = symbol.upper().strip()
        node = rolling.get(key) or rolling.get(symbol) or {}
        return {"symbol": key, "data": node}

    # Collection mode: strip meta keys unless explicitly requested
    if not include_meta:
        rolling = {k: v for k, v in rolling.items() if not str(k).startswith("_")}

    symbols: List[str] = sorted([str(k) for k in rolling.keys()])
    total = len(symbols)
    page = symbols[offset : offset + limit]

    if symbols_only:
        return {
            "symbols_total": total,
            "symbols": total,
            "offset": offset,
            "limit": limit,
            "returned": len(page),
            "symbols": page,
            "note": "Pass symbol=... to fetch a single node.",
        }

    data = {sym: rolling.get(sym) for sym in page}
    return {
        "symbols_total": total,
            "symbols": total,
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "data": data,
        "note": "Pass symbol=... for a single node; avoid requesting huge pages.",
    }

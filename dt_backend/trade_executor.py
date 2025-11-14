"""Translate ranked intraday predictions into actionable paper orders."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, List

from dt_backend.config_dt import DT_PATHS
from dt_backend.broker_api import PaperBroker, queue_orders, Order

try:
    from backend.live_prices_router import fetch_live_prices  # type: ignore
except Exception:  # pragma: no cover
    fetch_live_prices = None  # type: ignore

PENDING_PATH = DT_PATHS["dtbroker"] / "pending_orders.json"
SIGNALS_PATH = DT_PATHS["dtsignals"] / "prediction_rank_fetch.json.gz"


def _normalize(predictions: Dict[str, Any] | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(predictions, dict) and "ranks" in predictions:
        return predictions.get("ranks", [])
    if isinstance(predictions, dict):
        return [
            {
                "symbol": sym,
                "score": node.get("score", node.get("predicted", 0.0)),
                "confidence": node.get("confidence", 0.0),
                "label": node.get("label"),
                "currentPrice": node.get("currentPrice"),
            }
            for sym, node in predictions.items()
        ]
    return list(predictions or [])


def _load_rank_file() -> List[Dict[str, Any]]:
    import gzip

    if not SIGNALS_PATH.exists():
        return []
    with gzip.open(SIGNALS_PATH, "rt", encoding="utf-8") as f:
        js = json.load(f)
    return js.get("ranks", [])


def sync_predictions_to_broker(predictions: Dict[str, Any] | List[Dict[str, Any]], *, max_positions: int = 10, risk_pct: float = 0.01) -> Dict[str, Any]:
    """Persist pending trades derived from the freshest prediction batch."""
    rows = _normalize(predictions)
    if not rows:
        rows = _load_rank_file()
    if not rows:
        return {"status": "skipped", "reason": "no_predictions"}

    rows = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    top = rows[:max_positions]
    symbols = [r.get("symbol") for r in top if r.get("symbol")]

    live_prices = {}
    if fetch_live_prices and symbols:
        try:
            live_prices = fetch_live_prices(symbols=symbols)
        except Exception:
            live_prices = {}

    broker = PaperBroker()
    placed: List[Order] = []
    for row in top:
        sym = row.get("symbol")
        if not sym:
            continue
        side = "buy" if float(row.get("score", 0.0)) >= 0 else "sell"
        price = None
        if isinstance(live_prices, dict):
            price = (live_prices.get(sym) or {}).get("price")
        if price is None:
            price = row.get("currentPrice") or row.get("price")
        if not price:
            continue
        cash_alloc = max(1.0, broker.cash * risk_pct)
        qty = max(1, int(cash_alloc // float(price)))
        try:
            order = broker.place_order(sym, side, qty, float(price), metadata=row)
            placed.append(order)
        except Exception:
            continue

    if not placed:
        return {"status": "skipped", "reason": "no_orders"}

    queue_orders(placed)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "orders": [asdict(order) for order in placed],
    }
    PENDING_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"status": "ok", "orders": len(placed), "symbols": [o.symbol for o in placed]}

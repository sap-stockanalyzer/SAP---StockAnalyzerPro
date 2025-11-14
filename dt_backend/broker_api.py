"""Paper broker implementation for the intraday (DT) stack.
Persists lightweight state under ml_data_dt/sim_broker/ so day-trading
bots and schedulers can coordinate via filesystem-only contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
from datetime import datetime
import json

from dt_backend.config_dt import DT_PATHS

try:  # optional dependency for safe casting
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _json_safe(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if np is not None and isinstance(value, np.generic):  # type: ignore[attr-defined]
        return value.item()
    return str(value)

BROKER_DIR = DT_PATHS["dtbroker"]
BROKER_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = BROKER_DIR / "state.json"
ORDERS_LOG = DT_PATHS["dtlogs"] / "trade_orders.jsonl"
ORDERS_LOG.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Order:
    symbol: str
    side: str
    qty: int
    price: float
    timestamp: str
    metadata: Dict[str, Any]


class PaperBroker:
    """Minimal, file-backed broker suitable for CI and offline testing."""

    def __init__(self, starting_equity: float = 100_000.0) -> None:
        self.starting_equity = starting_equity
        self._state = {
            "cash": starting_equity,
            "positions": {},
            "orders": [],
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if STATE_PATH.exists():
            try:
                self._state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _save(self) -> None:
        self._state["updated_at"] = datetime.utcnow().isoformat() + "Z"
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    @property
    def cash(self) -> float:
        return float(self._state.get("cash", self.starting_equity))

    @property
    def positions(self) -> Dict[str, Any]:
        return dict(self._state.get("positions", {}))

    def reset(self) -> None:
        self._state = {
            "cash": self.starting_equity,
            "positions": {},
            "orders": [],
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        self._save()

    def place_order(self, symbol: str, side: str, qty: int, price: float, *, metadata: Dict[str, Any] | None = None) -> Order:
        qty = int(max(1, qty))
        price = float(price)
        side = side.lower()
        meta = {k: _json_safe(v) for k, v in (metadata or {}).items()}

        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {side}")

        cost = qty * price
        if side == "buy" and cost > self.cash:
            qty = max(1, int(self.cash // max(price, 1e-6)))
            cost = qty * price
        if qty <= 0:
            raise ValueError("Not enough cash to place order")

        pos = self._state.setdefault("positions", {})
        node = pos.get(symbol, {"qty": 0, "avg_price": 0.0})
        if side == "buy":
            new_qty = node["qty"] + qty
            node["avg_price"] = ((node["avg_price"] * node["qty"]) + cost) / max(new_qty, 1)
            node["qty"] = new_qty
            self._state["cash"] = max(0.0, self.cash - cost)
        else:  # sell
            sell_qty = min(qty, node.get("qty", 0))
            if sell_qty <= 0:
                raise ValueError("No inventory to sell")
            node["qty"] = max(0, node.get("qty", 0) - sell_qty)
            self._state["cash"] = self.cash + sell_qty * price
            qty = sell_qty
        pos[symbol] = node

        order = Order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            timestamp=datetime.utcnow().isoformat() + "Z",
            metadata=meta,
        )
        orders = self._state.setdefault("orders", [])
        orders.append(asdict(order))
        self._append_order_log(order)
        self._save()
        return order

    def pending_orders(self) -> List[Dict[str, Any]]:
        return list(self._state.get("orders", []))

    def _append_order_log(self, order: Order) -> None:
        try:
            with open(ORDERS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(order)) + "\n")
        except Exception:
            pass


def queue_orders(orders: List[Order]) -> None:
    """Utility to persist a pending order batch for schedulers/bots."""
    payload = [asdict(o) for o in orders]
    out_path = BROKER_DIR / "pending_orders.json"
    out_path.write_text(json.dumps({"generated_at": datetime.utcnow().isoformat() + "Z", "orders": payload}, indent=2), encoding="utf-8")

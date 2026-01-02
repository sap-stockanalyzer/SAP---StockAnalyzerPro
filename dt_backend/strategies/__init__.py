"""dt_backend/strategies

Phase 3 — Strategy bots (entry logic only).

These modules generate *trade plans* based on intraday features + levels + regime.

They intentionally do NOT place orders. They only propose:
  - side (BUY/SELL/FLAT)
  - confidence + score
  - stop / take-profit suggestions (ATR/VWAP based)
  - time stop guidance

Execution semantics (brackets, trailing, partials) come later (Phase 5).
"""

from __future__ import annotations

from .strategy_engine_dt import select_best_setup, build_setups_for_symbol

__all__ = ["select_best_setup", "build_setups_for_symbol"]

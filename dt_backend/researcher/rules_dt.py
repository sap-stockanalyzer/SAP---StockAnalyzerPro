# dt_backend/researcher/rules_dt.py — compatibility shim (Phase 9)
"""Compatibility layer.

Older modules may import dt_backend.researcher.rules_dt expecting helper
functions for adjusting setups based on researcher-promoted rules.

The Phase 9 implementation lives in:
  - rules_runtime_dt.py (fast gates used in live loops)

This file simply forwards to keep imports stable.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from dt_backend.researcher.rules_runtime_dt import bot_allowed
except Exception:  # pragma: no cover
    bot_allowed = None  # type: ignore


def rules_adjust_setup(setup: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Best-effort setup adjustment.

    Phase 9 focuses on *filters/rules*, not inventing new strategies. The
    current runtime module gates eligibility at the bot level (allow/deny)
    and does not mutate setup content.

    We return the input unchanged for now to avoid unintended behavior.
    """
    return setup

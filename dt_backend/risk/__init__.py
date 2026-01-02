"""Risk modules for dt_backend.

This package contains *non-model* safety layers:
  - Phase 0: hard risk rails (kill-switch, cooldowns, max exposure)
  - Phase 5.5: portfolio heat manager (sector/correlation caps)
  - Phase 8: news/event risk adjustments

Everything here should be safe-by-default and best-effort (never raise).
"""

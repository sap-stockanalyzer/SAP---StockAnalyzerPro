"""dt_backend/jobs/dt_scheduler.py

Simple on-server scheduler for the intraday system (Linux-friendly).

Purpose
-------
Run the DT system as a single long-running process without relying on cron:

  • While the market is open:
      - refresh live bars (1m/5m) on a short interval
      - run a trading cycle (context→features→predict→policy→exec_intent→execute)
  • When the market transitions from open → closed:
      - run end-of-day cleanup (persist to dt_brain, then clear intraday rolling)

Phase 0 safety
--------------
We acquire a scheduler lock file on startup so two schedulers can't run at once.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dt_backend.core.logger_dt import log, warn

from dt_backend.jobs.live_market_data_loop import fetch_live_bars_once
from dt_backend.jobs.daytrading_job import run_daytrading_cycle
from dt_backend.jobs.end_of_day_cleanup import run_end_of_day_cleanup
from dt_backend.jobs.dt_nightly_job import run_dt_nightly_job, last_dt_nightly_session_date
from dt_backend.core.dt_brain import read_dt_brain
from dt_backend.core.locks_dt import acquire_scheduler_lock, release_lock_file

try:
    from utils.time_utils import is_market_open, now_ny  # type: ignore
except Exception:  # pragma: no cover
    is_market_open = None  # type: ignore
    now_ny = None  # type: ignore

# ---------------------------------------------------------------------------
# Local market-hours fallback (keeps dt_scheduler working even if utils.time_utils
# isn't present).
# ---------------------------------------------------------------------------
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ny() -> datetime:
    """NY-local datetime with tzinfo when possible."""
    if callable(now_ny):
        return now_ny()  # type: ignore[misc]
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("America/New_York"))  # type: ignore[misc]
    # Worst case fallback: UTC "pretending" to be local; still stable for comparisons.
    return datetime.now(timezone.utc)


def _is_market_open_fallback(ny_now: Optional[datetime] = None) -> bool:
    """Market-open check.
    - Prefer utils.time_utils.is_market_open() if available.
    - Otherwise use a simple NY weekday + 09:30–16:00 time window (no holiday calendar).
    """
    if callable(is_market_open):
        try:
            return bool(is_market_open())  # type: ignore[call-arg]
        except Exception:
            pass

    n = ny_now or _now_ny()
    wd = int(n.weekday())  # 0=Mon
    if wd >= 5:
        return False
    hm = n.hour * 60 + n.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    return (hm >= open_min) and (hm < close_min)


def _close_plus_one_ts(ny: datetime) -> datetime:
    """16:00:01 NY for the session date of `ny`."""
    return ny.replace(hour=16, minute=0, second=1, microsecond=0)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _last_cleanup_session_date() -> Optional[str]:
    """Reads dt_brain meta for last EOD cleanup session date."""
    try:
        brain = read_dt_brain()
        meta = brain.get("_meta")
        if not isinstance(meta, dict):
            return None
        v = meta.get("last_eod_cleanup_session_date")
        return str(v) if v else None
    except Exception:
        return None


def run_dt_scheduler(
    *,
    bars_interval_sec: int = 60,
    trade_interval_sec: int = 60,
    max_symbols: Optional[int] = None,
    fetch_1m: bool = True,
    fetch_5m: bool = True,
    execute: bool = True,
) -> Dict[str, Any]:
    """Run an infinite loop scheduler."""

    sched_lock = acquire_scheduler_lock()
    if not sched_lock:
        return {"status": "locked"}

    try:
        log(
            "[dt_scheduler] start "
            f"bars_interval={bars_interval_sec}s trade_interval={trade_interval_sec}s "
            f"max_symbols={max_symbols} execute={execute}"
        )

        # Track transition so we can run EOD exactly once per NY session.
        was_open = False
        last_bars_t = 0.0
        last_trade_t = 0.0

        # Optional small cooldown after hard failures to avoid hot crash loops
        backoff_sec = 0.0

        while True:
            loop_t = time.time()

            # ✅ ALWAYS define time anchors at top of loop (prevents UnboundLocalError-style crashes)
            now_utc = datetime.now(timezone.utc)
            ny_now = _now_ny()

            # Use NY date as the canonical session key for EOD/nightly idempotence
            sess_ny = ny_now.date().isoformat()

            # Determine market state.
            open_now = _is_market_open_fallback(ny_now)

            # ----
            # EOD transition: open -> closed
            # ----
            if was_open and not open_now:
                try:
                    last_done = _last_cleanup_session_date()
                    if last_done == sess_ny:
                        log(f"[dt_scheduler] market closed; EOD already done for {sess_ny}")
                    else:
                        log(f"[dt_scheduler] market closed; running EOD cleanup for {sess_ny} …")
                        res = run_end_of_day_cleanup(clear_global_blocks=True)
                        log(f"[dt_scheduler] EOD cleanup result: {res}")

                    # Nightly should run after close+1s (idempotent per session date).
                    # We do NOT sleep-until-close inside scheduler; we just check readiness.
                    try:
                        close_plus_1 = _close_plus_one_ts(ny_now)
                        if ny_now >= close_plus_1:
                            last_n = last_dt_nightly_session_date() or ""
                            if last_n == sess_ny:
                                log(f"[dt_scheduler] DT nightly already done for {sess_ny}")
                            else:
                                log(f"[dt_scheduler] running DT nightly for {sess_ny} …")
                                nres = run_dt_nightly_job(session_date=sess_ny)
                                log(f"[dt_scheduler] DT nightly result: {nres}")
                        else:
                            # Not yet close+1s; we'll catch it in the late-start check below.
                            pass
                    except Exception as e:
                        warn(f"[dt_scheduler] DT nightly failed: {e}")

                except Exception as e:
                    warn(f"[dt_scheduler] EOD cleanup failed: {e}")

            # ----
            # If scheduler started AFTER close, still run DT nightly once (close+1s) for the session.
            # ----
            if not open_now:
                try:
                    close_plus_1 = _close_plus_one_ts(ny_now)
                    if ny_now >= close_plus_1:
                        last_n = last_dt_nightly_session_date() or ""
                        if last_n != sess_ny:
                            log(f"[dt_scheduler] market closed; scheduler late-start → running DT nightly for {sess_ny} …")
                            nres = run_dt_nightly_job(session_date=sess_ny)
                            log(f"[dt_scheduler] DT nightly result: {nres}")
                except Exception as e:
                    warn(f"[dt_scheduler] DT nightly late-start check failed: {e}")

            was_open = open_now

            # ----
            # If market open: update bars + run trade cycles
            # ----
            if open_now:
                if loop_t - last_bars_t >= max(1, int(bars_interval_sec)):
                    try:
                        fetch_live_bars_once(
                            max_symbols=max_symbols,
                            fetch_1m=fetch_1m,
                            fetch_5m=fetch_5m,
                        )
                    except Exception as e:
                        warn(f"[dt_scheduler] live bars cycle failed: {e}")
                        backoff_sec = min(10.0, (backoff_sec or 1.0) * 1.5)
                    else:
                        backoff_sec = 0.0
                    last_bars_t = loop_t

                if loop_t - last_trade_t >= max(1, int(trade_interval_sec)):
                    try:
                        out = run_daytrading_cycle(max_symbols=max_symbols, execute=execute)
                        log(f"[dt_scheduler] trade cycle done: {now_utc.isoformat()} {out}")
                    except Exception as e:
                        warn(f"[dt_scheduler] trade cycle failed: {e}")
                        backoff_sec = min(10.0, (backoff_sec or 1.0) * 1.5)
                    else:
                        backoff_sec = 0.0
                    last_trade_t = loop_t

            # Sleep: base tick 1s + optional backoff if we’re erroring hard
            time.sleep(1.0 + float(backoff_sec or 0.0))

    finally:
        release_lock_file(sched_lock)


def main() -> None:
    run_dt_scheduler()


if __name__ == "__main__":
    main()

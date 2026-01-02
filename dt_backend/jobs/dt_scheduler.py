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
    if callable(now_ny):
        return now_ny()  # type: ignore[misc]
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("America/New_York"))  # type: ignore[misc]
    return datetime.now(timezone.utc)


def _is_market_open_fallback() -> bool:
    # Prefer utils.time_utils if it exists.
    if callable(is_market_open):
        try:
            return bool(is_market_open())  # type: ignore[call-arg]
        except Exception:
            pass

    # Next-best: simple NY schedule (no holiday calendar).
    n = _now_ny()
    wd = int(n.weekday())  # 0=Mon
    if wd >= 5:
        return False
    hm = n.hour * 60 + n.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    return (hm >= open_min) and (hm < close_min)


def _close_plus_one_ts(ny: datetime) -> datetime:
    # 16:00:01 NY
    return ny.replace(hour=16, minute=0, second=1, microsecond=0)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _last_cleanup_session_date() -> Optional[str]:
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

        # Track transition so we can run EOD exactly once per session.
        was_open = False
        last_bars_t = 0.0
        last_trade_t = 0.0

        while True:
            t = time.time()

            # Determine market state.
            open_now = _is_market_open_fallback()

            # ----
            # EOD transition: open -> closed
            # ----
            if was_open and not open_now:
                try:
                    if callable(now_ny):
                        today = now_ny().date().isoformat()  # type: ignore[call-arg]
                    else:
                        today = datetime.now(timezone.utc).date().isoformat()

                    last_done = _last_cleanup_session_date()
                    if last_done == today:
                        log(f"[dt_scheduler] market closed; EOD already done for {today}")
                    else:
                        log(f"[dt_scheduler] market closed; running EOD cleanup for {today} …")
                        res = run_end_of_day_cleanup(clear_global_blocks=True)
                        log(f"[dt_scheduler] EOD cleanup result: {res}")

                        # Run DT nightly immediately after close (+1s), idempotent per session.
                        try:
                            ny = _now_ny()
                            ts = _close_plus_one_ts(ny)
                            if ny < ts:
                                time.sleep(max(0.0, (ts - ny).total_seconds()))
                            sess = ts.date().isoformat()
                            last_n = last_dt_nightly_session_date() or ""
                            if last_n == sess:
                                log(f"[dt_scheduler] DT nightly already done for {sess}")
                            else:
                                log(f"[dt_scheduler] running DT nightly for {sess} …")
                                nres = run_dt_nightly_job(session_date=sess)
                                log(f"[dt_scheduler] DT nightly result: {nres}")
                        except Exception as e:
                            warn(f"[dt_scheduler] DT nightly failed: {e}")
                except Exception as e:
                    warn(f"[dt_scheduler] EOD cleanup failed: {e}")

            # ----
            # If scheduler started AFTER close, still run DT nightly once (close+1s) for the session.
            # ----
            if not open_now:
                try:
                    ny = _now_ny()
                    ts = _close_plus_one_ts(ny)
                    if ny >= ts:
                        sess = ts.date().isoformat()
                        last_n = last_dt_nightly_session_date() or ""
                        if last_n != sess:
                            log(f"[dt_scheduler] market closed; scheduler late-start → running DT nightly for {sess} …")
                            nres = run_dt_nightly_job(session_date=sess)
                            log(f"[dt_scheduler] DT nightly result: {nres}")
                except Exception as e:
                    warn(f"[dt_scheduler] DT nightly late-start check failed: {e}")

            was_open = open_now

            # ----
            # If market open: update bars + run trade cycles
            # ----
            if open_now:
                if t - last_bars_t >= max(1, int(bars_interval_sec)):
                    try:
                        fetch_live_bars_once(
                            max_symbols=max_symbols,
                            fetch_1m=fetch_1m,
                            fetch_5m=fetch_5m,
                        )
                    except Exception as e:
                        warn(f"[dt_scheduler] live bars cycle failed: {e}")
                    last_bars_t = t

                if t - last_trade_t >= max(1, int(trade_interval_sec)):
                    try:
                        out = run_daytrading_cycle(max_symbols=max_symbols, execute=execute)
                        log(f"[dt_scheduler] trade cycle done: {_utc_now_iso()} {out}")
                    except Exception as e:
                        warn(f"[dt_scheduler] trade cycle failed: {e}")
                    last_trade_t = t

            time.sleep(1.0)
    finally:
        release_lock_file(sched_lock)


def main() -> None:
    run_dt_scheduler()


if __name__ == "__main__":
    main()

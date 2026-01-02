# dt_backend/historical_replay/fetch_raw_days_dt.py
"""CLI wrapper to download raw intraday days for replay.

Example:
  python -m dt_backend.historical_replay.fetch_raw_days_dt --start 2025-12-01 --end 2025-12-31

Writes:
  <ml_data_dt>/intraday/replay/raw_days/YYYY-MM-DD.json.gz
"""

from __future__ import annotations

import argparse
from dt_backend.core.logger_dt import log
from dt_backend.historical_replay.historical_replay_fetcher import fetch_range


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    days = fetch_range(args.start, args.end, universe=None)
    log(f"[fetch_raw_days_dt] done: {len(days)} day(s)")


if __name__ == "__main__":
    main()

"""
insights_scheduler.py — v1.6 (Unified Config + Persistent Premarket Loop)
Author: AION Analytics / StockAnalyzerPro

Purpose:
- Runs the daily premarket Top Picks build (weekdays only).
- Writes persistent logs in PATHS["ml_data"]/logs/insights_YYYYMMDD.log
  for dashboard visibility and status monitoring.
"""

from __future__ import annotations
import os, sys, threading, asyncio
from datetime import datetime, timedelta
from .insights_builder import build_daily_insights
from .config import PATHS  # ✅ unified config import

# ---------------------------------------------------------------------
# Log Setup
# ---------------------------------------------------------------------
LOG_DIR = PATHS["ml_data"] / "logs"  # ✅ use unified config path
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_filename = f"insights_{datetime.utcnow().strftime('%Y%m%d')}.log"
log_path = LOG_DIR / log_filename

# Duplicate output (stdout + log file)
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, "a", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = sys.stdout

# ---------------------------------------------------------------------
# Scheduler Config
# ---------------------------------------------------------------------
HOUR_UTC = int(os.getenv("INSIGHTS_HOUR_UTC", "13"))   # ~09:00 ET default
MIN_UTC  = int(os.getenv("INSIGHTS_MINUTE_UTC", "0"))

# ---------------------------------------------------------------------
# Core async loop
# ---------------------------------------------------------------------
async def _loop():
    while True:
        now = datetime.utcnow()
        # Weekdays only: Monday=0 ... Sunday=6
        if now.weekday() < 5:
            target = now.replace(hour=HOUR_UTC, minute=MIN_UTC, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            # Skip weekends entirely
            while target.weekday() >= 5:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()

            print(f"[{now:%Y-%m-%d %H:%M:%S}] 🕘 Next Insights run scheduled for {target:%Y-%m-%d %H:%M:%S} UTC")
            await asyncio.sleep(wait_s)

            try:
                print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}] 🏁 Running premarket Insights build...")
                res = build_daily_insights()
                print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}] ✅ Insights build complete: {res}")
            except Exception as e:
                print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}] ❌ Insights build error: {e}")
        else:
            # Weekend — sleep until next weekday 09:00 UTC
            next_day = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            print(f"[{now:%Y-%m-%d %H:%M:%S}] 💤 Weekend mode — sleeping until {next_day:%Y-%m-%d %H:%M:%S} UTC")
            await asyncio.sleep((next_day - now).total_seconds())

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def start_premarket_insights_loop():
    def run():
        asyncio.run(_loop())
    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M:%S}] 🕘 Premarket Insights loop started — logs → {log_path}")

if __name__ == "__main__":
    start_premarket_insights_loop()
    asyncio.get_event_loop().run_forever()

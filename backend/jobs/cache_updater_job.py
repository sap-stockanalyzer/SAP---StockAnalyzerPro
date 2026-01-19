# backend/jobs/cache_updater_job.py
"""
Frontend Cache Updater Job — AION Analytics

Background job that updates the unified frontend cache every 5 seconds.
This ensures the frontend always has fresh data without making multiple API calls.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime

from backend.services.unified_cache_service import UnifiedCacheService

try:
    from backend.core.config import TIMEZONE
except ImportError:
    from backend.config import TIMEZONE  # type: ignore


def update_frontend_cache() -> None:
    """
    Update the unified frontend cache.
    
    This is called by the scheduler every 5 seconds.
    Errors are logged but don't stop the job from running.
    """
    try:
        service = UnifiedCacheService()
        result = service.update_all()
        
        # Log success (can be commented out to reduce noise)
        timestamp = result.get("timestamp", "")
        errors = result.get("errors", {})
        
        if errors:
            print(f"[CacheUpdater] Updated cache at {timestamp} with errors: {list(errors.keys())}")
        else:
            # Uncomment for verbose logging:
            # print(f"[CacheUpdater] Updated cache at {timestamp}")
            pass
            
    except Exception as e:
        # Log error but don't fail
        print(f"[CacheUpdater] Failed to update cache: {type(e).__name__}: {e}")
        print(traceback.format_exc()[-1000:])


def run_cache_updater_loop() -> None:
    """
    Run the cache updater in a loop (for standalone execution).
    
    Updates every 5 seconds indefinitely.
    Useful for running as a separate process or in testing.
    """
    print("[CacheUpdater] Starting unified cache updater loop...")
    print("[CacheUpdater] Updates every 5 seconds. Press Ctrl+C to stop.")
    
    while True:
        try:
            update_frontend_cache()
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n[CacheUpdater] Shutting down...")
            break
        except Exception as e:
            print(f"[CacheUpdater] Loop error: {e}")
            time.sleep(5)  # Wait before retry


if __name__ == "__main__":
    # Allow running this job standalone for testing
    run_cache_updater_loop()

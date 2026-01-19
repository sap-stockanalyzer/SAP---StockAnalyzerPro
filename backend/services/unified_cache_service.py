# backend/services/unified_cache_service.py
"""
Unified Frontend Cache Service — AION Analytics

Provides a single source of truth for all frontend data needs.
Updates every 5 seconds via background job.
Reduces API call overhead and prevents race conditions.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from backend.core.config import PATHS, TIMEZONE
except ImportError:
    from backend.config import PATHS, TIMEZONE  # type: ignore


class UnifiedCacheService:
    """
    Manages a single unified cache file for all frontend needs.
    
    This service aggregates data from multiple sources:
    - Bots page bundle (swing + intraday)
    - Portfolio holdings
    - System status
    
    The cache is updated by a background job every 5 seconds.
    Frontend can fetch this single file instead of making multiple API calls.
    """
    
    def __init__(self):
        """Initialize the service with cache file path."""
        da_brains = Path(PATHS.get("da_brains", "ml_data"))
        da_brains.mkdir(parents=True, exist_ok=True)
        self.cache_file = da_brains / "frontend_unified_cache.json"
    
    def update_all(self) -> Dict[str, Any]:
        """
        Fetch all frontend data and update cache file.
        
        Returns:
            The aggregated data that was written to cache.
        """
        cache_data: Dict[str, Any] = {
            "timestamp": datetime.now(TIMEZONE).isoformat(),
            "version": "1.0",
            "data": {},
            "errors": {},
        }
        
        # 1. Bots page bundle (swing + intraday)
        try:
            from backend.routers.bots_page_router import bots_page_bundle
            import asyncio
            import inspect
            
            # Handle async function
            result = bots_page_bundle()
            if inspect.isawaitable(result):
                # Use asyncio.run() for clean async handling
                try:
                    cache_data["data"]["bots"] = asyncio.run(result)
                except RuntimeError as e:
                    # If event loop already running, need to use different approach
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context with running loop
                        cache_data["data"]["bots"] = loop.run_until_complete(result)
                    except RuntimeError:
                        # No running loop, create new one
                        cache_data["errors"]["bots"] = {
                            "error": "Cannot run async function in sync context with active loop",
                            "suggestion": "Call update_all from async context or use sync data fetching"
                        }
            else:
                cache_data["data"]["bots"] = result
        except Exception as e:
            cache_data["errors"]["bots"] = {
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1000:],
            }
        
        # 2. Portfolio top holdings
        try:
            from backend.routers.portfolio_router import get_top_holdings_by_pnl
            
            cache_data["data"]["portfolio"] = {
                "top_1w": get_top_holdings_by_pnl("1w", limit=3),
                "top_1m": get_top_holdings_by_pnl("1m", limit=3),
            }
        except Exception as e:
            cache_data["errors"]["portfolio"] = {
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1000:],
            }
        
        # 3. System status (lightweight)
        try:
            # Just mark that system is responding
            cache_data["data"]["system"] = {
                "status": "ok",
                "timestamp": datetime.now(TIMEZONE).isoformat(),
            }
        except Exception as e:
            cache_data["errors"]["system"] = {
                "error": f"{type(e).__name__}: {e}",
            }
        
        # Write to cache file
        try:
            self.cache_file.write_text(
                json.dumps(cache_data, indent=2, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            # Log but don't fail
            print(f"[UnifiedCache] Failed to write cache: {e}")
        
        return cache_data
    
    def get_cache(self) -> Dict[str, Any]:
        """
        Read unified cache from file.
        
        Returns:
            Cached data dict, or error dict if cache doesn't exist or is invalid.
        """
        if not self.cache_file.exists():
            return {
                "error": "Cache file does not exist. It should be created by the background job.",
                "timestamp": datetime.now(TIMEZONE).isoformat(),
                "data": {},
            }
        
        try:
            content = self.cache_file.read_text(encoding="utf-8")
            data = json.loads(content)
            
            # Add freshness indicator
            try:
                cache_time = datetime.fromisoformat(data.get("timestamp", ""))
                now = datetime.now(TIMEZONE)
                age_seconds = (now - cache_time).total_seconds()
                data["cache_age_seconds"] = age_seconds
            except Exception:
                pass
            
            return data
        except Exception as e:
            return {
                "error": f"Failed to read cache: {type(e).__name__}: {e}",
                "timestamp": datetime.now(TIMEZONE).isoformat(),
                "data": {},
            }
    
    def get_cache_age(self) -> Optional[float]:
        """
        Get age of cache file in seconds.
        
        Returns:
            Age in seconds, or None if cache doesn't exist.
        """
        if not self.cache_file.exists():
            return None
        
        try:
            cache = self.get_cache()
            cache_time = datetime.fromisoformat(cache.get("timestamp", ""))
            now = datetime.now(TIMEZONE)
            return (now - cache_time).total_seconds()
        except Exception:
            return None

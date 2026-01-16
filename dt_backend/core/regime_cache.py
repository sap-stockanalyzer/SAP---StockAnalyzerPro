"""
dt_backend/core/regime_cache.py

Pre-computed regime calculation cache for 50-100x performance boost in historical replay.

Caches:
- Regime classification results (regime_dt)
- Micro-regime results
- Daily plan metadata
- Market proxy features (trend, vol)

Usage (populate):
    >>> from dt_backend.core.regime_cache import populate_regime_cache
    >>> populate_regime_cache(start_date="2025-01-01", end_date="2025-01-15")

Usage (replay):
    >>> from dt_backend.core.regime_cache import load_cached_regime
    >>> regime_dt = load_cached_regime("2025-01-10")
"""

from __future__ import annotations

import json
import gzip
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from dt_backend.core.config_dt import DT_PATHS
    from dt_backend.core.data_pipeline_dt import log
except Exception:
    DT_PATHS: Dict[str, Path] = {
        "ml_data_dt": Path("ml_data_dt"),
    }
    
    def log(msg: str) -> None:
        print(msg, flush=True)


def _get_regime_cache_dir() -> Path:
    """Get the directory for regime cache."""
    root = DT_PATHS.get("ml_data_dt", Path("ml_data_dt"))
    cache_dir = root / "intraday" / "regime_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_path(date_str: str) -> Path:
    """Get the cache file path for a specific date."""
    cache_dir = _get_regime_cache_dir()
    return cache_dir / f"{date_str}.json.gz"


def save_regime_cache(
    date_str: str,
    regime_dt: Dict[str, Any],
    micro_regime_dt: Optional[Dict[str, Any]] = None,
    daily_plan_dt: Optional[Dict[str, Any]] = None,
    mkt_trend: float = 0.0,
    mkt_vol: float = 0.0,
) -> bool:
    """
    Save regime calculation results to cache.
    
    Args:
        date_str: Date string in ISO format (YYYY-MM-DD)
        regime_dt: Main regime classification result
        micro_regime_dt: Micro-regime result
        daily_plan_dt: Daily plan metadata
        mkt_trend: Market trend value
        mkt_vol: Market volatility value
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_path = _get_cache_path(date_str)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "date": date_str,
            "cached_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "regime_dt": regime_dt,
            "micro_regime_dt": micro_regime_dt,
            "daily_plan_dt": daily_plan_dt,
            "mkt_trend": float(mkt_trend),
            "mkt_vol": float(mkt_vol),
        }
        
        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        log(f"[regime_cache] ❌ Failed to save cache for {date_str}: {e}")
        return False


def load_cached_regime(date_str: str) -> Optional[Dict[str, Any]]:
    """
    Load cached regime results for a specific date.
    
    Args:
        date_str: Date string in ISO format (YYYY-MM-DD)
    
    Returns:
        Cached regime data or None if not found
    """
    try:
        cache_path = _get_cache_path(date_str)
        
        if not cache_path.exists():
            return None
        
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        return cache_data
        
    except Exception as e:
        log(f"[regime_cache] ⚠️ Failed to load cache for {date_str}: {e}")
        return None


def has_cached_regime(date_str: str) -> bool:
    """
    Check if a regime cache exists for a specific date.
    
    Args:
        date_str: Date string in ISO format
    
    Returns:
        True if cache exists, False otherwise
    """
    cache_path = _get_cache_path(date_str)
    return cache_path.exists()


def populate_regime_cache(
    start_date: str,
    end_date: str,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Pre-compute regime calculations for a date range.
    
    This function requires actual market data and should be called
    during data preparation, not during replay.
    
    Args:
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        force_recompute: If True, recompute even if cache exists
    
    Returns:
        Summary statistics about cache population
    """
    try:
        from datetime import datetime
        from dt_backend.core.regime_detector_dt import classify_intraday_regime
        from dt_backend.core.data_pipeline_dt import _read_rolling
        
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        
        stats = {
            "total_days": 0,
            "cached": 0,
            "skipped": 0,
            "failed": 0,
        }
        
        current = start
        while current <= end:
            date_str = current.isoformat()
            stats["total_days"] += 1
            
            # Skip if already cached and not forcing recompute
            if not force_recompute and has_cached_regime(date_str):
                stats["skipped"] += 1
                log(f"[regime_cache] ⏭️ Skipping {date_str} (already cached)")
                current += timedelta(days=1)
                continue
            
            try:
                # Compute regime (requires market data to be loaded)
                now_utc = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                regime_dt = classify_intraday_regime(now_utc=now_utc)
                
                # Extract additional data from rolling
                rolling = _read_rolling() or {}
                global_node = rolling.get("_GLOBAL_DT") or {}
                micro_regime_dt = global_node.get("micro_regime_dt")
                daily_plan_dt = global_node.get("daily_plan_dt")
                
                # Extract market proxies
                mkt_trend = regime_dt.get("mkt_trend", 0.0)
                mkt_vol = regime_dt.get("mkt_vol", 0.0)
                
                # Save to cache
                if save_regime_cache(
                    date_str,
                    regime_dt,
                    micro_regime_dt,
                    daily_plan_dt,
                    mkt_trend,
                    mkt_vol,
                ):
                    stats["cached"] += 1
                    log(f"[regime_cache] ✅ Cached {date_str}")
                else:
                    stats["failed"] += 1
                
            except Exception as e:
                stats["failed"] += 1
                log(f"[regime_cache] ❌ Failed to cache {date_str}: {e}")
            
            current += timedelta(days=1)
        
        log(f"[regime_cache] 📊 Completed: {stats}")
        return stats
        
    except Exception as e:
        log(f"[regime_cache] ❌ Failed to populate cache: {e}")
        return {"error": str(e)}


def list_cached_dates(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[str]:
    """
    List all cached dates within a range.
    
    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        List of date strings
    """
    try:
        cache_dir = _get_regime_cache_dir()
        
        if not cache_dir.exists():
            return []
        
        dates = []
        for cache_file in cache_dir.glob("*.json.gz"):
            # Extract date from filename (handle double extension)
            date_str = cache_file.name.removesuffix(".json.gz")
            
            # Apply filters
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            
            dates.append(date_str)
        
        return sorted(dates)
        
    except Exception as e:
        log(f"[regime_cache] ❌ Failed to list cached dates: {e}")
        return []


def clear_regime_cache(start_date: Optional[str] = None, end_date: Optional[str] = None) -> int:
    """
    Clear regime cache for a date range.
    
    Args:
        start_date: Optional start date (inclusive)
        end_date: Optional end date (inclusive)
    
    Returns:
        Number of cache files deleted
    """
    try:
        cache_dir = _get_regime_cache_dir()
        
        if not cache_dir.exists():
            return 0
        
        deleted_count = 0
        for cache_file in cache_dir.glob("*.json.gz"):
            # Extract date from filename (handle double extension)
            date_str = cache_file.name.removesuffix(".json.gz")
            
            # Apply filters
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            
            cache_file.unlink()
            deleted_count += 1
        
        log(f"[regime_cache] 🗑️ Deleted {deleted_count} cache files")
        return deleted_count
        
    except Exception as e:
        log(f"[regime_cache] ❌ Failed to clear cache: {e}")
        return 0


__all__ = [
    "save_regime_cache",
    "load_cached_regime",
    "has_cached_regime",
    "populate_regime_cache",
    "list_cached_dates",
    "clear_regime_cache",
]

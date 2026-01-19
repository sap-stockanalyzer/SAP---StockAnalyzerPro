# backend/routers/unified_cache_router.py
"""
Unified Cache API Router — AION Analytics

Provides a single endpoint for all frontend data needs.
Reduces API calls from 5+ to 1.
"""

from __future__ import annotations

from typing import Dict, Any

from fastapi import APIRouter

from backend.services.unified_cache_service import UnifiedCacheService

router = APIRouter(prefix="/api/cache", tags=["unified-cache"])


@router.get("/unified")
async def get_unified_cache() -> Dict[str, Any]:
    """
    Single endpoint providing all frontend data.
    
    Returns cached data that is updated every 5 seconds by background job.
    This eliminates the need for multiple API calls and prevents race conditions.
    
    Returns:
        {
            "timestamp": "ISO timestamp of cache generation",
            "cache_age_seconds": float (how old the cache is),
            "version": "1.0",
            "data": {
                "bots": { ... bots page bundle ... },
                "portfolio": { ... top holdings ... },
                "system": { ... system status ... }
            },
            "errors": { ... any errors during data collection ... }
        }
    """
    service = UnifiedCacheService()
    return service.get_cache()


@router.post("/unified/refresh")
async def refresh_unified_cache() -> Dict[str, Any]:
    """
    Manually trigger a cache refresh.
    
    Useful for testing or when immediate update is needed.
    Normally the cache is updated automatically every 5 seconds.
    
    Returns:
        The newly updated cache data.
    """
    service = UnifiedCacheService()
    return service.update_all()


@router.get("/unified/age")
async def get_cache_age() -> Dict[str, Any]:
    """
    Get the age of the current cache.
    
    Returns:
        {
            "age_seconds": float or null,
            "exists": bool
        }
    """
    service = UnifiedCacheService()
    age = service.get_cache_age()
    
    return {
        "age_seconds": age,
        "exists": age is not None,
    }

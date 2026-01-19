"""
Real-time Server-Sent Events (SSE) router for live data streaming.

Endpoints:
- GET /events/bots - Stream bots page bundle updates
- GET /events/admin/logs - Stream live admin logs
- GET /events/intraday - Stream intraday snapshot updates
- GET /events/replay/status - Stream replay status updates
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator, Dict, Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

try:
    from backend.core.config import TIMEZONE
except ImportError:
    from backend.config import TIMEZONE  # type: ignore

router = APIRouter(prefix="/events", tags=["events"])


async def _safe_call(func, *args, **kwargs) -> Dict[str, Any]:
    """Call a function safely and return error dict on failure."""
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}


@router.get("/bots")
async def stream_bots(request: Request):
    """
    Server-Sent Events stream for bots page data.
    Updates every 5 seconds.
    Enhanced error handling to keep stream alive even if data fetch fails.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        from backend.routers.bots_page_router import bots_page_bundle
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                try:
                    # Fetch latest data (cached internally)
                    data = await _safe_call(bots_page_bundle)
                    
                    # Reset error count on success
                    if "error" not in data:
                        consecutive_errors = 0
                    
                    # Send SSE event
                    yield f"data: {json.dumps(data)}\n\n"
                    
                except Exception as fetch_error:
                    # Log error but don't break the stream
                    consecutive_errors += 1
                    error_data = {
                        "error": str(fetch_error),
                        "error_type": type(fetch_error).__name__,
                        "timestamp": datetime.now(TIMEZONE).isoformat(),
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    
                    # If too many consecutive errors, break the stream
                    if consecutive_errors >= max_consecutive_errors:
                        break
                
                # Wait 5 seconds before next update
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            pass
        except Exception as stream_error:
            # Final fallback: send error and close
            error_data = {
                "error": f"Stream error: {stream_error}",
                "error_type": type(stream_error).__name__,
            }
            try:
                yield f"data: {json.dumps(error_data)}\n\n"
            except Exception:
                pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/admin/logs")
async def stream_admin_logs(request: Request):
    """
    Server-Sent Events stream for admin live logs.
    Updates every 2 seconds.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        from backend.admin.admin_tools_router import get_live_logs
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                logs = await _safe_call(get_live_logs)
                yield f"data: {json.dumps(logs)}\n\n"
                
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/intraday")
async def stream_intraday(request: Request):
    """
    Server-Sent Events stream for intraday snapshot.
    Updates every 5 seconds.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        from backend.intraday_service import get_intraday_snapshot
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                data = await _safe_call(get_intraday_snapshot, limit=120)
                yield f"data: {json.dumps(data)}\n\n"
                
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

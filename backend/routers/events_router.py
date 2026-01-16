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
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        from backend.routers.bots_page_router import bots_page_bundle
        
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                # Fetch latest data (cached internally)
                data = await _safe_call(bots_page_bundle)
                
                # Send SSE event
                yield f"data: {json.dumps(data)}\n\n"
                
                # Wait 5 seconds before next update
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

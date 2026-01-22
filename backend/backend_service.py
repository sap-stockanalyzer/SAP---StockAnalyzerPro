"""
backend.backend_service

Main FastAPI application for AION Analytics backend.
Consolidates all backend routers into a single unified service.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.error_handler import global_exception_handler

# Import working routers
from backend.routers.health_router import router as health_router
from backend.routers.events_router import router as events_router
from backend.routers.unified_cache_router import router as unified_cache_router
from backend.routers.admin_consolidated_router import router as admin_consolidated_router
from backend.routers.bots_page_router import router as bots_page_router
from backend.routers.dashboard_router import router as dashboard_router
from backend.routers.eod_bots_router import router as eod_bots_router
from backend.routers.system_run_router import router as system_run_router
from backend.admin.admin_tools_router import router as admin_tools_router

# Import testing router (optional)
try:
    from backend.routers.testing_router import router as testing_router
    testing_available = True
except ImportError:
    testing_router = None
    testing_available = False

# Routers commented out per requirements (not production-ready yet)
# from backend.routers.page_data_router import router as page_data_router  # COMMENTED OUT - not production-ready
# from backend.routers.settings_consolidated_router import router as settings_consolidated_router  # COMMENTED OUT - not production-ready


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AION Analytics Backend",
        version="2.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware - permissive for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    app.add_exception_handler(Exception, global_exception_handler)

    # Include all working routers
    ROUTERS = [
        # page_data_router,  # COMMENTED OUT - doesn't exist yet
        admin_consolidated_router,  # KEEP - works
        # settings_consolidated_router,  # COMMENTED OUT - doesn't exist yet
        health_router,
        events_router,
        unified_cache_router,
        bots_page_router,
        dashboard_router,
        eod_bots_router,
        admin_tools_router,
        system_run_router,
    ]
    
    # Add testing router if available
    if testing_available and testing_router:
        ROUTERS.append(testing_router)
    
    # Include all routers in the app
    for router in ROUTERS:
        app.include_router(router)

    return app


# Create the app instance
app = create_app()
"""dt_backend.api.app

FastAPI wrapper for dt_backend (intraday engine).
This layer is additive: it does NOT modify existing dt_backend modules.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dt_backend.api.routers import health, jobs, data
from dt_backend.routers import learning_router, emergency_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="AION dt_backend",
        version="1.0.0-fastapi",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS: permissive by default for local dev; tighten in prod via env if desired.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="")
    app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
    app.include_router(data.router, prefix="/data", tags=["data"])
    app.include_router(learning_router.router, prefix="")  # Already has /api/dt/learning prefix
    app.include_router(emergency_router.router, prefix="")  # Emergency stop endpoints
    return app

app = create_app()

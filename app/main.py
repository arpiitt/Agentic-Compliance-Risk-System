"""
FastAPI application entrypoint — production-grade, with clean startup.
All infrastructure failures are handled gracefully for local dev.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router
from app.config import get_settings
from app.retrieval.embeddings import init_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Graceful startup — each service is optional for local dev."""
    logger.info(" Starting Agentic Compliance & Risk Research Assistant...")

    # Postgres DB
    try:
        from app.db.database import init_db
        await init_db()
        logger.info(" Database initialized")
    except Exception as exc:
        logger.warning("  Database unavailable (trace persistence disabled): %s", exc)

    # Redis
    try:
        from app.cache.redis_cache import get_redis
        await get_redis()
        logger.info(" Redis connected")
    except Exception as exc:
        logger.warning("  Redis unavailable (caching disabled): %s", exc)

    # Qdrant
    try:
        from app.retrieval.qdrant_client import ensure_collection_exists
        await ensure_collection_exists()
        logger.info(" Qdrant collection ready")
    except Exception as exc:
        logger.warning("  Qdrant unavailable (policy KB search disabled): %s", exc)

    # Gemini
    init_embeddings()
    logger.info(" Gemini SDK initialized")

    logger.info(" App ready on http://localhost:8000")
    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        from app.cache.redis_cache import close_redis
        await close_redis()
    except Exception:
        pass
    try:
        from app.db.database import close_db
        await close_db()
    except Exception:
        pass


#  Build app 
app = FastAPI(
    title="Agentic Compliance & Risk Research Assistant",
    description="Multi-agent LangGraph system for cited, verified SEC compliance risk reports.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount Prometheus /metrics endpoint directly (avoids slowapi + instrumentator conflicts)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routes
app.include_router(router, prefix="")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)[:300]},
    )

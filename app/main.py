import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from app.api.v1.api import api_router
from app.core.clients.qdrant import ensure_collections_exist
from app.core.clients.embedding import EmbeddingError
from app.services.acronym_service import load_acronym_cache
from app.utils.json_handler import CustomJSONResponse
from app.config import settings
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    try:
        await ensure_collections_exist()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

    try:
        # load_acronym_cache() is synchronous (blocking Postgres + Redis I/O) —
        # must run in a thread, not be awaited directly, or it stalls the event
        # loop (and every other in-flight request) for its full duration.
        await run_in_threadpool(load_acronym_cache)
    except Exception as e:
        # Not fatal: get_expansions_batch() already falls back to Postgres per lookup,
        # so a failed warm-up only costs a few extra DB round-trips on first
        # touch — it must not take down the whole service (e.g. a fresh
        # deploy where the migration hasn't run yet, or a transient Redis
        # blip, would otherwise prevent boot entirely).
        logger.warning(f"Acronym cache warm-up failed, continuing without it: {str(e)}")

    logger.info("Application startup completed")

    yield

    try:
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")


# Conditionally set root_path based on environment
root_path_config = "" if settings.ENVIRONMENT == "local" else "/vector"

app = FastAPI(
    title="Vector Search API",
    description="API for document ingestion, search, and management with vector database",
    version="1.0.0",
    lifespan=lifespan,
    root_path=root_path_config,
    default_response_class=CustomJSONResponse
)

app.include_router(api_router, prefix="/api")


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request, exc: EmbeddingError):
    """Map embedding/query-vector validation failures to a clear HTTP 422.

    Without this, an empty/malformed query vector surfaced as an opaque Qdrant
    ``400 Vector dimension error: expected dim: 384, got 0`` (or an unhandled 500).
    """
    logger.warning(f"Embedding validation failed for {request.url.path}: {exc}")
    return CustomJSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    from app.core.clients.qdrant import qdrant_client
    from app.core.clients.redis_cache import redis_cache
    import redis
    from fastapi import HTTPException

    try:
        qdrant_client.get_collections()
        redis_cache.redis_client.ping()
        return {
            "status": "healthy",
            "services": {"qdrant": "connected", "redis": "connected"},
        }
    except redis.ConnectionError as e:
        logger.error(f"Redis connection failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Redis connection failed")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Request
from sqlalchemy import text

from src.api.dependencies import DBSession
from src.api.schemas.common import HealthResponse

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request, session: DBSession) -> HealthResponse:
    """Comprehensive system health check."""
    app_state = request.app.state
    start_time: float = getattr(app_state, "start_time", time.time())

    db_ok = await _check_db(session)
    redis_ok = await _check_redis(app_state)
    model_loaded = getattr(app_state, "ensemble", None) is not None
    active_models = getattr(app_state, "active_model_names", [])

    overall = "healthy" if (db_ok and model_loaded) else "degraded"

    return HealthResponse(
        status=overall,
        db_connected=db_ok,
        redis_connected=redis_ok,
        model_loaded=model_loaded,
        active_models=list(active_models),
        uptime_seconds=round(time.time() - start_time, 2),
    )


@router.get("/health/models")
async def model_info(request: Request) -> dict:
    """Return details about currently loaded models."""
    app_state = request.app.state
    ensemble = getattr(app_state, "ensemble", None)
    if ensemble is None:
        return {"loaded": False, "models": []}

    model_names = list(ensemble.models.keys()) if hasattr(ensemble, "models") else []
    model_types = ensemble.model_types if hasattr(ensemble, "model_types") else {}
    weights = ensemble.weights if hasattr(ensemble, "weights") else {}

    return {
        "loaded": True,
        "models": [
            {
                "name": name,
                "type": model_types.get(name, "unknown"),
                "weight": weights.get(name) if weights else None,
            }
            for name in model_names
        ],
        "has_meta_learner": (
            hasattr(ensemble, "meta_learner") and ensemble.meta_learner is not None
        ),
    }


async def _check_db(session) -> bool:
    try:
        await session.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("health.db_check_failed", exc_info=True)
        return False


async def _check_redis(app_state) -> bool:
    redis_client = getattr(app_state, "redis_client", None)
    if redis_client is None:
        return False
    try:
        return await redis_client.ping()
    except Exception:
        logger.warning("health.redis_check_failed", exc_info=True)
        return False

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from src.api.v1.admin import router as admin_router
from src.api.v1.alerts import router as alerts_router
from src.api.v1.health import router as health_router
from src.api.v1.historical import router as historical_router
from src.api.v1.predictions import router as predictions_router
from src.config import load_app_config, load_region_config
from src.db.engine import _init_defaults as init_db
from src.ml.uncertainty import UncertaintyEstimator
from src.streaming.redis_client import RedisStreamClient

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise DB pool, load models, connect Redis.
    Shutdown: release all resources.
    """
    app.state.start_time = time.time()
    app.state.alert_subscribers = []

    cfg = load_app_config()

    engine, _ = init_db()
    app.state.db_engine = engine
    logger.info("startup.db_initialised")

    region_configs: dict[str, dict] = {}
    for region_code in cfg.active_regions:
        try:
            rc = load_region_config(region_code)
            region_configs[region_code] = {
                "risk_thresholds": rc.risk_thresholds,
                "model_weights": rc.model_weights,
            }
        except FileNotFoundError:
            region_configs[region_code] = {
                "risk_thresholds": {"GREEN": 0.25, "YELLOW": 0.55, "ORANGE": 0.80},
            }
            logger.warning("startup.region_config_missing", region=region_code)

    app.state.region_configs = region_configs
    app.state.active_regions = cfg.active_regions

    app.state.ensemble = None
    app.state.active_model_names = []
    app.state.uncertainty_estimator = UncertaintyEstimator(
        n_mc_samples=cfg.uncertainty_mc_samples,
        confidence_level=0.90,
    )
    logger.info("startup.models_ready", ensemble_loaded=False)

    redis_client = RedisStreamClient(cfg.redis)
    try:
        await redis_client.connect()
        app.state.redis_client = redis_client
        logger.info("startup.redis_connected")
    except Exception:
        app.state.redis_client = None
        logger.warning("startup.redis_unavailable", exc_info=True)

    logger.info("startup.complete", active_regions=cfg.active_regions)

    yield

    if app.state.redis_client is not None:
        await app.state.redis_client.disconnect()

    if hasattr(app.state, "db_engine") and app.state.db_engine is not None:
        await app.state.db_engine.dispose()

    logger.info("shutdown.complete")


def create_app() -> FastAPI:
    """Application factory for the Outage Prediction API."""
    cfg = load_app_config()
    app = FastAPI(
        title="Outage Prediction API",
        version="1.0.0",
        description=(
            "Research-prototype API for outage-risk experiments. The committed app "
            "does not load a trained model at startup and is not an operational utility service."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_second=20.0, burst=50)

    v1_prefix = "/api/v1"
    app.include_router(predictions_router, prefix=v1_prefix)
    app.include_router(alerts_router, prefix=v1_prefix)
    app.include_router(historical_router, prefix=v1_prefix)
    app.include_router(health_router, prefix=v1_prefix)
    if cfg.enable_admin_routes:
        app.include_router(admin_router, prefix=v1_prefix)

    @app.get("/")
    async def root() -> dict:
        return {
            "status": "ok",
            "service": "Outage Prediction API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app

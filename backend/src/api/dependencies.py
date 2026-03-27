from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.db.engine import get_db_session as _get_db_session
from backend.src.services.alert_service import AlertService
from backend.src.services.prediction_service import PredictionService


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in _get_db_session():
        yield session


async def get_prediction_service(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> PredictionService:
    app_state = request.app.state
    ensemble = getattr(app_state, "ensemble", None)
    uncertainty_estimator = getattr(app_state, "uncertainty_estimator", None)
    redis_client = getattr(app_state, "redis_client", None)
    region_configs = getattr(app_state, "region_configs", {})

    return PredictionService(
        session=session,
        ensemble=ensemble,
        uncertainty_estimator=uncertainty_estimator,
        redis_client=redis_client,
        region_configs=region_configs,
    )


async def get_alert_service(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> AlertService:
    redis_client = getattr(request.app.state, "redis_client", None)
    return AlertService(session=session, redis_client=redis_client)


DBSession = Annotated[AsyncSession, Depends(get_db_session)]
PredictionDep = Annotated[PredictionService, Depends(get_prediction_service)]
AlertDep = Annotated[AlertService, Depends(get_alert_service)]

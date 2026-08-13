from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Query

from src.api.dependencies import DBSession, PredictionDep
from src.api.schemas.predictions import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResult,
)
from src.db.models import Prediction
from src.db.repositories import PredictionRepository

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/realtime", response_model=PredictionResult)
async def predict_realtime(
    body: PredictionRequest,
    service: PredictionDep,
) -> PredictionResult:
    """Run a real-time prediction for a single H3 cell.

    Computes features, runs the ensemble, estimates uncertainty, and
    classifies the risk level. If severity >= YELLOW an alert is created
    automatically.
    """
    return await service.predict_realtime(
        h3_cell=body.h3_cell,
        region=body.region,
        features_override=body.features_override,
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(body: BatchPredictionRequest) -> BatchPredictionResponse:
    """Submit a batch prediction job.

    In production this would enqueue work onto a Celery/ARQ task queue.
    For now the job is accepted and assigned a tracking ID.
    """
    job_id = str(uuid.uuid4())
    return BatchPredictionResponse(
        job_id=job_id,
        status="queued",
        results=None,
    )


@router.get("/history", response_model=list[PredictionResult])
async def prediction_history(
    session: DBSession,
    region: str | None = Query(default=None),
    h3_cell: str | None = Query(default=None),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=5000),
) -> list[PredictionResult]:
    """Retrieve historical predictions filtered by region, cell, and time range."""
    repo = PredictionRepository(session)

    if h3_cell and start_time and end_time:
        rows = await repo.get_history(
            h3_index=h3_cell,
            start=start_time,
            end=end_time,
            limit=limit,
        )
    elif region:
        rows = await repo.get_latest_by_region(region_code=region, limit=limit)
    else:
        rows = await repo.get_latest_by_region(region_code="TX", limit=limit)

    return [_row_to_result(r) for r in rows]


def _row_to_result(row: Prediction) -> PredictionResult:
    from src.api.schemas.predictions import UncertaintyEstimate

    lower = row.uncertainty_lower or 0.0
    upper = row.uncertainty_upper or 1.0
    prob = row.risk_probability or 0.0

    return PredictionResult(
        prediction_id=str(row.id),
        h3_cell=row.h3_index_res7 or "",
        region=row.region_code or "",
        risk_probability=prob,
        uncertainty=UncertaintyEstimate(
            lower=lower,
            upper=upper,
            aleatoric=0.0,
            epistemic=max(0.0, (upper - lower) / 2),
            confidence_level=0.90,
        ),
        risk_level=row.risk_level or "GREEN",
        model_version=row.model_version or "unknown",
        top_features=None,
        computed_at=row.predicted_at,
    )

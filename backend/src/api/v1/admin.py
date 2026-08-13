from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import and_, update

from src.api.dependencies import DBSession
from src.db.models import ModelRegistryEntry

router = APIRouter(prefix="/admin", tags=["admin"])


class ThresholdUpdate(BaseModel):
    region: str
    green_max: float = 0.25
    yellow_max: float = 0.55
    orange_max: float = 0.80


class ModelPromoteRequest(BaseModel):
    model_name: str
    version: str
    region: str | None = None


@router.put("/thresholds")
async def update_thresholds(body: ThresholdUpdate, request: Request) -> dict:
    """Update risk classification thresholds for a region.

    Thresholds are stored in the application state and persisted to the
    region config on next config sync cycle.
    """
    if body.green_max >= body.yellow_max or body.yellow_max >= body.orange_max:
        raise HTTPException(
            status_code=422,
            detail="Thresholds must satisfy green_max < yellow_max < orange_max",
        )

    region_configs: dict = getattr(request.app.state, "region_configs", {})
    if body.region not in region_configs:
        region_configs[body.region] = {}

    region_configs[body.region]["risk_thresholds"] = {
        "GREEN": body.green_max,
        "YELLOW": body.yellow_max,
        "ORANGE": body.orange_max,
    }
    request.app.state.region_configs = region_configs

    return {
        "region": body.region,
        "thresholds": region_configs[body.region]["risk_thresholds"],
        "updated_at": datetime.now(tz=UTC).isoformat(),
    }


@router.post("/models/promote")
async def promote_model(
    body: ModelPromoteRequest,
    session: DBSession,
) -> dict:
    """Promote a model version to active status.

    Deactivates any currently active model with the same name and region,
    then marks the specified version as active.
    """
    deactivate_conditions = [ModelRegistryEntry.model_name == body.model_name]
    if body.region:
        deactivate_conditions.append(ModelRegistryEntry.region_code == body.region)

    await session.execute(
        update(ModelRegistryEntry)
        .where(and_(*deactivate_conditions, ModelRegistryEntry.is_active.is_(True)))
        .values(is_active=False)
    )

    result = await session.execute(
        update(ModelRegistryEntry)
        .where(
            and_(
                ModelRegistryEntry.model_name == body.model_name,
                ModelRegistryEntry.version == body.version,
            )
        )
        .values(is_active=True, promoted_at=datetime.now(tz=UTC))
        .returning(ModelRegistryEntry.id)
    )
    promoted = result.scalar_one_or_none()
    if promoted is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model {body.model_name} version {body.version} not found",
        )

    return {
        "model_name": body.model_name,
        "version": body.version,
        "region": body.region,
        "promoted_at": datetime.now(tz=UTC).isoformat(),
    }


@router.get("/config")
async def get_active_config(request: Request) -> dict:
    """Return the active runtime configuration."""
    app_state = request.app.state
    region_configs = getattr(app_state, "region_configs", {})
    active_regions = getattr(app_state, "active_regions", [])

    return {
        "active_regions": active_regions,
        "region_configs": {
            code: {
                "risk_thresholds": cfg.get("risk_thresholds", {}),
            }
            for code, cfg in region_configs.items()
        },
        "model_loaded": getattr(app_state, "ensemble", None) is not None,
        "active_model_names": getattr(app_state, "active_model_names", []),
    }

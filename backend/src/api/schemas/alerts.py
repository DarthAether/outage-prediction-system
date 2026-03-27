from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AlertResponse(BaseModel):
    id: int
    severity: str
    region_code: str
    h3_cell: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    uncertainty_range: float = Field(ge=0.0)
    description: str
    recommended_actions: list[str]
    created_at: datetime
    expires_at: datetime | None = None
    acknowledged: bool = False

    model_config = {"from_attributes": True}


class AlertAcknowledge(BaseModel):
    acknowledged_by: str = Field(
        min_length=1, description="Identifier of the person or system acknowledging"
    )

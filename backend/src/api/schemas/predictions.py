from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UncertaintyEstimate(BaseModel):
    lower: float = Field(ge=0.0, le=1.0, description="Lower bound of confidence interval")
    upper: float = Field(ge=0.0, le=1.0, description="Upper bound of confidence interval")
    aleatoric: float = Field(ge=0.0, description="Data-inherent noise uncertainty")
    epistemic: float = Field(ge=0.0, description="Model structural uncertainty")
    confidence_level: float = Field(default=0.90, ge=0.0, le=1.0, description="CI confidence level")


class PredictionRequest(BaseModel):
    h3_cell: str = Field(min_length=1, description="H3 cell index at resolution 7")
    region: str = Field(min_length=1, description="Region code (e.g. TX)")
    features_override: dict[str, float] | None = Field(
        default=None,
        description="Optional feature overrides for what-if analysis",
    )


class PredictionResult(BaseModel):
    prediction_id: str
    h3_cell: str
    region: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    uncertainty: UncertaintyEstimate
    risk_level: Literal["GREEN", "YELLOW", "ORANGE", "RED"]
    model_version: str
    top_features: list[dict[str, float]] | None = Field(
        default=None,
        description="Top contributing features with importance scores",
    )
    computed_at: datetime


class BatchPredictionRequest(BaseModel):
    region: str = Field(min_length=1)
    h3_cells: list[str] | None = Field(
        default=None,
        description="Specific cells; if None the entire region is processed",
    )
    timestamp: datetime | None = Field(
        default=None,
        description="Override prediction reference time",
    )


class BatchPredictionResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    results: list[PredictionResult] | None = None

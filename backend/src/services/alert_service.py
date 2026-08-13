from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.alerts import AlertResponse
from src.api.schemas.predictions import PredictionResult
from src.db.repositories import AlertRepository

logger = structlog.get_logger(__name__)

RISK_DESCRIPTIONS: dict[str, str] = {
    "RED": (
        "Critical outage risk detected. Severe weather and grid stress indicate "
        "high probability of widespread power disruption."
    ),
    "ORANGE": (
        "Elevated outage risk. Significant weather activity combined with "
        "infrastructure vulnerability warrants proactive measures."
    ),
    "YELLOW": (
        "Moderate outage risk. Weather conditions may cause localized disruptions. "
        "Monitor situation closely."
    ),
}

RECOMMENDED_ACTIONS: dict[str, list[str]] = {
    "RED": [
        "Activate emergency operations center",
        "Pre-position line crews in affected area",
        "Issue public safety advisory for the region",
        "Coordinate with hospitals and critical facilities for backup power",
        "Stage mobile generators at priority substations",
        "Notify mutual aid partners for potential resource requests",
    ],
    "ORANGE": [
        "Alert field crews and pre-stage repair equipment",
        "Verify backup generator readiness at critical facilities",
        "Increase monitoring frequency for affected substations",
        "Review vegetation management status along at-risk corridors",
        "Prepare customer communication templates",
    ],
    "YELLOW": [
        "Increase monitoring of weather forecast updates",
        "Review crew availability for next 24-48 hours",
        "Confirm outage management system readiness",
        "Verify communication channels with field personnel",
    ],
}

ALERT_EXPIRY_HOURS: dict[str, int] = {
    "RED": 12,
    "ORANGE": 24,
    "YELLOW": 48,
}


class AlertService:
    """Manages the lifecycle of outage risk alerts."""

    def __init__(
        self,
        session: AsyncSession,
        redis_client: Any | None = None,
    ) -> None:
        self._session = session
        self._repo = AlertRepository(session)
        self._redis = redis_client

    async def create_alert(self, prediction: PredictionResult) -> AlertResponse:
        """Create an alert from a prediction result.

        Only call this when risk_level is YELLOW or above.
        """
        level = prediction.risk_level
        now = datetime.now(tz=UTC)
        expiry_hours = ALERT_EXPIRY_HOURS.get(level, 24)
        expires_at = now + timedelta(hours=expiry_hours)

        uncertainty_range = prediction.uncertainty.upper - prediction.uncertainty.lower

        alert_data = {
            "region_code": prediction.region,
            "h3_index_res7": prediction.h3_cell,
            "severity": level,
            "risk_probability": prediction.risk_probability,
            "uncertainty_range": round(uncertainty_range, 4),
            "description": RISK_DESCRIPTIONS.get(level, "Outage risk detected."),
            "recommended_actions": RECOMMENDED_ACTIONS.get(level, []),
            "expires_at": expires_at,
        }

        row = await self._repo.create(alert_data)

        response = AlertResponse(
            id=row.id,
            severity=row.severity or level,
            region_code=row.region_code or prediction.region,
            h3_cell=row.h3_index_res7 or prediction.h3_cell,
            risk_probability=row.risk_probability or prediction.risk_probability,
            uncertainty_range=row.uncertainty_range or uncertainty_range,
            description=row.description or "",
            recommended_actions=row.recommended_actions or [],
            created_at=row.created_at,
            expires_at=row.expires_at,
            acknowledged=row.acknowledged,
        )

        await self._broadcast_alert(response)

        logger.info(
            "alert.created",
            alert_id=row.id,
            severity=level,
            region=prediction.region,
            h3_cell=prediction.h3_cell,
        )

        return response

    async def get_active_alerts(
        self,
        region: str | None = None,
        severity: str | None = None,
    ) -> list[AlertResponse]:
        rows = await self._repo.get_active(region_code=region, severity=severity)
        return [
            AlertResponse(
                id=r.id,
                severity=r.severity or "YELLOW",
                region_code=r.region_code or "",
                h3_cell=r.h3_index_res7 or "",
                risk_probability=r.risk_probability or 0.0,
                uncertainty_range=r.uncertainty_range or 0.0,
                description=r.description or "",
                recommended_actions=r.recommended_actions or [],
                created_at=r.created_at,
                expires_at=r.expires_at,
                acknowledged=r.acknowledged,
            )
            for r in rows
        ]

    async def acknowledge(self, alert_id: int, by: str) -> AlertResponse:
        row = await self._repo.acknowledge(alert_id, acknowledged_by=by)
        if row is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return AlertResponse(
            id=row.id,
            severity=row.severity or "YELLOW",
            region_code=row.region_code or "",
            h3_cell=row.h3_index_res7 or "",
            risk_probability=row.risk_probability or 0.0,
            uncertainty_range=row.uncertainty_range or 0.0,
            description=row.description or "",
            recommended_actions=row.recommended_actions or [],
            created_at=row.created_at,
            expires_at=row.expires_at,
            acknowledged=row.acknowledged,
        )

    async def _broadcast_alert(self, alert: AlertResponse) -> None:
        payload = alert.model_dump_json()

        if self._redis is not None:
            try:
                await self._redis.publish_alert(alert)
            except Exception:
                logger.warning("alert.redis_publish_failed", exc_info=True)

        subscribers = getattr(self, "_ws_subscribers", None)
        if subscribers is None:
            return
        for queue in list(subscribers):
            with contextlib.suppress(Exception):
                queue.put_nowait(payload)

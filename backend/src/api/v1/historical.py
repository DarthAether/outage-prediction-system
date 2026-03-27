from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.dependencies import DBSession
from backend.src.db.models import (
    ModelRegistryEntry,
    OutageObservation,
    Prediction,
    WeatherEvent,
)

router = APIRouter(prefix="/historical", tags=["historical"])


@router.get("/outages")
async def historical_outages(
    session: DBSession,
    region: str | None = Query(default=None),
    county_fips: str | None = Query(default=None),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=10_000),
) -> list[dict]:
    """Return historical outage observations aggregated by county and time."""
    now = datetime.now(tz=timezone.utc)
    start = start_time or (now - timedelta(days=30))
    end = end_time or now

    conditions = [
        OutageObservation.observed_at >= start,
        OutageObservation.observed_at <= end,
    ]
    if county_fips:
        conditions.append(OutageObservation.county_fips == county_fips)
    if region:
        conditions.append(OutageObservation.state_fips == region)

    stmt = (
        select(OutageObservation)
        .where(and_(*conditions))
        .order_by(OutageObservation.observed_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "id": r.id,
            "observed_at": r.observed_at.isoformat() if r.observed_at else None,
            "county_fips": r.county_fips,
            "state_fips": r.state_fips,
            "customers_out": r.customers_out,
            "total_customers": r.total_customers,
            "outage_fraction": r.outage_fraction,
            "h3_cell": r.h3_index_res7,
        }
        for r in rows
    ]


@router.get("/weather")
async def historical_weather(
    session: DBSession,
    region: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=10_000),
) -> list[dict]:
    """Return historical weather events filtered by region and type."""
    now = datetime.now(tz=timezone.utc)
    start = start_time or (now - timedelta(days=30))
    end = end_time or now

    conditions = [
        WeatherEvent.event_time >= start,
        WeatherEvent.event_time <= end,
    ]
    if region:
        conditions.append(WeatherEvent.state_fips == region)
    if event_type:
        conditions.append(WeatherEvent.event_type == event_type)

    stmt = (
        select(WeatherEvent)
        .where(and_(*conditions))
        .order_by(WeatherEvent.event_time.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "id": r.id,
            "event_time": r.event_time.isoformat() if r.event_time else None,
            "event_type": r.event_type,
            "magnitude": r.magnitude,
            "h3_cell": r.h3_index_res7,
            "state_fips": r.state_fips,
            "county_fips": r.county_fips,
            "damage_property": float(r.damage_property) if r.damage_property else None,
            "injuries": r.injuries,
            "deaths": r.deaths,
        }
        for r in rows
    ]


@router.get("/model-performance")
async def model_performance(
    session: DBSession,
    region: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict]:
    """Return model metrics over time from the model registry."""
    conditions = []
    if region:
        conditions.append(ModelRegistryEntry.region_code == region)

    stmt = (
        select(ModelRegistryEntry)
        .where(and_(*conditions)) if conditions else select(ModelRegistryEntry)
    )
    stmt = stmt.order_by(ModelRegistryEntry.created_at.desc()).limit(limit)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "model_name": r.model_name,
            "version": r.version,
            "region_code": r.region_code,
            "metrics": r.metrics,
            "is_active": r.is_active,
            "promoted_at": r.promoted_at.isoformat() if r.promoted_at else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]

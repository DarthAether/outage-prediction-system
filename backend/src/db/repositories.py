from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.db.models import Alert, OutageObservation, Prediction, WeatherEvent


class WeatherEventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert_batch(self, events: list[dict]) -> int:
        if not events:
            return 0
        stmt = pg_insert(WeatherEvent).values(events)
        stmt = stmt.on_conflict_do_nothing()
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    async def get_by_region_and_timerange(
        self,
        state_fips: str,
        start: datetime,
        end: datetime,
        event_type: str | None = None,
        limit: int = 10_000,
    ) -> Sequence[WeatherEvent]:
        conditions = [
            WeatherEvent.state_fips == state_fips,
            WeatherEvent.event_time >= start,
            WeatherEvent.event_time <= end,
        ]
        if event_type is not None:
            conditions.append(WeatherEvent.event_type == event_type)
        stmt = (
            select(WeatherEvent)
            .where(and_(*conditions))
            .order_by(WeatherEvent.event_time.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_by_h3_and_timerange(
        self,
        h3_index: str,
        start: datetime,
        end: datetime,
        resolution: int = 7,
    ) -> Sequence[WeatherEvent]:
        col = (
            WeatherEvent.h3_index_res7
            if resolution == 7
            else WeatherEvent.h3_index_res9
        )
        stmt = (
            select(WeatherEvent)
            .where(
                and_(
                    col == h3_index,
                    WeatherEvent.event_time >= start,
                    WeatherEvent.event_time <= end,
                )
            )
            .order_by(WeatherEvent.event_time.desc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()


class OutageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert_batch(self, observations: list[dict]) -> int:
        if not observations:
            return 0
        stmt = pg_insert(OutageObservation).values(observations)
        stmt = stmt.on_conflict_do_nothing()
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    async def get_by_county_and_timerange(
        self,
        county_fips: str,
        start: datetime,
        end: datetime,
        limit: int = 10_000,
    ) -> Sequence[OutageObservation]:
        stmt = (
            select(OutageObservation)
            .where(
                and_(
                    OutageObservation.county_fips == county_fips,
                    OutageObservation.observed_at >= start,
                    OutageObservation.observed_at <= end,
                )
            )
            .order_by(OutageObservation.observed_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_outage_fraction(
        self,
        county_fips: str,
        start: datetime,
        end: datetime,
    ) -> float | None:
        stmt = select(func.avg(OutageObservation.outage_fraction)).where(
            and_(
                OutageObservation.county_fips == county_fips,
                OutageObservation.observed_at >= start,
                OutageObservation.observed_at <= end,
            )
        )
        result = await self._session.execute(stmt)
        value = result.scalar_one_or_none()
        return float(value) if value is not None else None


class PredictionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(self, prediction: dict) -> Prediction:
        row = Prediction(**prediction)
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_history(
        self,
        h3_index: str,
        start: datetime,
        end: datetime,
        model_version: str | None = None,
        limit: int = 1_000,
    ) -> Sequence[Prediction]:
        conditions = [
            Prediction.h3_index_res7 == h3_index,
            Prediction.predicted_at >= start,
            Prediction.predicted_at <= end,
        ]
        if model_version is not None:
            conditions.append(Prediction.model_version == model_version)
        stmt = (
            select(Prediction)
            .where(and_(*conditions))
            .order_by(Prediction.predicted_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_latest_by_region(
        self,
        region_code: str,
        limit: int = 500,
    ) -> Sequence[Prediction]:
        subq = (
            select(
                Prediction.h3_index_res7,
                func.max(Prediction.predicted_at).label("max_ts"),
            )
            .where(Prediction.region_code == region_code)
            .group_by(Prediction.h3_index_res7)
            .subquery()
        )
        stmt = (
            select(Prediction)
            .join(
                subq,
                and_(
                    Prediction.h3_index_res7 == subq.c.h3_index_res7,
                    Prediction.predicted_at == subq.c.max_ts,
                ),
            )
            .where(Prediction.region_code == region_code)
            .order_by(Prediction.risk_probability.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()


class AlertRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, alert_data: dict) -> Alert:
        row = Alert(**alert_data)
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_active(
        self,
        region_code: str | None = None,
        severity: str | None = None,
    ) -> Sequence[Alert]:
        conditions = [
            Alert.acknowledged.is_(False),
            (Alert.expires_at.is_(None)) | (Alert.expires_at > func.now()),
        ]
        if region_code is not None:
            conditions.append(Alert.region_code == region_code)
        if severity is not None:
            conditions.append(Alert.severity == severity)
        stmt = (
            select(Alert)
            .where(and_(*conditions))
            .order_by(Alert.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def acknowledge(
        self,
        alert_id: int,
        acknowledged_by: str,
    ) -> Alert | None:
        stmt = (
            update(Alert)
            .where(Alert.id == alert_id)
            .values(
                acknowledged=True,
                acknowledged_by=acknowledged_by,
                acknowledged_at=func.now(),
            )
            .returning(Alert)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def cleanup_expired(self) -> int:
        stmt = delete(Alert).where(
            and_(
                Alert.expires_at.isnot(None),
                Alert.expires_at < func.now(),
                Alert.acknowledged.is_(True),
            )
        )
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

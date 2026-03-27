"""Ingestion scheduler for periodic and one-time data loading.

Manages the orchestration of all data source ingestors with support for
backfill operations, periodic scheduled runs, and full-system ingestion.
Uses asyncio for concurrent execution of independent data sources.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .base import BaseIngestor, IngestionResult
from .census import CensusIngestor
from .eagle_i import EagleIIngestor
from .eia import EiaIngestor
from .ercot import ErcotIngestor
from .metar import MetarIngestor
from .noaa_storms import NoaaStormsIngestor
from .nws_alerts import NwsAlertsIngestor

logger = structlog.get_logger(__name__)

INGESTOR_REGISTRY: dict[str, Callable[..., BaseIngestor]] = {
    "noaa_storms": NoaaStormsIngestor,
    "eagle_i": EagleIIngestor,
    "nws_alerts": NwsAlertsIngestor,
    "ercot": ErcotIngestor,
    "census": CensusIngestor,
    "eia": EiaIngestor,
    "metar": MetarIngestor,
}

DEFAULT_SCHEDULE: dict[str, dict] = {
    "nws_alerts": {"interval_minutes": 15, "description": "Active NWS alerts (near real-time)"},
    "metar": {"interval_minutes": 60, "description": "METAR surface observations (hourly)"},
    "ercot": {"interval_minutes": 60, "description": "ERCOT grid load data (hourly)"},
    "eagle_i": {"interval_minutes": 30, "description": "EAGLE-I outage data (every 30 min)"},
    "noaa_storms": {"interval_minutes": 1440, "description": "NOAA storm events (daily)"},
    "census": {"interval_minutes": 0, "description": "Census data (one-time/annual)"},
    "eia": {"interval_minutes": 0, "description": "EIA infrastructure (one-time/annual)"},
}


@dataclass
class ScheduleEntry:
    """Tracks the schedule state for a single data source."""
    source_name: str
    interval_minutes: int
    last_run: datetime | None = None
    next_run: datetime | None = None
    last_result: IngestionResult | None = None
    enabled: bool = True
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5

    @property
    def is_due(self) -> bool:
        if not self.enabled or self.interval_minutes <= 0:
            return False
        if self.next_run is None:
            return True
        return datetime.utcnow() >= self.next_run

    def mark_complete(self, result: IngestionResult) -> None:
        self.last_run = datetime.utcnow()
        self.last_result = result
        if result.success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.enabled = False
                logger.error(
                    "scheduler.source_disabled",
                    source=self.source_name,
                    failures=self.consecutive_failures,
                )
        self.next_run = self.last_run + timedelta(minutes=self.interval_minutes)


class IngestionScheduler:
    """Manages periodic and one-time ingestion of all data sources.

    Provides methods for backfilling historical data, running single
    periodic updates, and scheduling continuous ingestion loops.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        data_dir: str = "data/raw",
        default_region: str | None = "TX",
        ingestor_kwargs: dict[str, dict] | None = None,
    ):
        self.session_factory = session_factory
        self.data_dir = data_dir
        self.default_region = default_region
        self.ingestor_kwargs = ingestor_kwargs or {}
        self._schedules: dict[str, ScheduleEntry] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

        self._init_schedules()

    def _init_schedules(self) -> None:
        """Initialize schedule entries from the default schedule config."""
        for source, config in DEFAULT_SCHEDULE.items():
            self._schedules[source] = ScheduleEntry(
                source_name=source,
                interval_minutes=config["interval_minutes"],
            )

    def _create_ingestor(self, source: str) -> BaseIngestor:
        """Create an ingestor instance for the given source."""
        factory = INGESTOR_REGISTRY.get(source)
        if not factory:
            raise ValueError(f"Unknown data source: {source}")

        kwargs = self.ingestor_kwargs.get(source, {})

        if source in ("noaa_storms", "eagle_i", "ercot", "eia"):
            kwargs.setdefault("data_dir", self.data_dir)
        if source in ("nws_alerts", "metar"):
            kwargs.setdefault("target_states", [self.default_region] if self.default_region else ["TX"])

        return factory(**kwargs)

    async def run_backfill(
        self,
        source: str,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
        chunk_days: int = 30,
    ) -> list[IngestionResult]:
        """Run a backfill for a source over a date range, chunked by time.

        Splits large date ranges into smaller chunks to avoid memory issues
        and provide incremental progress logging.
        """
        region = region_code or self.default_region
        results: list[IngestionResult] = []

        logger.info(
            "scheduler.backfill_start",
            source=source,
            start=str(start_date),
            end=str(end_date),
            region=region,
        )

        ingestor = self._create_ingestor(source)
        current_start = start_date

        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)

            async with self.session_factory() as session:
                try:
                    result = await ingestor.run(
                        start_date=current_start,
                        end_date=current_end,
                        session=session,
                        region_code=region,
                    )
                    results.append(result)

                    logger.info(
                        "scheduler.backfill_chunk",
                        source=source,
                        chunk_start=str(current_start),
                        chunk_end=str(current_end),
                        loaded=result.records_loaded,
                        success=result.success,
                    )
                except Exception as e:
                    logger.error(
                        "scheduler.backfill_chunk_failed",
                        source=source,
                        chunk_start=str(current_start),
                        error=str(e),
                    )

            current_start = current_end + timedelta(days=1)

        total_loaded = sum(r.records_loaded for r in results)
        total_dropped = sum(r.records_dropped for r in results)
        logger.info(
            "scheduler.backfill_complete",
            source=source,
            chunks=len(results),
            total_loaded=total_loaded,
            total_dropped=total_dropped,
        )

        return results

    async def run_periodic(
        self,
        source: str,
        region_code: str | None = None,
        lookback_hours: int | None = None,
    ) -> IngestionResult:
        """Run a single periodic ingestion for a source.

        Uses the schedule's last_run time to determine the fetch window,
        or falls back to a default lookback period.
        """
        region = region_code or self.default_region
        schedule = self._schedules.get(source)

        if lookback_hours:
            end_date = date.today()
            start_date = end_date - timedelta(hours=lookback_hours)
        elif schedule and schedule.last_run:
            start_date = schedule.last_run.date()
            end_date = date.today()
        else:
            interval = DEFAULT_SCHEDULE.get(source, {}).get("interval_minutes", 1440)
            lookback = max(interval / 60, 24)
            end_date = date.today()
            start_date = end_date - timedelta(hours=lookback)

        ingestor = self._create_ingestor(source)

        async with self.session_factory() as session:
            result = await ingestor.run(
                start_date=start_date,
                end_date=end_date,
                session=session,
                region_code=region,
            )

        if schedule:
            schedule.mark_complete(result)

        logger.info(
            "scheduler.periodic_complete",
            source=source,
            loaded=result.records_loaded,
            success=result.success,
        )

        return result

    async def run_all_periodic(
        self,
        region_code: str | None = None,
    ) -> dict[str, IngestionResult]:
        """Run periodic ingestion for all due sources concurrently."""
        region = region_code or self.default_region
        results: dict[str, IngestionResult] = {}

        due_sources = [
            name for name, entry in self._schedules.items()
            if entry.is_due
        ]

        if not due_sources:
            logger.info("scheduler.no_sources_due")
            return results

        logger.info("scheduler.running_due_sources", sources=due_sources)

        tasks = []
        for source in due_sources:
            task = asyncio.create_task(
                self.run_periodic(source, region_code=region),
                name=f"ingest_{source}",
            )
            tasks.append((source, task))

        for source, task in tasks:
            try:
                result = await task
                results[source] = result
            except Exception as e:
                logger.error(
                    "scheduler.source_failed",
                    source=source,
                    error=str(e),
                )
                schedule = self._schedules.get(source)
                if schedule:
                    schedule.consecutive_failures += 1

        return results

    async def schedule_all(
        self,
        region_code: str | None = None,
        check_interval_seconds: int = 60,
    ) -> None:
        """Run the continuous scheduling loop.

        Checks for due sources every check_interval_seconds and runs them.
        Runs until stop() is called.
        """
        self._running = True
        region = region_code or self.default_region

        logger.info(
            "scheduler.loop_started",
            check_interval=check_interval_seconds,
            region=region,
            sources=list(self._schedules.keys()),
        )

        try:
            while self._running:
                try:
                    results = await self.run_all_periodic(region_code=region)

                    for source, result in results.items():
                        logger.info(
                            "scheduler.source_result",
                            source=source,
                            success=result.success,
                            loaded=result.records_loaded,
                        )

                except Exception as e:
                    logger.error("scheduler.loop_error", error=str(e))

                await asyncio.sleep(check_interval_seconds)

        except asyncio.CancelledError:
            logger.info("scheduler.loop_cancelled")
        finally:
            self._running = False
            logger.info("scheduler.loop_stopped")

    def stop(self) -> None:
        """Signal the scheduling loop to stop."""
        self._running = False

    def get_status(self) -> dict[str, dict]:
        """Return current status of all scheduled sources."""
        status: dict[str, dict] = {}
        for name, entry in self._schedules.items():
            status[name] = {
                "enabled": entry.enabled,
                "interval_minutes": entry.interval_minutes,
                "last_run": str(entry.last_run) if entry.last_run else None,
                "next_run": str(entry.next_run) if entry.next_run else None,
                "consecutive_failures": entry.consecutive_failures,
                "last_success": (
                    entry.last_result.success if entry.last_result else None
                ),
                "last_records_loaded": (
                    entry.last_result.records_loaded if entry.last_result else None
                ),
                "description": DEFAULT_SCHEDULE.get(name, {}).get("description", ""),
            }
        return status

    def enable_source(self, source: str) -> None:
        """Re-enable a disabled source."""
        if source in self._schedules:
            self._schedules[source].enabled = True
            self._schedules[source].consecutive_failures = 0
            logger.info("scheduler.source_enabled", source=source)

    def disable_source(self, source: str) -> None:
        """Disable a source from periodic runs."""
        if source in self._schedules:
            self._schedules[source].enabled = False
            logger.info("scheduler.source_disabled_manual", source=source)

    def set_interval(self, source: str, interval_minutes: int) -> None:
        """Update the polling interval for a source."""
        if source in self._schedules:
            self._schedules[source].interval_minutes = interval_minutes
            logger.info(
                "scheduler.interval_updated",
                source=source,
                interval_minutes=interval_minutes,
            )

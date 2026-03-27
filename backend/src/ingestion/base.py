"""Base ingestion interface for all data source ingestors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.invalid_records / self.total_records


@dataclass
class IngestionResult:
    source: str
    start_date: date
    end_date: date
    records_fetched: int
    records_validated: int
    records_loaded: int
    records_dropped: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.records_loaded > 0


class BaseIngestor(ABC):
    """Abstract base for all data source ingestors.

    Each ingestor implements the ETL pattern:
    fetch -> validate -> transform -> load
    """

    source_name: str = "unknown"

    @abstractmethod
    async def fetch(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Fetch raw data from the source for the given date range."""
        ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate the raw data. Returns cleaned DataFrame and validation report."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform validated data into the target schema."""
        ...

    @abstractmethod
    async def load(self, df: pd.DataFrame, session: AsyncSession) -> int:
        """Load transformed data into the database. Returns count of inserted records."""
        ...

    async def run(
        self,
        start_date: date,
        end_date: date,
        session: AsyncSession,
        region_code: str | None = None,
    ) -> IngestionResult:
        """Execute the full ETL pipeline: fetch -> validate -> transform -> load."""
        import time

        t0 = time.monotonic()
        errors: list[str] = []

        logger.info(
            "ingestion.start",
            source=self.source_name,
            start_date=str(start_date),
            end_date=str(end_date),
            region=region_code,
        )

        try:
            raw_df = await self.fetch(start_date, end_date, region_code)
            records_fetched = len(raw_df)
            logger.info("ingestion.fetched", source=self.source_name, records=records_fetched)

            if raw_df.empty:
                return IngestionResult(
                    source=self.source_name,
                    start_date=start_date,
                    end_date=end_date,
                    records_fetched=0,
                    records_validated=0,
                    records_loaded=0,
                    records_dropped=0,
                    duration_seconds=time.monotonic() - t0,
                    errors=["No data returned from source"],
                )

            clean_df, validation = self.validate(raw_df)
            if not validation.valid and validation.valid_records == 0:
                return IngestionResult(
                    source=self.source_name,
                    start_date=start_date,
                    end_date=end_date,
                    records_fetched=records_fetched,
                    records_validated=0,
                    records_loaded=0,
                    records_dropped=validation.invalid_records,
                    duration_seconds=time.monotonic() - t0,
                    errors=validation.errors,
                )

            if validation.warnings:
                for w in validation.warnings:
                    logger.warning("ingestion.validation_warning", source=self.source_name, msg=w)

            transformed_df = self.transform(clean_df)
            records_loaded = await self.load(transformed_df, session)

            duration = time.monotonic() - t0
            logger.info(
                "ingestion.complete",
                source=self.source_name,
                fetched=records_fetched,
                loaded=records_loaded,
                dropped=validation.invalid_records,
                duration_s=round(duration, 2),
            )

            return IngestionResult(
                source=self.source_name,
                start_date=start_date,
                end_date=end_date,
                records_fetched=records_fetched,
                records_validated=validation.valid_records,
                records_loaded=records_loaded,
                records_dropped=validation.invalid_records,
                duration_seconds=duration,
                errors=errors,
            )

        except Exception as e:
            duration = time.monotonic() - t0
            logger.error("ingestion.failed", source=self.source_name, error=str(e))
            return IngestionResult(
                source=self.source_name,
                start_date=start_date,
                end_date=end_date,
                records_fetched=0,
                records_validated=0,
                records_loaded=0,
                records_dropped=0,
                duration_seconds=duration,
                errors=[str(e)],
            )

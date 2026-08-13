from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Region(Base):
    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    boundary = mapped_column(Geometry("MULTIPOLYGON", srid=4326), nullable=True)
    config_path: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")


class WeatherEvent(Base):
    __tablename__ = "weather_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_time: Mapped[datetime] = mapped_column(nullable=False)
    source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    event_type: Mapped[str | None] = mapped_column(String(60), nullable=True)
    magnitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    magnitude_type: Mapped[str | None] = mapped_column(String(10), nullable=True)
    location = mapped_column(Geometry("POINT", srid=4326), nullable=True)
    h3_index_res7: Mapped[str | None] = mapped_column(String(15), nullable=True)
    h3_index_res9: Mapped[str | None] = mapped_column(String(15), nullable=True)
    state_fips: Mapped[str | None] = mapped_column(String(5), nullable=True)
    county_fips: Mapped[str | None] = mapped_column(String(5), nullable=True)
    damage_property: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    damage_crops: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    injuries: Mapped[int] = mapped_column(Integer, default=0)
    deaths: Mapped[int] = mapped_column(Integer, default=0)
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    episode_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    raw_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")


class OutageObservation(Base):
    __tablename__ = "outage_observations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    observed_at: Mapped[datetime] = mapped_column(nullable=False)
    county_fips: Mapped[str | None] = mapped_column(String(5), nullable=True)
    state_fips: Mapped[str | None] = mapped_column(String(2), nullable=True)
    customers_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_customers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outage_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    h3_index_res7: Mapped[str | None] = mapped_column(String(15), nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="eagle_i")
    ingested_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")


class GridLoad(Base):
    __tablename__ = "grid_load"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    recorded_at: Mapped[datetime] = mapped_column(nullable=False)
    region_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    load_mw: Mapped[float | None] = mapped_column(Float, nullable=True)
    capacity_mw: Mapped[float | None] = mapped_column(Float, nullable=True)
    reserve_margin_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    frequency_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")


class Infrastructure(Base):
    __tablename__ = "infrastructure"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    h3_index_res7: Mapped[str | None] = mapped_column(String(15), nullable=True)
    county_fips: Mapped[str | None] = mapped_column(String(5), nullable=True)
    transmission_line_km: Mapped[float] = mapped_column(Float, default=0)
    distribution_line_km: Mapped[float] = mapped_column(Float, default=0)
    substations_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_line_age_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    vegetation_density: Mapped[float | None] = mapped_column(Float, nullable=True)
    land_cover_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")


class Socioeconomic(Base):
    __tablename__ = "socioeconomic"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    h3_index_res7: Mapped[str | None] = mapped_column(String(15), nullable=True)
    county_fips: Mapped[str | None] = mapped_column(String(5), nullable=True)
    population_density: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    critical_facilities_count: Mapped[int] = mapped_column(Integer, default=0)
    housing_age_median: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")


class FeatureStoreEntry(Base):
    __tablename__ = "feature_store"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    computed_at: Mapped[datetime] = mapped_column(nullable=False)
    h3_index_res7: Mapped[str | None] = mapped_column(String(15), nullable=True)
    feature_version: Mapped[str | None] = mapped_column(String(10), nullable=True)
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    target_outage_occurred: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    target_customers_affected: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    predicted_at: Mapped[datetime] = mapped_column(nullable=False)
    h3_index_res7: Mapped[str | None] = mapped_column(String(15), nullable=True)
    region_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    features_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    region_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    h3_index_res7: Mapped[str | None] = mapped_column(String(15), nullable=True)
    severity: Mapped[str | None] = mapped_column(String(10), nullable=True)
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_range: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_actions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(nullable=True)


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("model_name", "version", name="uq_model_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    region_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")

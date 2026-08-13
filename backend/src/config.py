from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "outage_prediction"
    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")

    @property
    def dsn(self) -> str:
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_dsn(self) -> str:
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    stream_name: str = "predictions"

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}"


class RegionConfig(BaseModel):
    code: str
    name: str
    states: list[str]
    risk_thresholds: dict[str, float]
    dominant_weather_types: list[str]
    model_weights: dict[str, float] | None = None
    feature_importance_overrides: dict[str, float] | None = None
    seasonal_adjustments: dict[str, dict[str, float]] | None = None


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_")

    db: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    mlflow_tracking_uri: str = "http://localhost:5000"
    active_regions: list[str] = ["TX"]
    default_h3_resolution: int = 7
    prediction_horizon_hours: int = 24
    uncertainty_mc_samples: int = 50
    batch_prediction_interval_minutes: int = 60
    cors_origins: list[str] = ["http://localhost:3000"]
    enable_admin_routes: bool = False


def load_region_config(
    region_code: str,
    config_dir: str = "config/regions",
) -> RegionConfig:
    config_path = Path(config_dir)
    candidates = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
    for filepath in candidates:
        with open(filepath) as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
        if data.get("code") == region_code:
            return RegionConfig(**data)
    raise FileNotFoundError(
        f"No region config found for '{region_code}' in {config_path.resolve()}"
    )


@functools.lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    base_path = Path("config/base.yaml")
    overrides: dict[str, Any] = {}
    if base_path.exists():
        with open(base_path) as fh:
            overrides = yaml.safe_load(fh) or {}
    return AppConfig(**overrides)

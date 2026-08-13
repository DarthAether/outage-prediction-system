from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from redis.asyncio import Redis

from src.api.schemas.alerts import AlertResponse
from src.api.schemas.predictions import PredictionResult
from src.config import RedisConfig

logger = structlog.get_logger(__name__)

PREDICTION_CHANNEL = "outage:predictions"
ALERT_CHANNEL = "outage:alerts"


class RedisStreamClient:
    """Async Redis client for publishing and subscribing to prediction/alert streams."""

    def __init__(self, config: RedisConfig | None = None) -> None:
        self._config = config or RedisConfig()
        self._redis: Redis | None = None
        self._pubsub: Any | None = None

    async def connect(self) -> None:
        self._redis = Redis.from_url(
            self._config.url,
            decode_responses=True,
            max_connections=20,
        )
        logger.info("redis.connected", url=self._config.url)

    async def disconnect(self) -> None:
        if self._pubsub is not None:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
            self._pubsub = None
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
        logger.info("redis.disconnected")

    async def ping(self) -> bool:
        if self._redis is None:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False

    async def publish_prediction(self, prediction: PredictionResult) -> int:
        if self._redis is None:
            raise RuntimeError("Redis client is not connected")
        payload = prediction.model_dump_json()
        count = await self._redis.publish(PREDICTION_CHANNEL, payload)
        logger.debug(
            "redis.prediction_published",
            prediction_id=prediction.prediction_id,
            subscribers=count,
        )
        return count

    async def publish_alert(self, alert: AlertResponse) -> int:
        if self._redis is None:
            raise RuntimeError("Redis client is not connected")
        payload = alert.model_dump_json()
        count = await self._redis.publish(ALERT_CHANNEL, payload)
        logger.debug(
            "redis.alert_published",
            alert_id=alert.id,
            severity=alert.severity,
            subscribers=count,
        )
        return count

    async def add_to_stream(self, prediction: PredictionResult) -> str:
        """Append a prediction to a Redis Stream for durable ordered storage."""
        if self._redis is None:
            raise RuntimeError("Redis client is not connected")
        entry = {
            "prediction_id": prediction.prediction_id,
            "h3_cell": prediction.h3_cell,
            "region": prediction.region,
            "risk_probability": str(prediction.risk_probability),
            "risk_level": prediction.risk_level,
            "computed_at": prediction.computed_at.isoformat(),
        }
        message_id = await self._redis.xadd(self._config.stream_name, entry, maxlen=100_000)
        return message_id

    async def subscribe_alerts(self) -> AsyncGenerator[AlertResponse, None]:
        """Subscribe to the alert channel and yield AlertResponse objects."""
        if self._redis is None:
            raise RuntimeError("Redis client is not connected")

        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(ALERT_CHANNEL)

        try:
            async for message in self._pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                    yield AlertResponse(**data)
                except (json.JSONDecodeError, TypeError, KeyError):
                    logger.warning(
                        "redis.alert_parse_failed",
                        raw=str(message["data"])[:200],
                    )
        finally:
            await self._pubsub.unsubscribe(ALERT_CHANNEL)

    async def subscribe_predictions(self) -> AsyncGenerator[PredictionResult, None]:
        """Subscribe to the prediction channel and yield PredictionResult objects."""
        if self._redis is None:
            raise RuntimeError("Redis client is not connected")

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(PREDICTION_CHANNEL)

        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                    yield PredictionResult(**data)
                except (json.JSONDecodeError, TypeError, KeyError):
                    logger.warning(
                        "redis.prediction_parse_failed",
                        raw=str(message["data"])[:200],
                    )
        finally:
            await pubsub.unsubscribe(PREDICTION_CHANNEL)
            await pubsub.close()

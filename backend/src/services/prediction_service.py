from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.schemas.predictions import (
    PredictionResult,
    UncertaintyEstimate,
)
from backend.src.db.repositories import PredictionRepository
from backend.src.ml.ensemble import OutageEnsemble
from backend.src.ml.uncertainty import UncertaintyEstimator

logger = structlog.get_logger(__name__)

DEFAULT_THRESHOLDS = {
    "GREEN": 0.25,
    "YELLOW": 0.55,
    "ORANGE": 0.80,
}

MODEL_VERSION = "ensemble-v1.0"

FEATURE_DEFAULTS: dict[str, float] = {
    "weather_count_24h": 0.0,
    "weather_max_mag_24h": 0.0,
    "weather_mean_mag_24h": 0.0,
    "weather_distinct_types_24h": 0.0,
    "lag_outage_24h": 0.0,
    "lag_outage_48h": 0.0,
    "current_load_mw": 0.0,
    "reserve_margin_pct": 0.15,
    "load_capacity_ratio": 0.7,
    "neighbor_weather_count": 0.0,
    "neighbor_outage_mean": 0.0,
    "transmission_line_km": 0.0,
    "distribution_line_km": 0.0,
    "substations_count": 0,
    "avg_line_age_years": 20.0,
    "vegetation_density": 0.3,
    "population_density": 100.0,
    "median_income_normalized": 0.5,
    "critical_facility_density": 0.0,
    "housing_age_median": 30.0,
    "hour_sin": 0.0,
    "hour_cos": 1.0,
    "dow_sin": 0.0,
    "dow_cos": 1.0,
    "month_sin": 0.0,
    "month_cos": 1.0,
    "is_weekend": 0.0,
}


class PredictionService:
    """Orchestrates real-time outage risk prediction.

    Pipeline: feature computation -> ensemble prediction ->
    uncertainty estimation -> risk classification -> optional alert creation.
    """

    def __init__(
        self,
        session: AsyncSession,
        ensemble: OutageEnsemble | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        redis_client: Any | None = None,
        region_configs: dict[str, dict] | None = None,
    ) -> None:
        self._session = session
        self._ensemble = ensemble
        self._uncertainty = uncertainty_estimator or UncertaintyEstimator()
        self._redis = redis_client
        self._region_configs = region_configs or {}
        self._pred_repo = PredictionRepository(session)

    async def predict_realtime(
        self,
        h3_cell: str,
        region: str,
        features_override: dict[str, float] | None = None,
    ) -> PredictionResult:
        prediction_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc)

        features = self._compute_features(h3_cell, region, now, features_override)
        feature_names = sorted(features.keys())
        x_tabular = np.array([[features[f] for f in feature_names]])

        if self._ensemble is not None:
            risk_prob, uncertainty_pred = self._run_ensemble(x_tabular)
        else:
            risk_prob, uncertainty_pred = self._fallback_prediction(features)

        thresholds = self._get_thresholds(region)
        risk_level = self._classify_risk(risk_prob, thresholds)

        top_features = self._compute_top_features(features, feature_names)

        uncertainty = UncertaintyEstimate(
            lower=uncertainty_pred["lower"],
            upper=uncertainty_pred["upper"],
            aleatoric=uncertainty_pred["aleatoric"],
            epistemic=uncertainty_pred["epistemic"],
            confidence_level=uncertainty_pred.get("confidence_level", 0.90),
        )

        await self._persist_prediction(
            prediction_id, h3_cell, region, risk_prob, uncertainty, risk_level, features, now
        )

        result = PredictionResult(
            prediction_id=prediction_id,
            h3_cell=h3_cell,
            region=region,
            risk_probability=risk_prob,
            uncertainty=uncertainty,
            risk_level=risk_level,
            model_version=MODEL_VERSION,
            top_features=top_features,
            computed_at=now,
        )

        if risk_level != "GREEN" and self._redis is not None:
            try:
                await self._redis.publish_prediction(result)
            except Exception:
                logger.warning("prediction.redis_publish_failed", exc_info=True)

        logger.info(
            "prediction.completed",
            prediction_id=prediction_id,
            h3_cell=h3_cell,
            region=region,
            risk_probability=risk_prob,
            risk_level=risk_level,
        )

        return result

    def _compute_features(
        self,
        h3_cell: str,
        region: str,
        timestamp: datetime,
        overrides: dict[str, float] | None,
    ) -> dict[str, float]:
        features = dict(FEATURE_DEFAULTS)

        hour = timestamp.hour
        dow = timestamp.weekday()
        month = timestamp.month
        features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        features["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
        features["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
        features["month_sin"] = float(np.sin(2 * np.pi * month / 12))
        features["month_cos"] = float(np.cos(2 * np.pi * month / 12))
        features["is_weekend"] = float(dow >= 5)

        if overrides:
            for key, value in overrides.items():
                if key in features:
                    features[key] = value

        return features

    def _run_ensemble(
        self, x_tabular: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        ensemble_proba = self._ensemble.predict_proba(X_tabular=x_tabular)
        risk_prob = float(np.clip(ensemble_proba[0], 0.0, 1.0))

        member_preds = self._ensemble._get_member_predictions(X_tabular=x_tabular)
        pred_matrix = np.column_stack(list(member_preds.values()))

        uncertainty_results = self._uncertainty.predict_with_uncertainty(
            ensemble_preds=pred_matrix,
            mc_preds=None,
        )
        u = uncertainty_results[0]

        return risk_prob, {
            "lower": u.ci_lower,
            "upper": u.ci_upper,
            "aleatoric": u.aleatoric,
            "epistemic": u.epistemic,
            "confidence_level": u.confidence_level,
        }

    @staticmethod
    def _fallback_prediction(
        features: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        weather_score = min(1.0, features.get("weather_count_24h", 0) / 10.0)
        outage_score = features.get("lag_outage_24h", 0.0)
        load_score = max(0.0, features.get("load_capacity_ratio", 0.7) - 0.7) / 0.3
        infra_age = features.get("avg_line_age_years", 20.0) / 50.0

        risk_prob = float(np.clip(
            0.3 * weather_score + 0.25 * outage_score + 0.25 * load_score + 0.2 * infra_age,
            0.0, 1.0,
        ))
        spread = 0.15
        return risk_prob, {
            "lower": max(0.0, risk_prob - spread),
            "upper": min(1.0, risk_prob + spread),
            "aleatoric": spread * 0.6,
            "epistemic": spread * 0.4,
            "confidence_level": 0.90,
        }

    def _get_thresholds(self, region: str) -> dict[str, float]:
        region_cfg = self._region_configs.get(region, {})
        return region_cfg.get("risk_thresholds", DEFAULT_THRESHOLDS)

    @staticmethod
    def _classify_risk(probability: float, thresholds: dict[str, float]) -> str:
        if probability >= thresholds.get("ORANGE", 0.80):
            return "RED"
        if probability >= thresholds.get("YELLOW", 0.55):
            return "ORANGE"
        if probability >= thresholds.get("GREEN", 0.25):
            return "YELLOW"
        return "GREEN"

    @staticmethod
    def _compute_top_features(
        features: dict[str, float],
        feature_names: list[str],
    ) -> list[dict[str, float]]:
        scored = []
        for name in feature_names:
            val = features.get(name, 0.0)
            default = FEATURE_DEFAULTS.get(name, 0.0)
            deviation = abs(val - default) if default != 0 else abs(val)
            scored.append({name: round(deviation, 6)})

        scored.sort(key=lambda d: list(d.values())[0], reverse=True)
        return scored[:10]

    async def _persist_prediction(
        self,
        prediction_id: str,
        h3_cell: str,
        region: str,
        risk_prob: float,
        uncertainty: UncertaintyEstimate,
        risk_level: str,
        features: dict,
        timestamp: datetime,
    ) -> None:
        try:
            await self._pred_repo.insert({
                "predicted_at": timestamp,
                "h3_index_res7": h3_cell,
                "region_code": region,
                "model_version": MODEL_VERSION,
                "risk_probability": risk_prob,
                "uncertainty_lower": uncertainty.lower,
                "uncertainty_upper": uncertainty.upper,
                "risk_level": risk_level,
                "features_snapshot": features,
            })
        except Exception:
            logger.warning("prediction.persist_failed", exc_info=True)

"""Feature Store for Outage Prediction.

Orchestrates all feature builders to compute, cache, and retrieve
feature vectors for model training and real-time inference.
"""

from datetime import datetime

import pandas as pd
import structlog

from .compound_events import CompoundEventFeatureBuilder
from .spatial import SpatialFeatureBuilder
from .temporal import TemporalFeatureBuilder

logger = structlog.get_logger(__name__)


class FeatureStore:
    """Manages feature computation, storage, and retrieval.

    Coordinates temporal, spatial, compound event, and socioeconomic
    feature builders into a unified feature vector per H3 cell and timestamp.
    """

    VERSION = "v1.0"

    def __init__(self, h3_resolution: int = 7):
        self.temporal = TemporalFeatureBuilder()
        self.spatial = SpatialFeatureBuilder(h3_resolution)
        self.compound = CompoundEventFeatureBuilder(h3_resolution)
        self.h3_resolution = h3_resolution

    def compute_features(
        self,
        h3_cell: str,
        timestamp: datetime,
        weather_df: pd.DataFrame,
        outage_df: pd.DataFrame,
        load_df: pd.DataFrame,
        infrastructure_data: dict | None = None,
        socioeconomic_data: dict | None = None,
    ) -> dict[str, float]:
        """Compute all features for a single prediction point.

        Args:
            h3_cell: Target H3 cell index.
            timestamp: Prediction reference time.
            weather_df: Weather events in the cell and neighborhood.
            outage_df: Outage observations in the cell and neighborhood.
            load_df: Grid load data for the region.
            infrastructure_data: Static infrastructure info for the cell.
            socioeconomic_data: Census-derived data for the cell.

        Returns:
            Flat dict of all feature name -> value pairs.
        """
        features: dict[str, float] = {}

        # Temporal features
        temporal_feats = self.temporal.compute_all(weather_df, outage_df, load_df, timestamp)
        features.update(temporal_feats)

        # Spatial features
        spatial_feats = self.spatial.compute_all(
            h3_cell, weather_df, outage_df, timestamp, infrastructure_data
        )
        features.update(spatial_feats)

        # Compound event features (research contribution 1)
        compound_feats = self.compound.compute_all(weather_df, timestamp)
        features.update(compound_feats)

        # Socioeconomic features
        if socioeconomic_data:
            features["population_density"] = float(
                socioeconomic_data.get("population_density", 0) or 0
            )
            features["median_income_normalized"] = (
                float(socioeconomic_data.get("median_income", 0) or 0) / 100_000.0
            )
            features["critical_facility_density"] = float(
                socioeconomic_data.get("critical_facilities_count", 0) or 0
            )
            features["housing_age_median"] = float(
                socioeconomic_data.get("housing_age_median", 0) or 0
            )
        else:
            features["population_density"] = 0.0
            features["median_income_normalized"] = 0.0
            features["critical_facility_density"] = 0.0
            features["housing_age_median"] = 0.0

        return features

    def build_training_dataset(
        self,
        weather_df: pd.DataFrame,
        outage_df: pd.DataFrame,
        load_df: pd.DataFrame,
        h3_cells: list[str],
        timestamps: list[datetime],
        infrastructure_lookup: dict[str, dict] | None = None,
        socioeconomic_lookup: dict[str, dict] | None = None,
    ) -> pd.DataFrame:
        """Materialize a full training DataFrame from raw data.

        Iterates over all H3 cells and timestamps to build the
        feature matrix with corresponding target labels.

        Args:
            weather_df: All weather events with h3_index_res7.
            outage_df: All outage observations with h3_index_res7.
            load_df: Grid load time series.
            h3_cells: List of H3 cells to compute features for.
            timestamps: List of timestamps for each prediction point.
            infrastructure_lookup: Dict mapping h3_cell -> infrastructure data.
            socioeconomic_lookup: Dict mapping h3_cell -> socioeconomic data.

        Returns:
            DataFrame where each row is a feature vector with target columns.
        """
        rows = []
        total = len(h3_cells) * len(timestamps)
        processed = 0

        for cell in h3_cells:
            infra = (infrastructure_lookup or {}).get(cell)
            socio = (socioeconomic_lookup or {}).get(cell)

            for ts in timestamps:
                features = self.compute_features(
                    cell, ts, weather_df, outage_df, load_df, infra, socio
                )
                features["h3_cell"] = cell
                features["timestamp"] = ts

                # Target: did an outage occur in the next 24 hours?
                target_window_start = ts
                target_window_end = ts + pd.Timedelta(hours=24)
                if not outage_df.empty and "h3_index_res7" in outage_df.columns:
                    target_mask = (
                        (outage_df["h3_index_res7"] == cell)
                        & (outage_df["observed_at"] >= target_window_start)
                        & (outage_df["observed_at"] <= target_window_end)
                        & (outage_df["outage_fraction"] > 0.01)
                    )
                    features["target_outage"] = int(target_mask.any())
                    matching = outage_df.loc[target_mask]
                    features["target_max_outage_fraction"] = (
                        float(matching["outage_fraction"].max()) if not matching.empty else 0.0
                    )
                else:
                    features["target_outage"] = 0
                    features["target_max_outage_fraction"] = 0.0

                rows.append(features)
                processed += 1

                if processed % 1000 == 0:
                    logger.info(
                        "feature_store.progress",
                        processed=processed,
                        total=total,
                        pct=round(100 * processed / total, 1),
                    )

        df = pd.DataFrame(rows)
        logger.info(
            "feature_store.built",
            rows=len(df),
            features=len(df.columns) - 4,  # exclude h3, timestamp, targets
            positive_rate=round(df["target_outage"].mean(), 4)
            if "target_outage" in df.columns
            else 0,
        )
        return df

    @staticmethod
    def get_feature_groups() -> dict[str, list[str]]:
        """Return feature group definitions for ablation studies.

        Each group corresponds to a conceptual category that can be
        removed independently to measure its contribution.
        """
        return {
            "temporal": [
                c
                for c in [
                    "weather_count_1h",
                    "weather_count_3h",
                    "weather_count_6h",
                    "weather_count_12h",
                    "weather_count_24h",
                    "weather_count_48h",
                    "weather_count_72h",
                    "weather_max_mag_1h",
                    "weather_max_mag_3h",
                    "weather_max_mag_6h",
                    "weather_max_mag_12h",
                    "weather_max_mag_24h",
                    "weather_max_mag_48h",
                    "weather_max_mag_72h",
                    "weather_mean_mag_1h",
                    "weather_mean_mag_3h",
                    "weather_mean_mag_6h",
                    "weather_mean_mag_12h",
                    "weather_mean_mag_24h",
                    "weather_mean_mag_48h",
                    "weather_mean_mag_72h",
                    "weather_distinct_types_1h",
                    "weather_distinct_types_3h",
                    "weather_distinct_types_6h",
                    "weather_distinct_types_12h",
                    "weather_distinct_types_24h",
                    "weather_distinct_types_48h",
                    "weather_distinct_types_72h",
                    "lag_outage_1h",
                    "lag_outage_3h",
                    "lag_outage_6h",
                    "lag_outage_12h",
                    "lag_outage_24h",
                    "lag_outage_48h",
                    "lag_outage_168h",
                    "hour_sin",
                    "hour_cos",
                    "dow_sin",
                    "dow_cos",
                    "month_sin",
                    "month_cos",
                    "is_weekend",
                    "trend_outage_3h",
                    "trend_outage_6h",
                    "trend_outage_12h",
                    "current_load_mw",
                    "reserve_margin_pct",
                    "load_capacity_ratio",
                ]
            ],
            "spatial": [
                "neighbor_weather_count",
                "neighbor_weather_max_mag",
                "neighbor_weather_mean_mag",
                "neighbor_outage_mean",
                "neighbor_outage_max",
                "neighbor_outage_spread",
                "transmission_line_km",
                "distribution_line_km",
                "substations_count",
                "avg_line_age_years",
                "vegetation_density",
                "line_density_per_km2",
                "infrastructure_age_risk",
            ],
            "compound": [
                c
                for c in []  # dynamically populated from compound builder output
            ],
            "socioeconomic": [
                "population_density",
                "median_income_normalized",
                "critical_facility_density",
                "housing_age_median",
            ],
        }

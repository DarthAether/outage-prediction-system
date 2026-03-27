"""Tests for spatial feature engineering with H3 indexing."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.spatial import (
    SpatialFeatureBuilder,
    get_h3_neighbors,
    h3_cell_area_km2,
    h3_to_lat_lon,
    lat_lon_to_h3,
)


class TestH3Utilities:
    def test_lat_lon_to_h3_valid(self):
        # Austin, TX
        cell = lat_lon_to_h3(30.2672, -97.7431, resolution=7)
        assert isinstance(cell, str)
        assert len(cell) > 0

    def test_h3_neighbors_includes_center(self):
        cell = lat_lon_to_h3(30.2672, -97.7431, resolution=7)
        neighbors = get_h3_neighbors(cell, k=1)
        assert cell in neighbors
        assert len(neighbors) == 7  # center + 6 neighbors

    def test_h3_neighbors_k2(self):
        cell = lat_lon_to_h3(30.2672, -97.7431, resolution=7)
        neighbors = get_h3_neighbors(cell, k=2)
        assert len(neighbors) == 19  # 1 + 6 + 12

    def test_h3_to_lat_lon_roundtrip(self):
        lat, lon = 30.2672, -97.7431
        cell = lat_lon_to_h3(lat, lon, resolution=7)
        recovered_lat, recovered_lon = h3_to_lat_lon(cell)
        assert abs(recovered_lat - lat) < 0.1
        assert abs(recovered_lon - lon) < 0.1

    def test_cell_area_resolution_7(self):
        area = h3_cell_area_km2(7)
        assert abs(area - 5.161) < 0.01


class TestNeighborhoodAggregation:
    def setup_method(self):
        self.builder = SpatialFeatureBuilder(h3_resolution=7)

    def test_empty_data_returns_zeros(self):
        empty_weather = pd.DataFrame(columns=["event_time", "h3_index_res7", "magnitude"])
        empty_outage = pd.DataFrame(columns=["observed_at", "h3_index_res7", "outage_fraction"])
        ts = datetime(2023, 6, 15, 12, 0)

        features = self.builder.neighborhood_aggregation(
            "87489e346ffffff", empty_weather, empty_outage, ts
        )

        assert features["neighbor_weather_count"] == 0.0
        assert features["neighbor_outage_mean"] == 0.0
        assert features["neighbor_outage_spread"] == 0.0


class TestInfrastructureFeatures:
    def setup_method(self):
        self.builder = SpatialFeatureBuilder(h3_resolution=7)

    def test_with_data(self):
        infra = {
            "transmission_line_km": 15.0,
            "distribution_line_km": 30.0,
            "substations_count": 3,
            "avg_line_age_years": 25.0,
            "vegetation_density": 0.6,
        }
        features = self.builder.infrastructure_features(infra)

        assert features["transmission_line_km"] == 15.0
        assert features["line_density_per_km2"] > 0
        assert 0 <= features["infrastructure_age_risk"] <= 1.0

    def test_none_returns_zeros(self):
        features = self.builder.infrastructure_features(None)

        assert features["transmission_line_km"] == 0.0
        assert features["infrastructure_age_risk"] == 0.0

    def test_age_risk_normalized(self):
        young = {"avg_line_age_years": 5, "transmission_line_km": 0,
                 "distribution_line_km": 0, "substations_count": 0, "vegetation_density": 0}
        old = {"avg_line_age_years": 50, "transmission_line_km": 0,
               "distribution_line_km": 0, "substations_count": 0, "vegetation_density": 0}

        young_features = self.builder.infrastructure_features(young)
        old_features = self.builder.infrastructure_features(old)

        assert young_features["infrastructure_age_risk"] < old_features["infrastructure_age_risk"]
        assert old_features["infrastructure_age_risk"] == pytest.approx(1.0, abs=0.01)

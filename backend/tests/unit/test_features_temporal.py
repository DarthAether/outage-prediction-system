"""Tests for temporal feature engineering."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.temporal import TemporalFeatureBuilder


class TestCyclicalEncoding:
    def test_values_bounded(self):
        ts = datetime(2023, 6, 15, 14, 30)
        features = TemporalFeatureBuilder.cyclical_encoding(ts)

        for key in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert -1.0 <= features[key] <= 1.0, f"{key} out of bounds: {features[key]}"

    def test_midnight_encoding(self):
        ts = datetime(2023, 1, 1, 0, 0)
        features = TemporalFeatureBuilder.cyclical_encoding(ts)
        assert abs(features["hour_sin"]) < 0.01  # sin(0) ≈ 0
        assert abs(features["hour_cos"] - 1.0) < 0.01  # cos(0) ≈ 1

    def test_weekend_detection(self):
        saturday = datetime(2023, 6, 10, 12, 0)
        monday = datetime(2023, 6, 12, 12, 0)

        assert TemporalFeatureBuilder.cyclical_encoding(saturday)["is_weekend"] == 1.0
        assert TemporalFeatureBuilder.cyclical_encoding(monday)["is_weekend"] == 0.0

    def test_all_keys_present(self):
        ts = datetime(2023, 6, 15, 12, 0)
        features = TemporalFeatureBuilder.cyclical_encoding(ts)
        expected = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"}
        assert set(features.keys()) == expected


class TestRollingWeatherStats:
    def setup_method(self):
        self.builder = TemporalFeatureBuilder()

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["event_time", "event_type", "magnitude"])
        ts = datetime(2023, 6, 15, 12, 0)
        features = self.builder.rolling_weather_stats(empty, ts)

        assert features["weather_count_24h"] == 0.0
        assert features["weather_max_mag_24h"] == 0.0

    def test_counts_within_window(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [
                ts - timedelta(hours=2),
                ts - timedelta(hours=5),
                ts - timedelta(hours=30),
            ],
            "event_type": ["Wind", "Wind", "Wind"],
            "magnitude": [50, 30, 80],
        })
        features = self.builder.rolling_weather_stats(df, ts, windows=[6, 24])

        assert features["weather_count_6h"] == 2.0
        assert features["weather_count_24h"] == 2.0


class TestLagOutageFeatures:
    def setup_method(self):
        self.builder = TemporalFeatureBuilder()

    def test_lag_retrieval(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "observed_at": [ts - timedelta(hours=1), ts - timedelta(hours=24)],
            "outage_fraction": [0.05, 0.02],
        })
        features = self.builder.lag_outage_features(df, ts)

        assert features["lag_outage_1h"] == pytest.approx(0.05, abs=0.01)
        assert features["lag_outage_24h"] == pytest.approx(0.02, abs=0.01)

    def test_missing_lags_default_zero(self):
        ts = datetime(2023, 6, 15, 12, 0)
        empty = pd.DataFrame(columns=["observed_at", "outage_fraction"])
        features = self.builder.lag_outage_features(empty, ts)

        for lag in [1, 3, 6, 12, 24, 48, 168]:
            assert features[f"lag_outage_{lag}h"] == 0.0


class TestTrendFeatures:
    def setup_method(self):
        self.builder = TemporalFeatureBuilder()

    def test_rising_trend(self):
        ts = datetime(2023, 6, 15, 12, 0)
        times = [ts - timedelta(hours=h) for h in range(6, 0, -1)]
        fracs = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
        df = pd.DataFrame({"observed_at": times, "outage_fraction": fracs})
        features = self.builder.trend_features(df, ts)

        assert features["trend_outage_6h"] > 0

    def test_empty_data_zero_trend(self):
        ts = datetime(2023, 6, 15, 12, 0)
        empty = pd.DataFrame(columns=["observed_at", "outage_fraction"])
        features = self.builder.trend_features(empty, ts)

        assert features["trend_outage_3h"] == 0.0


class TestGridLoadFeatures:
    def setup_method(self):
        self.builder = TemporalFeatureBuilder()

    def test_load_ratio_computed(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "recorded_at": [ts],
            "load_mw": [60000.0],
            "capacity_mw": [80000.0],
            "reserve_margin_pct": [25.0],
        })
        features = self.builder.grid_load_features(df, ts)

        assert features["current_load_mw"] == 60000.0
        assert features["load_capacity_ratio"] == pytest.approx(0.75, abs=0.01)

    def test_no_data_returns_zeros(self):
        ts = datetime(2023, 6, 15, 12, 0)
        empty = pd.DataFrame(columns=["recorded_at", "load_mw", "capacity_mw", "reserve_margin_pct"])
        features = self.builder.grid_load_features(empty, ts)

        assert features["current_load_mw"] == 0.0

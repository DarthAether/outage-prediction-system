"""Tests for compound weather event feature engineering."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.compound_events import (
    CompoundEventFeatureBuilder,
    classify_event,
    CATEGORY_PAIRS,
    EVENT_CATEGORIES,
)


class TestClassifyEvent:
    def test_wind_events(self):
        assert classify_event("Thunderstorm Wind") == "wind"
        assert classify_event("Hurricane") == "wind"
        assert classify_event("Tornado") == "wind"

    def test_ice_events(self):
        assert classify_event("Ice Storm") == "ice"
        assert classify_event("Heavy Snow") == "ice"
        assert classify_event("Winter Storm") == "ice"

    def test_heat_events(self):
        assert classify_event("Heat") == "heat"
        assert classify_event("Excessive Heat") == "heat"

    def test_unknown_event(self):
        assert classify_event("Unknown Event Type") is None
        assert classify_event("") is None


class TestCoOccurrenceFeatures:
    def setup_method(self):
        self.builder = CompoundEventFeatureBuilder()

    def test_no_events_returns_zeros(self):
        empty_df = pd.DataFrame(columns=["event_time", "event_type", "magnitude"])
        ts = datetime(2023, 6, 15, 12, 0)
        features = self.builder.co_occurrence_features(empty_df, ts)

        assert features["compound_event_count"] == 0.0
        assert features["has_compound_event"] == 0.0
        for a, b in CATEGORY_PAIRS:
            assert features[f"cooccur_{a}_{b}"] == 0.0

    def test_single_category_no_compound(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [ts - timedelta(hours=1), ts - timedelta(hours=2)],
            "event_type": ["Thunderstorm Wind", "High Wind"],
            "magnitude": [50, 60],
        })
        features = self.builder.co_occurrence_features(df, ts)

        assert features["compound_event_count"] == 1.0
        assert features["has_compound_event"] == 0.0

    def test_two_categories_detected(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [ts - timedelta(hours=1), ts - timedelta(hours=2)],
            "event_type": ["Thunderstorm Wind", "Ice Storm"],
            "magnitude": [50, 30],
        })
        features = self.builder.co_occurrence_features(df, ts)

        assert features["compound_event_count"] == 2.0
        assert features["has_compound_event"] == 1.0
        assert features["cooccur_ice_wind"] == 1.0

    def test_events_outside_window_excluded(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [ts - timedelta(hours=1), ts - timedelta(hours=30)],
            "event_type": ["Thunderstorm Wind", "Ice Storm"],
            "magnitude": [50, 30],
        })
        features = self.builder.co_occurrence_features(df, ts, window_hours=24)

        assert features["compound_event_count"] == 1.0
        assert features["has_compound_event"] == 0.0


class TestInteractionTerms:
    def setup_method(self):
        self.builder = CompoundEventFeatureBuilder()

    def test_no_interaction_single_category(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [ts - timedelta(hours=1)],
            "event_type": ["Thunderstorm Wind"],
            "magnitude": [50],
            "damage_property": [10000],
        })
        features = self.builder.interaction_terms(df, ts)

        assert features["interact_ice_mag_x_wind_mag"] == 0.0
        assert features["cat_wind_max_magnitude"] == 50.0
        assert features["cat_ice_max_magnitude"] == 0.0

    def test_interaction_product_computed(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [ts - timedelta(hours=1), ts - timedelta(hours=2)],
            "event_type": ["Thunderstorm Wind", "Ice Storm"],
            "magnitude": [50, 30],
            "damage_property": [10000, 5000],
        })
        features = self.builder.interaction_terms(df, ts)

        assert features["interact_ice_mag_x_wind_mag"] == 50.0 * 30.0


class TestSequentialEscalation:
    def setup_method(self):
        self.builder = CompoundEventFeatureBuilder()

    def test_no_events_returns_zeros(self):
        empty_df = pd.DataFrame(columns=["event_time", "event_type", "magnitude"])
        ts = datetime(2023, 6, 15, 12, 0)
        features = self.builder.sequential_escalation(empty_df, ts)

        assert features["seq_storm_count_72h"] == 0.0
        assert features["seq_escalation_score"] == 0.0
        assert features["seq_infrastructure_fatigue"] == 0.0

    def test_escalating_severity_detected(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [
                ts - timedelta(hours=10),
                ts - timedelta(hours=5),
                ts - timedelta(hours=1),
            ],
            "event_type": ["Thunderstorm Wind"] * 3,
            "magnitude": [20, 50, 80],
        })
        features = self.builder.sequential_escalation(df, ts)

        assert features["seq_storm_count_72h"] == 3.0
        assert features["seq_escalation_score"] == 1.0  # all increasing

    def test_fatigue_increases_with_recency(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [ts - timedelta(hours=h) for h in [60, 40, 20, 5, 1]],
            "event_type": ["Thunderstorm Wind"] * 5,
            "magnitude": [50, 50, 50, 50, 50],
        })
        features = self.builder.sequential_escalation(df, ts)

        assert features["seq_infrastructure_fatigue"] > 0
        assert features["seq_storm_count_72h"] == 5.0


class TestCompoundSeverityIndex:
    def setup_method(self):
        self.builder = CompoundEventFeatureBuilder()

    def test_zero_severity_no_events(self):
        co_occ = {"compound_event_count": 0, "has_compound_event": 0}
        interact = {}
        seq = {"seq_infrastructure_fatigue": 0, "seq_escalation_score": 0, "seq_storm_count_72h": 0}

        idx = self.builder.compound_severity_index(co_occ, interact, seq)
        assert idx == 0.0

    def test_severity_bounded_zero_one(self):
        co_occ = {"compound_event_count": 10}
        interact = {"interact_wind_mag_x_ice_mag": 10000}
        seq = {"seq_infrastructure_fatigue": 100, "seq_escalation_score": 1.0, "seq_storm_count_72h": 50}

        idx = self.builder.compound_severity_index(co_occ, interact, seq)
        assert 0.0 <= idx <= 1.0


class TestComputeAll:
    def setup_method(self):
        self.builder = CompoundEventFeatureBuilder()

    def test_compute_all_returns_all_feature_types(self):
        ts = datetime(2023, 6, 15, 12, 0)
        df = pd.DataFrame({
            "event_time": [ts - timedelta(hours=1), ts - timedelta(hours=2)],
            "event_type": ["Thunderstorm Wind", "Ice Storm"],
            "magnitude": [50, 30],
            "damage_property": [10000, 5000],
        })
        features = self.builder.compute_all(df, ts)

        assert "cooccur_ice_wind" in features
        assert "interact_ice_mag_x_wind_mag" in features
        assert "seq_storm_count_72h" in features
        assert "compound_severity_index" in features
        assert isinstance(features["compound_severity_index"], float)

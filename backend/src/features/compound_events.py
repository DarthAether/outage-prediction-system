"""Compound Weather Event Feature Engineering.

This module implements the novel compound event interaction modeling that forms
the primary research contribution. Existing outage prediction literature treats
weather events independently; this module captures nonlinear interactions between
co-occurring or sequential weather phenomena.

Three types of compound features are computed:
1. Co-occurrence features: Which event categories are simultaneously active
2. Interaction terms: Product features capturing severity of co-occurring events
3. Sequential escalation: How repeated or escalating events compound risk
"""

from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

EVENT_CATEGORIES: dict[str, list[str]] = {
    "wind": [
        "Thunderstorm Wind",
        "High Wind",
        "Hurricane",
        "Hurricane (Typhoon)",
        "Tornado",
        "Strong Wind",
        "Tropical Storm",
    ],
    "ice": [
        "Ice Storm",
        "Winter Storm",
        "Heavy Snow",
        "Winter Weather",
        "Blizzard",
        "Freezing Rain",
        "Sleet",
    ],
    "heat": ["Heat", "Excessive Heat"],
    "flood": ["Flash Flood", "Flood", "Coastal Flood"],
    "drought": ["Drought"],
    "fire": ["Wildfire"],
}

CATEGORY_PAIRS = list(combinations(sorted(EVENT_CATEGORIES.keys()), 2))


def classify_event(event_type: str) -> str | None:
    """Map a NOAA event type string to its compound event category."""
    for category, types in EVENT_CATEGORIES.items():
        if event_type in types:
            return category
    return None


class CompoundEventFeatureBuilder:
    """Builds compound weather event interaction features.

    Designed to be called per H3 cell and timestamp, using weather event
    records from the database or a pre-loaded DataFrame.
    """

    def __init__(self, h3_resolution: int = 7):
        self.h3_resolution = h3_resolution

    def co_occurrence_features(
        self,
        events_df: pd.DataFrame,
        timestamp: datetime,
        window_hours: int = 24,
    ) -> dict[str, float]:
        """Compute binary and count co-occurrence features for event category pairs.

        For each pair of event categories, checks if both occurred within
        the specified time window. Also counts the total number of distinct
        active categories.

        Args:
            events_df: DataFrame with columns [event_time, event_type, magnitude]
                       filtered to the target H3 cell and its neighborhood.
            timestamp: The reference time for the prediction.
            window_hours: Look-back window in hours.

        Returns:
            Dict with keys like 'cooccur_fire_wind', 'compound_event_count', etc.
        """
        window_start = timestamp - timedelta(hours=window_hours)
        mask = (events_df["event_time"] >= window_start) & (events_df["event_time"] <= timestamp)
        window_events = events_df.loc[mask]

        if window_events.empty:
            features = {f"cooccur_{a}_{b}": 0.0 for a, b in CATEGORY_PAIRS}
            features["compound_event_count"] = 0.0
            features["has_compound_event"] = 0.0
            return features

        active_categories = set()
        for event_type in window_events["event_type"].unique():
            cat = classify_event(event_type)
            if cat:
                active_categories.add(cat)

        features = {}
        for cat_a, cat_b in CATEGORY_PAIRS:
            features[f"cooccur_{cat_a}_{cat_b}"] = float(
                cat_a in active_categories and cat_b in active_categories
            )

        features["compound_event_count"] = float(len(active_categories))
        features["has_compound_event"] = float(len(active_categories) >= 2)

        return features

    def interaction_terms(
        self,
        events_df: pd.DataFrame,
        timestamp: datetime,
        window_hours: int = 24,
    ) -> dict[str, float]:
        """Compute product interaction features between co-occurring event categories.

        For each pair of active categories, computes the product of their
        maximum severity measures within the window. This captures the intuition
        that high-wind + heavy-ice is disproportionately worse than either alone.

        Args:
            events_df: DataFrame with [event_time, event_type, magnitude, damage_property].
            timestamp: Reference time.
            window_hours: Look-back window.

        Returns:
            Dict with keys like 'interact_wind_mag_x_ice_mag', etc.
        """
        window_start = timestamp - timedelta(hours=window_hours)
        mask = (events_df["event_time"] >= window_start) & (events_df["event_time"] <= timestamp)
        window_events = events_df.loc[mask]

        category_stats: dict[str, dict[str, float]] = {}
        for _, row in window_events.iterrows():
            cat = classify_event(row["event_type"])
            if cat is None:
                continue
            if cat not in category_stats:
                category_stats[cat] = {"max_magnitude": 0.0, "total_damage": 0.0, "count": 0}

            mag = float(row.get("magnitude", 0) or 0)
            dmg = float(row.get("damage_property", 0) or 0)
            category_stats[cat]["max_magnitude"] = max(category_stats[cat]["max_magnitude"], mag)
            category_stats[cat]["total_damage"] += dmg
            category_stats[cat]["count"] += 1

        features = {}
        for cat_a, cat_b in CATEGORY_PAIRS:
            a_mag = category_stats.get(cat_a, {}).get("max_magnitude", 0.0)
            b_mag = category_stats.get(cat_b, {}).get("max_magnitude", 0.0)
            a_dmg = category_stats.get(cat_a, {}).get("total_damage", 0.0)
            b_dmg = category_stats.get(cat_b, {}).get("total_damage", 0.0)

            features[f"interact_{cat_a}_mag_x_{cat_b}_mag"] = a_mag * b_mag
            features[f"interact_{cat_a}_dmg_x_{cat_b}_dmg"] = np.log1p(a_dmg) * np.log1p(b_dmg)

        for cat, stats in category_stats.items():
            features[f"cat_{cat}_max_magnitude"] = stats["max_magnitude"]
            features[f"cat_{cat}_total_damage"] = stats["total_damage"]
            features[f"cat_{cat}_event_count"] = float(stats["count"])

        all_cats = sorted(EVENT_CATEGORIES.keys())
        for cat in all_cats:
            if cat not in category_stats:
                features[f"cat_{cat}_max_magnitude"] = 0.0
                features[f"cat_{cat}_total_damage"] = 0.0
                features[f"cat_{cat}_event_count"] = 0.0

        return features

    def sequential_escalation(
        self,
        events_df: pd.DataFrame,
        timestamp: datetime,
        lookback_hours: int = 72,
    ) -> dict[str, float]:
        """Detect escalating or repeated weather event sequences.

        Captures infrastructure fatigue from repeated storms and escalating
        severity trends that compound outage risk beyond what individual
        event features can express.

        Args:
            events_df: DataFrame with [event_time, event_type, magnitude].
            timestamp: Reference time.
            lookback_hours: Extended look-back window (default 72h = 3 days).

        Returns:
            Dict with sequential escalation features.
        """
        window_start = timestamp - timedelta(hours=lookback_hours)
        mask = (events_df["event_time"] >= window_start) & (events_df["event_time"] <= timestamp)
        window_events = events_df.loc[mask].sort_values("event_time")

        if window_events.empty:
            return {
                "seq_storm_count_72h": 0.0,
                "seq_escalation_score": 0.0,
                "seq_infrastructure_fatigue": 0.0,
                "seq_max_gap_hours": 0.0,
                "seq_distinct_types": 0.0,
                "seq_damage_acceleration": 0.0,
            }

        storm_count = len(window_events)
        distinct_types = window_events["event_type"].nunique()

        magnitudes = pd.to_numeric(window_events["magnitude"], errors="coerce").fillna(0).values
        if len(magnitudes) >= 2:
            diffs = np.diff(magnitudes)
            escalation_score = float(np.mean(diffs > 0))
        else:
            escalation_score = 0.0

        time_diffs = window_events["event_time"].diff().dt.total_seconds() / 3600.0
        max_gap = float(time_diffs.max()) if len(time_diffs) > 1 else float(lookback_hours)

        n_sub = len(window_events)
        if n_sub <= 1:
            fatigue = 0.0
        else:
            recency_weights = np.linspace(0.5, 1.0, n_sub)
            mags = pd.to_numeric(window_events["magnitude"], errors="coerce").fillna(0).values
            fatigue = float(np.dot(mags, recency_weights) / recency_weights.sum())

        damages = (
            pd.to_numeric(
                window_events.get("damage_property", pd.Series(0, index=window_events.index)),
                errors="coerce",
            )
            .fillna(0)
            .values
        )

        if len(damages) >= 3:
            half = len(damages) // 2
            first_half_avg = np.mean(damages[:half]) if half > 0 else 0
            second_half_avg = np.mean(damages[half:])
            damage_accel = float(np.log1p(second_half_avg) - np.log1p(first_half_avg))
        else:
            damage_accel = 0.0

        return {
            "seq_storm_count_72h": float(storm_count),
            "seq_escalation_score": escalation_score,
            "seq_infrastructure_fatigue": fatigue,
            "seq_max_gap_hours": max_gap,
            "seq_distinct_types": float(distinct_types),
            "seq_damage_acceleration": damage_accel,
        }

    def compound_severity_index(
        self,
        co_occurrence: dict[str, float],
        interactions: dict[str, float],
        sequential: dict[str, float],
    ) -> float:
        """Compute a single composite compound severity score (0 to 1).

        Combines co-occurrence, interaction, and sequential features into
        a unified severity metric using calibrated weights.

        This score is designed to be interpretable: 0 means no compound
        risk, 1 means maximum observed compound severity.
        """
        co_score = co_occurrence.get("compound_event_count", 0) / len(EVENT_CATEGORIES)

        interaction_magnitudes = [
            v for k, v in interactions.items() if k.startswith("interact_") and "mag_x_" in k
        ]
        if interaction_magnitudes:
            interact_score = min(1.0, np.mean(interaction_magnitudes) / 100.0)
        else:
            interact_score = 0.0

        fatigue = sequential.get("seq_infrastructure_fatigue", 0)
        escalation = sequential.get("seq_escalation_score", 0)
        storm_count = sequential.get("seq_storm_count_72h", 0)
        seq_score = min(1.0, (fatigue / 50.0 + escalation + storm_count / 20.0) / 3.0)

        w_co, w_interact, w_seq = 0.3, 0.4, 0.3
        composite = w_co * co_score + w_interact * interact_score + w_seq * seq_score
        return float(np.clip(composite, 0.0, 1.0))

    def compute_all(
        self,
        events_df: pd.DataFrame,
        timestamp: datetime,
        window_hours: int = 24,
        lookback_hours: int = 72,
    ) -> dict[str, float]:
        """Compute all compound event features for a single prediction point.

        This is the main entry point. Returns a flat dict of all features
        that can be stored in the feature store.
        """
        co_occ = self.co_occurrence_features(events_df, timestamp, window_hours)
        interact = self.interaction_terms(events_df, timestamp, window_hours)
        seq = self.sequential_escalation(events_df, timestamp, lookback_hours)
        severity_idx = self.compound_severity_index(co_occ, interact, seq)

        features = {}
        features.update(co_occ)
        features.update(interact)
        features.update(seq)
        features["compound_severity_index"] = severity_idx
        return features

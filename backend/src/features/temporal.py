"""Temporal Feature Engineering.

Builds time-series features including rolling statistics, lag features,
cyclical time encodings, trend metrics, and grid load indicators.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class TemporalFeatureBuilder:
    """Computes temporal features for outage prediction at a given H3 cell and timestamp."""

    WINDOWS_HOURS = [1, 3, 6, 12, 24, 48, 72]
    LAG_HOURS = [1, 3, 6, 12, 24, 48, 168]

    def rolling_weather_stats(
        self,
        weather_df: pd.DataFrame,
        timestamp: datetime,
        windows: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute rolling statistics on weather events over multiple time windows.

        For each window, counts events and computes max/mean magnitude.

        Args:
            weather_df: Weather events for the target cell and its neighbors.
                       Columns: [event_time, event_type, magnitude].
            timestamp: Prediction reference time.
            windows: List of look-back window sizes in hours.

        Returns:
            Dict of features like weather_count_6h, weather_max_mag_24h, etc.
        """
        if windows is None:
            windows = self.WINDOWS_HOURS

        features: dict[str, float] = {}
        for w in windows:
            start = timestamp - timedelta(hours=w)
            mask = (weather_df["event_time"] >= start) & (weather_df["event_time"] <= timestamp)
            subset = weather_df.loc[mask]

            count = len(subset)
            if count > 0:
                mags = pd.to_numeric(subset["magnitude"], errors="coerce").fillna(0)
                max_mag = float(mags.max())
                mean_mag = float(mags.mean())
                distinct_types = subset["event_type"].nunique()
            else:
                max_mag = 0.0
                mean_mag = 0.0
                distinct_types = 0

            features[f"weather_count_{w}h"] = float(count)
            features[f"weather_max_mag_{w}h"] = max_mag
            features[f"weather_mean_mag_{w}h"] = mean_mag
            features[f"weather_distinct_types_{w}h"] = float(distinct_types)

        return features

    def lag_outage_features(
        self,
        outage_df: pd.DataFrame,
        timestamp: datetime,
        lags: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute lagged outage fraction values.

        Captures the autoregressive nature of outages -- if an area
        had outages recently, it is more likely to have them again.

        Args:
            outage_df: Outage observations with [observed_at, outage_fraction].
            timestamp: Reference time.
            lags: Lag offsets in hours.

        Returns:
            Dict with lag_outage_1h, lag_outage_24h, etc.
        """
        if lags is None:
            lags = self.LAG_HOURS

        features: dict[str, float] = {}
        for lag in lags:
            target_time = timestamp - timedelta(hours=lag)
            tolerance = timedelta(minutes=30)
            mask = (outage_df["observed_at"] >= target_time - tolerance) & (
                outage_df["observed_at"] <= target_time + tolerance
            )
            matches = outage_df.loc[mask]

            if not matches.empty:
                features[f"lag_outage_{lag}h"] = float(matches["outage_fraction"].mean())
            else:
                features[f"lag_outage_{lag}h"] = 0.0

        return features

    @staticmethod
    def cyclical_encoding(timestamp: datetime) -> dict[str, float]:
        """Encode time components as sin/cos pairs.

        Cyclical encoding preserves the circular nature of time:
        hour 23 is close to hour 0, December is close to January.

        Returns:
            Dict with hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos.
        """
        hour = timestamp.hour + timestamp.minute / 60.0
        dow = timestamp.weekday()
        month = timestamp.month - 1  # 0-indexed

        return {
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "month_sin": float(np.sin(2 * np.pi * month / 12)),
            "month_cos": float(np.cos(2 * np.pi * month / 12)),
            "is_weekend": float(dow >= 5),
        }

    def trend_features(
        self,
        outage_df: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, float]:
        """Compute rate of change in outage fraction over recent windows.

        Rising trends indicate deteriorating conditions; falling trends
        indicate recovery. This captures dynamics that lag features miss.

        Args:
            outage_df: Outage observations with [observed_at, outage_fraction].
            timestamp: Reference time.

        Returns:
            Dict with trend_3h, trend_6h, trend_12h.
        """
        features: dict[str, float] = {}

        for window in [3, 6, 12]:
            start = timestamp - timedelta(hours=window)
            mid = timestamp - timedelta(hours=window / 2)

            first_half = outage_df[
                (outage_df["observed_at"] >= start) & (outage_df["observed_at"] < mid)
            ]
            second_half = outage_df[
                (outage_df["observed_at"] >= mid) & (outage_df["observed_at"] <= timestamp)
            ]

            avg_first = first_half["outage_fraction"].mean() if not first_half.empty else 0.0
            avg_second = second_half["outage_fraction"].mean() if not second_half.empty else 0.0

            features[f"trend_outage_{window}h"] = float(avg_second - avg_first)

        return features

    def grid_load_features(
        self,
        load_df: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, float]:
        """Compute grid load features relative to capacity and seasonal norms.

        High load relative to capacity (low reserve margin) increases
        outage risk, especially during extreme weather.

        Args:
            load_df: Grid load data with [recorded_at, load_mw, capacity_mw, reserve_margin_pct].
            timestamp: Reference time.

        Returns:
            Dict with current_load_mw, reserve_margin, load_ratio, etc.
        """
        tolerance = timedelta(hours=1)
        mask = (load_df["recorded_at"] >= timestamp - tolerance) & (
            load_df["recorded_at"] <= timestamp + tolerance
        )
        recent = load_df.loc[mask]

        if recent.empty:
            return {
                "current_load_mw": 0.0,
                "reserve_margin_pct": 0.0,
                "load_capacity_ratio": 0.0,
            }

        latest = recent.iloc[-1]
        load_mw = float(latest.get("load_mw", 0) or 0)
        capacity = float(latest.get("capacity_mw", 0) or 0)
        reserve = float(latest.get("reserve_margin_pct", 0) or 0)
        ratio = load_mw / capacity if capacity > 0 else 0.0

        return {
            "current_load_mw": load_mw,
            "reserve_margin_pct": reserve,
            "load_capacity_ratio": ratio,
        }

    def compute_all(
        self,
        weather_df: pd.DataFrame,
        outage_df: pd.DataFrame,
        load_df: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, float]:
        """Compute all temporal features for a single prediction point."""
        features: dict[str, float] = {}
        features.update(self.rolling_weather_stats(weather_df, timestamp))
        features.update(self.lag_outage_features(outage_df, timestamp))
        features.update(self.cyclical_encoding(timestamp))
        features.update(self.trend_features(outage_df, timestamp))
        features.update(self.grid_load_features(load_df, timestamp))
        return features

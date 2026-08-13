"""Spatial Feature Engineering with H3 Hexagonal Indexing.

Uses Uber's H3 discrete global grid system for resolution-adaptive
spatial aggregation. H3 resolution 7 (~5.16 km^2 per cell) is the
default, matching typical distribution feeder service areas.
"""

from datetime import datetime, timedelta

import h3
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def lat_lon_to_h3(lat: float, lon: float, resolution: int = 7) -> str:
    """Convert latitude/longitude to H3 cell index."""
    return h3.latlng_to_cell(lat, lon, resolution)


def get_h3_neighbors(h3_index: str, k: int = 1) -> list[str]:
    """Get k-ring neighbors of an H3 cell (including the cell itself)."""
    return list(h3.grid_disk(h3_index, k))


def h3_to_lat_lon(h3_index: str) -> tuple[float, float]:
    """Get the centroid lat/lon of an H3 cell."""
    lat, lon = h3.cell_to_latlng(h3_index)
    return lat, lon


def h3_cell_area_km2(resolution: int = 7) -> float:
    """Approximate area of an H3 cell at a given resolution in km^2."""
    areas = {
        0: 4_357_449.416,
        1: 609_788.441,
        2: 86_801.780,
        3: 12_393.434,
        4: 1_770.347,
        5: 252.903,
        6: 36.129,
        7: 5.161,
        8: 0.737,
        9: 0.105,
        10: 0.015,
        11: 0.002,
    }
    return areas.get(resolution, 5.161)


class SpatialFeatureBuilder:
    """Builds spatial features using H3-based neighborhood aggregation."""

    def __init__(self, h3_resolution: int = 7):
        self.resolution = h3_resolution

    def neighborhood_aggregation(
        self,
        h3_cell: str,
        weather_df: pd.DataFrame,
        outage_df: pd.DataFrame,
        timestamp: datetime,
        k: int = 2,
        window_hours: int = 24,
    ) -> dict[str, float]:
        """Aggregate weather and outage statistics over k-ring neighbors.

        Captures spatial contagion: outages in neighboring cells indicate
        that a regional weather event is affecting the area, increasing
        local risk.

        Args:
            h3_cell: Target H3 cell index.
            weather_df: Weather events with [event_time, h3_index_res7, magnitude].
            outage_df: Outage observations with [observed_at, h3_index_res7, outage_fraction].
            timestamp: Reference time.
            k: Number of rings for neighborhood.
            window_hours: Look-back window.

        Returns:
            Dict with neighbor_outage_mean, neighbor_weather_count, etc.
        """
        neighbors = get_h3_neighbors(h3_cell, k)
        neighbors_excluding_self = [n for n in neighbors if n != h3_cell]
        start = timestamp - timedelta(hours=window_hours)

        # Weather in neighborhood
        if not weather_df.empty and "h3_index_res7" in weather_df.columns:
            w_mask = (
                weather_df["h3_index_res7"].isin(neighbors_excluding_self)
                & (weather_df["event_time"] >= start)
                & (weather_df["event_time"] <= timestamp)
            )
            neighbor_weather = weather_df.loc[w_mask]
            neighbor_weather_count = len(neighbor_weather)
            if neighbor_weather_count > 0:
                mags = pd.to_numeric(neighbor_weather["magnitude"], errors="coerce").fillna(0)
                neighbor_weather_max = float(mags.max())
                neighbor_weather_mean = float(mags.mean())
            else:
                neighbor_weather_max = 0.0
                neighbor_weather_mean = 0.0
        else:
            neighbor_weather_count = 0
            neighbor_weather_max = 0.0
            neighbor_weather_mean = 0.0

        # Outages in neighborhood
        if not outage_df.empty and "h3_index_res7" in outage_df.columns:
            o_mask = (
                outage_df["h3_index_res7"].isin(neighbors_excluding_self)
                & (outage_df["observed_at"] >= start)
                & (outage_df["observed_at"] <= timestamp)
            )
            neighbor_outages = outage_df.loc[o_mask]
            if not neighbor_outages.empty:
                neighbor_outage_mean = float(neighbor_outages["outage_fraction"].mean())
                neighbor_outage_max = float(neighbor_outages["outage_fraction"].max())
                cells_with_outage = neighbor_outages[neighbor_outages["outage_fraction"] > 0.01][
                    "h3_index_res7"
                ].nunique()
                outage_spread = cells_with_outage / max(len(neighbors_excluding_self), 1)
            else:
                neighbor_outage_mean = 0.0
                neighbor_outage_max = 0.0
                outage_spread = 0.0
        else:
            neighbor_outage_mean = 0.0
            neighbor_outage_max = 0.0
            outage_spread = 0.0

        return {
            "neighbor_weather_count": float(neighbor_weather_count),
            "neighbor_weather_max_mag": neighbor_weather_max,
            "neighbor_weather_mean_mag": neighbor_weather_mean,
            "neighbor_outage_mean": neighbor_outage_mean,
            "neighbor_outage_max": neighbor_outage_max,
            "neighbor_outage_spread": float(outage_spread),
            "neighbor_ring_k": float(k),
        }

    def infrastructure_features(
        self,
        infrastructure_data: dict | None,
    ) -> dict[str, float]:
        """Extract infrastructure density features for an H3 cell.

        Args:
            infrastructure_data: Dict or row from infrastructure table with
                transmission_line_km, distribution_line_km, substations_count,
                avg_line_age_years, vegetation_density.

        Returns:
            Dict with infrastructure features.
        """
        if infrastructure_data is None:
            return {
                "transmission_line_km": 0.0,
                "distribution_line_km": 0.0,
                "substations_count": 0.0,
                "avg_line_age_years": 0.0,
                "vegetation_density": 0.0,
                "line_density_per_km2": 0.0,
                "infrastructure_age_risk": 0.0,
            }

        trans_km = float(infrastructure_data.get("transmission_line_km", 0) or 0)
        dist_km = float(infrastructure_data.get("distribution_line_km", 0) or 0)
        substations = float(infrastructure_data.get("substations_count", 0) or 0)
        line_age = float(infrastructure_data.get("avg_line_age_years", 0) or 0)
        veg_density = float(infrastructure_data.get("vegetation_density", 0) or 0)

        cell_area = h3_cell_area_km2(self.resolution)
        total_line_km = trans_km + dist_km
        line_density = total_line_km / cell_area if cell_area > 0 else 0.0

        age_risk = min(1.0, line_age / 50.0)

        return {
            "transmission_line_km": trans_km,
            "distribution_line_km": dist_km,
            "substations_count": substations,
            "avg_line_age_years": line_age,
            "vegetation_density": veg_density,
            "line_density_per_km2": line_density,
            "infrastructure_age_risk": age_risk,
        }

    def compute_all(
        self,
        h3_cell: str,
        weather_df: pd.DataFrame,
        outage_df: pd.DataFrame,
        timestamp: datetime,
        infrastructure_data: dict | None = None,
        k: int = 2,
    ) -> dict[str, float]:
        """Compute all spatial features for a single prediction point."""
        features: dict[str, float] = {}
        features.update(self.neighborhood_aggregation(h3_cell, weather_df, outage_df, timestamp, k))
        features.update(self.infrastructure_features(infrastructure_data))
        return features

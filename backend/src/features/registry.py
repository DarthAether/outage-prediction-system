"""Feature Registry - metadata catalog of all engineered features.

Documents every feature with its name, type, description, builder source,
and version history. Used for model documentation, paper tables, and
ensuring reproducibility.
"""

from dataclasses import dataclass


@dataclass
class FeatureMetadata:
    name: str
    dtype: str
    description: str
    builder: str
    group: str
    version: str = "v1.0"


FEATURE_REGISTRY: list[FeatureMetadata] = [
    # Temporal - Rolling Weather
    FeatureMetadata(
        "weather_count_1h",
        "float",
        "Number of weather events in past 1 hour",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    FeatureMetadata(
        "weather_count_6h",
        "float",
        "Number of weather events in past 6 hours",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    FeatureMetadata(
        "weather_count_24h",
        "float",
        "Number of weather events in past 24 hours",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    FeatureMetadata(
        "weather_count_72h",
        "float",
        "Number of weather events in past 72 hours",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    FeatureMetadata(
        "weather_max_mag_24h",
        "float",
        "Max event magnitude in past 24 hours",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    FeatureMetadata(
        "weather_mean_mag_24h",
        "float",
        "Mean event magnitude in past 24 hours",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    # Temporal - Lag Outage
    FeatureMetadata(
        "lag_outage_1h", "float", "Outage fraction 1 hour ago", "TemporalFeatureBuilder", "temporal"
    ),
    FeatureMetadata(
        "lag_outage_24h",
        "float",
        "Outage fraction 24 hours ago",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    FeatureMetadata(
        "lag_outage_168h",
        "float",
        "Outage fraction 1 week ago",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    # Temporal - Cyclical
    FeatureMetadata(
        "hour_sin", "float", "Sine encoding of hour of day", "TemporalFeatureBuilder", "temporal"
    ),
    FeatureMetadata(
        "hour_cos", "float", "Cosine encoding of hour of day", "TemporalFeatureBuilder", "temporal"
    ),
    FeatureMetadata(
        "month_sin", "float", "Sine encoding of month", "TemporalFeatureBuilder", "temporal"
    ),
    FeatureMetadata(
        "is_weekend", "float", "Binary weekend indicator", "TemporalFeatureBuilder", "temporal"
    ),
    # Temporal - Trends
    FeatureMetadata(
        "trend_outage_3h",
        "float",
        "Outage fraction change over 3 hours",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    FeatureMetadata(
        "trend_outage_12h",
        "float",
        "Outage fraction change over 12 hours",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    # Temporal - Grid Load
    FeatureMetadata(
        "current_load_mw", "float", "Current grid load in MW", "TemporalFeatureBuilder", "temporal"
    ),
    FeatureMetadata(
        "reserve_margin_pct",
        "float",
        "Grid reserve margin percentage",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    FeatureMetadata(
        "load_capacity_ratio",
        "float",
        "Load to capacity ratio",
        "TemporalFeatureBuilder",
        "temporal",
    ),
    # Spatial - Neighborhood
    FeatureMetadata(
        "neighbor_weather_count",
        "float",
        "Weather event count in k-ring neighbors",
        "SpatialFeatureBuilder",
        "spatial",
    ),
    FeatureMetadata(
        "neighbor_outage_mean",
        "float",
        "Mean outage fraction in neighbors",
        "SpatialFeatureBuilder",
        "spatial",
    ),
    FeatureMetadata(
        "neighbor_outage_spread",
        "float",
        "Fraction of neighbors with active outages",
        "SpatialFeatureBuilder",
        "spatial",
    ),
    # Spatial - Infrastructure
    FeatureMetadata(
        "transmission_line_km",
        "float",
        "Transmission line length in cell (km)",
        "SpatialFeatureBuilder",
        "spatial",
    ),
    FeatureMetadata(
        "vegetation_density",
        "float",
        "Vegetation density near power lines",
        "SpatialFeatureBuilder",
        "spatial",
    ),
    FeatureMetadata(
        "infrastructure_age_risk",
        "float",
        "Normalized infrastructure age risk (0-1)",
        "SpatialFeatureBuilder",
        "spatial",
    ),
    # Compound Events (NOVEL)
    FeatureMetadata(
        "cooccur_ice_wind",
        "float",
        "Binary: ice and wind events co-occurring",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    FeatureMetadata(
        "cooccur_heat_drought",
        "float",
        "Binary: heat and drought co-occurring",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    FeatureMetadata(
        "compound_event_count",
        "float",
        "Number of distinct active weather categories",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    FeatureMetadata(
        "has_compound_event",
        "float",
        "Binary: 2+ weather categories active",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    FeatureMetadata(
        "interact_wind_mag_x_ice_mag",
        "float",
        "Product of wind and ice magnitudes",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    FeatureMetadata(
        "seq_storm_count_72h",
        "float",
        "Total storm count in 72-hour lookback",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    FeatureMetadata(
        "seq_escalation_score",
        "float",
        "Severity escalation trend (0-1)",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    FeatureMetadata(
        "seq_infrastructure_fatigue",
        "float",
        "Recency-weighted cumulative storm stress",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    FeatureMetadata(
        "compound_severity_index",
        "float",
        "Composite compound event severity (0-1)",
        "CompoundEventFeatureBuilder",
        "compound",
    ),
    # Socioeconomic
    FeatureMetadata(
        "population_density",
        "float",
        "Population per km^2",
        "SocioeconomicFeatureBuilder",
        "socioeconomic",
    ),
    FeatureMetadata(
        "median_income_normalized",
        "float",
        "Median income / 100k",
        "SocioeconomicFeatureBuilder",
        "socioeconomic",
    ),
    FeatureMetadata(
        "composite_vulnerability_index",
        "float",
        "Community vulnerability score (0-1)",
        "SocioeconomicFeatureBuilder",
        "socioeconomic",
    ),
]


def get_feature_names_by_group(group: str) -> list[str]:
    """Get all feature names belonging to a specific group."""
    return [f.name for f in FEATURE_REGISTRY if f.group == group]


def get_all_feature_names() -> list[str]:
    """Get all registered feature names."""
    return [f.name for f in FEATURE_REGISTRY]


def get_feature_groups() -> dict[str, list[str]]:
    """Get features organized by group for ablation studies."""
    groups: dict[str, list[str]] = {}
    for f in FEATURE_REGISTRY:
        groups.setdefault(f.group, []).append(f.name)
    return groups

"""Shared test fixtures for the outage prediction test suite."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_weather_events():
    """Generate realistic weather event records for testing."""
    n = 200
    rng = np.random.RandomState(42)
    event_types = [
        "Thunderstorm Wind", "High Wind", "Heavy Snow", "Ice Storm",
        "Winter Storm", "Tornado", "Hail", "Flash Flood", "Heat",
        "Excessive Heat", "Drought", "Wildfire",
    ]
    base_time = datetime(2023, 6, 1)

    return pd.DataFrame({
        "event_time": [base_time + timedelta(hours=rng.randint(0, 720)) for _ in range(n)],
        "event_type": rng.choice(event_types, n),
        "magnitude": rng.uniform(0, 120, n),
        "damage_property": rng.uniform(0, 500_000, n),
        "h3_index_res7": [f"872830{rng.randint(0, 99):02d}ffff" for _ in range(n)],
        "source": "noaa_storms",
    })


@pytest.fixture
def sample_outage_observations():
    """Generate realistic outage observation records."""
    n = 500
    rng = np.random.RandomState(42)
    base_time = datetime(2023, 6, 1)

    return pd.DataFrame({
        "observed_at": [base_time + timedelta(hours=i) for i in range(n)],
        "county_fips": "201",
        "state_fips": "48",
        "customers_out": rng.poisson(50, n),
        "total_customers": 10000,
        "outage_fraction": rng.uniform(0, 0.1, n),
        "h3_index_res7": "87283082ffff",
    })


@pytest.fixture
def sample_grid_load():
    """Generate realistic grid load data."""
    n = 200
    rng = np.random.RandomState(42)
    base_time = datetime(2023, 6, 1)

    return pd.DataFrame({
        "recorded_at": [base_time + timedelta(hours=i) for i in range(n)],
        "region_code": "TX",
        "load_mw": 50000 + rng.normal(0, 5000, n),
        "capacity_mw": 80000.0,
        "reserve_margin_pct": 20 + rng.normal(0, 5, n),
        "frequency_hz": 60.0 + rng.normal(0, 0.01, n),
    })


@pytest.fixture
def sample_training_data():
    """Generate a complete training dataset with features and targets."""
    rng = np.random.RandomState(42)
    n = 1000
    n_features = 30

    features = rng.randn(n, n_features).astype(np.float32)
    targets = (rng.rand(n) > 0.7).astype(np.float32)

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=columns)
    df["target_outage"] = targets
    df["timestamp"] = pd.date_range("2023-01-01", periods=n, freq="h")
    df["h3_cell"] = "87283082ffff"
    df["target_max_outage_fraction"] = targets * rng.uniform(0, 0.3, n)

    return df

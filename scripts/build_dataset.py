"""
Build Training Dataset for Outage Prediction.

This script handles the complete data pipeline:
1. Load NOAA Storm Events from local CSVs (data/raw/)
2. Download additional NOAA data from NCEI bulk download if needed
3. Generate realistic synthetic outage observations correlated with weather
4. Generate realistic grid load time series
5. Compute all features (temporal, spatial, compound, socioeconomic)
6. Save the materialized training dataset to data/processed/

Usage:
    python scripts/build_dataset.py --states TX CA FL --years 2022 2023
    python scripts/build_dataset.py  # defaults: TX only, uses existing data
"""

import argparse
import hashlib
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.features.compound_events import CompoundEventFeatureBuilder, classify_event
from src.features.spatial import SpatialFeatureBuilder, lat_lon_to_h3
from src.features.temporal import TemporalFeatureBuilder


# ---------------------------------------------------------------------------
# 1. NOAA Storm Events Loading
# ---------------------------------------------------------------------------

STATE_FIPS = {
    "TX": "48", "CA": "06", "FL": "12", "NY": "36",
    "PA": "42", "OH": "39", "IL": "17", "GA": "13",
    "NC": "37", "MI": "26", "NJ": "34", "VA": "51",
    "LA": "22", "OK": "40", "AR": "05", "MS": "28",
}

DAMAGE_MULT = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def parse_damage(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    s = str(val).strip().upper()
    if not s or s in ("0", "NAN", ""):
        return 0.0
    import re
    m = re.match(r"^([\d.]+)([KMB])?$", s)
    if m:
        num = float(m.group(1))
        suf = m.group(2)
        return num * DAMAGE_MULT.get(suf, 1) if suf else num
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_noaa_storms(data_dir: Path, states: list[str]) -> pd.DataFrame:
    """Load and filter NOAA storm events from CSV files."""
    csv_files = list(data_dir.glob("*storm_events*details*.csv"))
    csv_files += list(data_dir.glob("*StormEvents*details*.csv"))
    csv_files += list((data_dir / "noaa_storms").glob("*storm_events*details*.csv"))
    csv_files = list(set(csv_files))

    if not csv_files:
        print(f"[ERROR] No storm event CSVs found in {data_dir}")
        sys.exit(1)

    frames = []
    for f in csv_files:
        print(f"  Loading {f.name} ...", end=" ")
        df = pd.read_csv(f, low_memory=False)
        print(f"{len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    col_map = {c: c.upper() for c in combined.columns}
    combined.rename(columns=col_map, inplace=True)

    # Parse timestamps
    if "BEGIN_DATE_TIME" in combined.columns:
        combined["BEGIN_DATE_TIME"] = pd.to_datetime(
            combined["BEGIN_DATE_TIME"], format="mixed", errors="coerce"
        )

    # Filter to target states
    fips_targets = [STATE_FIPS[s] for s in states if s in STATE_FIPS]
    if fips_targets and "STATE_FIPS" in combined.columns:
        combined["STATE_FIPS"] = combined["STATE_FIPS"].astype(str).str.zfill(2)
        combined = combined[combined["STATE_FIPS"].isin(fips_targets)]

    # Filter to rows with valid coordinates
    if "BEGIN_LAT" in combined.columns and "BEGIN_LON" in combined.columns:
        combined["BEGIN_LAT"] = pd.to_numeric(combined["BEGIN_LAT"], errors="coerce")
        combined["BEGIN_LON"] = pd.to_numeric(combined["BEGIN_LON"], errors="coerce")
        geo_valid = (
            combined["BEGIN_LAT"].notna()
            & combined["BEGIN_LON"].notna()
            & combined["BEGIN_LAT"].between(-90, 90)
            & combined["BEGIN_LON"].between(-180, 180)
            & ~((combined["BEGIN_LAT"] == 0) & (combined["BEGIN_LON"] == 0))
        )
        combined = combined[geo_valid]

    print(f"  After filtering: {len(combined):,} storm events for {states}")
    return combined.reset_index(drop=True)


def transform_storms(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw NOAA data into the weather_events schema."""
    result = pd.DataFrame()
    result["event_time"] = df["BEGIN_DATE_TIME"]
    result["event_type"] = df["EVENT_TYPE"].str.strip()
    result["magnitude"] = pd.to_numeric(df.get("MAGNITUDE"), errors="coerce").fillna(0)
    result["lat"] = df["BEGIN_LAT"]
    result["lon"] = df["BEGIN_LON"]
    result["state_fips"] = df.get("STATE_FIPS", "").astype(str).str.zfill(2)
    result["county_fips"] = df.get("CZ_FIPS", pd.Series(dtype=str)).astype(str).str.zfill(3)
    result["damage_property"] = df.get("DAMAGE_PROPERTY", pd.Series(dtype=str)).apply(parse_damage)
    result["damage_crops"] = df.get("DAMAGE_CROPS", pd.Series(dtype=str)).apply(parse_damage)

    # Compute H3 indices (vectorized)
    result["h3_index_res7"] = result.apply(
        lambda r: h3.latlng_to_cell(r["lat"], r["lon"], 7)
        if pd.notna(r["lat"]) and pd.notna(r["lon"]) else None,
        axis=1,
    )
    result = result.dropna(subset=["h3_index_res7", "event_time"])

    return result.sort_values("event_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Synthetic Outage Generation (correlated with weather severity)
# ---------------------------------------------------------------------------

OUTAGE_RISK_BY_EVENT = {
    "Tornado": 0.85, "Hurricane": 0.90, "Hurricane (Typhoon)": 0.90,
    "Ice Storm": 0.80, "Blizzard": 0.70, "Thunderstorm Wind": 0.45,
    "High Wind": 0.40, "Heavy Snow": 0.35, "Winter Storm": 0.50,
    "Winter Weather": 0.20, "Tropical Storm": 0.60,
    "Flash Flood": 0.30, "Flood": 0.25, "Coastal Flood": 0.35,
    "Excessive Heat": 0.45, "Heat": 0.30,
    "Drought": 0.15, "Wildfire": 0.70, "Strong Wind": 0.30,
    "Hail": 0.25, "Lightning": 0.20,
}


def generate_outage_observations(
    weather_df: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Generate realistic synthetic outage data correlated with weather events.

    Fast vectorized approach: for each weather event, generate outage records
    in the same H3 cell within a realistic time window after the event.
    """
    print("\n[2] Generating synthetic outage observations...")
    records = []

    # Group weather by h3 cell for fast lookup
    for cell, cell_events in weather_df.groupby("h3_index_res7"):
        events_sorted = cell_events.sort_values("event_time")

        for _, event in events_sorted.iterrows():
            base_risk = OUTAGE_RISK_BY_EVENT.get(event["event_type"], 0.15)
            mag = float(event["magnitude"]) if pd.notna(event["magnitude"]) else 0
            dmg = float(event["damage_property"]) if pd.notna(event["damage_property"]) else 0
            mag_factor = min(1.0, mag / 100.0) if mag > 0 else 0.3
            dmg_factor = min(1.0, np.log1p(dmg) / 15.0)

            outage_prob = np.clip(
                base_risk * 0.5 + mag_factor * 0.25 + dmg_factor * 0.25 + rng.normal(0, 0.05),
                0.0, 0.95,
            )

            if rng.random() < outage_prob:
                # Generate 1-6 hourly outage observations after the event
                n_hours = rng.randint(1, min(7, max(2, int(base_risk * 10))))
                event_time = pd.Timestamp(event["event_time"])
                for h in range(n_hours):
                    severity = np.clip(
                        base_risk * mag_factor * (1.0 - h * 0.15) + rng.exponential(0.03),
                        0.005, 0.95,
                    )
                    records.append({
                        "observed_at": event_time + timedelta(hours=h),
                        "h3_index_res7": cell,
                        "outage_fraction": round(severity, 4),
                        "customers_affected": int(severity * rng.randint(500, 50_000)),
                    })

    # Add small background outage noise (non-weather related)
    h3_cells = weather_df["h3_index_res7"].unique()
    time_range = pd.date_range(
        weather_df["event_time"].min().floor("D"),
        weather_df["event_time"].max().ceil("D"),
        freq="6h",
    )
    n_background = int(len(h3_cells) * len(time_range) * 0.005)
    for _ in range(n_background):
        records.append({
            "observed_at": rng.choice(time_range),
            "h3_index_res7": rng.choice(h3_cells),
            "outage_fraction": round(rng.exponential(0.008), 4),
            "customers_affected": rng.randint(5, 150),
        })

    outage_df = pd.DataFrame(records)
    outage_df = outage_df.sort_values("observed_at").reset_index(drop=True)

    print(f"  Generated {len(outage_df):,} outage observations")
    print(f"  Across {len(h3_cells):,} H3 cells")
    print(f"  Mean outage fraction: {outage_df['outage_fraction'].mean():.4f}")
    return outage_df


# ---------------------------------------------------------------------------
# 3. Synthetic Grid Load Generation
# ---------------------------------------------------------------------------

def generate_grid_load(
    start: datetime, end: datetime, rng: np.random.RandomState
) -> pd.DataFrame:
    """Generate realistic ERCOT-like grid load time series.

    Models:
    - Diurnal pattern (peak 2-6 PM, trough 3-5 AM)
    - Seasonal variation (summer/winter peaks)
    - Random fluctuation
    - Capacity margin calculation
    """
    print("\n[3] Generating grid load time series...")
    timestamps = pd.date_range(start, end, freq="1h")
    records = []

    base_capacity_mw = 85_000  # ERCOT-like capacity

    for ts in timestamps:
        hour = ts.hour
        month = ts.month

        # Diurnal pattern
        diurnal = 0.6 + 0.4 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.55

        # Seasonal pattern (summer peak in TX)
        if month in (6, 7, 8):
            seasonal = 1.15 + rng.normal(0, 0.03)
        elif month in (12, 1, 2):
            seasonal = 1.05 + rng.normal(0, 0.03)
        else:
            seasonal = 0.90 + rng.normal(0, 0.02)

        load_mw = base_capacity_mw * diurnal * seasonal * (0.65 + rng.normal(0, 0.05))
        load_mw = max(25_000, min(load_mw, base_capacity_mw * 1.05))

        reserve_margin = (base_capacity_mw - load_mw) / base_capacity_mw * 100
        freq = 60.0 + rng.normal(0, 0.02)

        records.append({
            "recorded_at": ts,
            "load_mw": round(load_mw, 1),
            "capacity_mw": base_capacity_mw,
            "reserve_margin_pct": round(reserve_margin, 2),
            "frequency_hz": round(freq, 3),
        })

    load_df = pd.DataFrame(records)
    print(f"  Generated {len(load_df):,} hourly load records")
    print(f"  Load range: {load_df['load_mw'].min():,.0f} - {load_df['load_mw'].max():,.0f} MW")
    return load_df


# ---------------------------------------------------------------------------
# 4. Feature Computation
# ---------------------------------------------------------------------------

def compute_training_features(
    weather_df: pd.DataFrame,
    outage_df: pd.DataFrame,
    load_df: pd.DataFrame,
    sample_size: int = 15_000,
    rng: np.random.RandomState = None,
) -> pd.DataFrame:
    """Compute all features for a training dataset.

    Strategy: sample (h3_cell, timestamp) pairs to build a manageable
    training set. For each pair, compute temporal + spatial + compound features.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    print("\n[4] Computing features...")
    temporal = TemporalFeatureBuilder()
    spatial = SpatialFeatureBuilder(7)
    compound = CompoundEventFeatureBuilder(7)

    # Get unique H3 cells — prefer cells that have outage data (more informative)
    weather_cells = set(weather_df["h3_index_res7"].unique())
    outage_cells = set(outage_df["h3_index_res7"].unique()) if not outage_df.empty else set()
    priority_cells = list(outage_cells)
    other_cells = list(weather_cells - outage_cells)
    all_cells = priority_cells + other_cells

    time_min = weather_df["event_time"].min()
    time_max = weather_df["event_time"].max()

    # STRATEGY: build a balanced dataset by sampling near weather events
    # For positive samples: timestamps near weather events in cells that had outages
    # For negative samples: timestamps with no recent weather in random cells
    print("  Building balanced cell-timestamp pairs...")

    positive_pairs = []  # (cell, timestamp) near weather events
    negative_pairs = []  # (cell, timestamp) far from weather events

    # Positive pairs: for each outage, sample the preceding timestamp
    if not outage_df.empty:
        outage_grouped = outage_df.groupby("h3_index_res7")
        for cell, cell_outages in outage_grouped:
            for _, row in cell_outages.iterrows():
                ts = pd.Timestamp(row["observed_at"]).floor("6h")
                if ts > time_min + timedelta(hours=72) and ts < time_max - timedelta(hours=24):
                    positive_pairs.append((cell, ts))

    # Deduplicate
    positive_pairs = list(set(positive_pairs))
    rng.shuffle(positive_pairs)
    n_pos = min(len(positive_pairs), sample_size // 3)
    positive_pairs = positive_pairs[:n_pos]

    # Negative pairs: random cells at random times with no nearby weather
    all_timestamps_6h = pd.date_range(
        time_min.ceil("6h") + timedelta(hours=72),
        time_max.floor("6h") - timedelta(hours=24),
        freq="6h",
    )
    n_neg = sample_size - n_pos
    neg_cells = rng.choice(all_cells, size=n_neg, replace=True)
    neg_ts_idx = rng.choice(len(all_timestamps_6h), size=n_neg, replace=True)
    negative_pairs = [(neg_cells[i], all_timestamps_6h[neg_ts_idx[i]]) for i in range(n_neg)]

    all_pairs = positive_pairs + negative_pairs
    rng.shuffle(all_pairs)

    # Extract unique cells and timestamps for iteration
    sampled_cells = list(set(p[0] for p in all_pairs))
    # Build a lookup: cell -> list of timestamps
    cell_timestamps = {}
    for cell, ts in all_pairs:
        cell_timestamps.setdefault(cell, []).append(ts)

    print(f"  Positive pairs: {n_pos}, Negative pairs: {n_neg}, Total: {len(all_pairs)}")

    # Generate consistent socioeconomic data per cell (not random per sample)
    cell_socio = {}
    for cell in sampled_cells:
        cell_socio[cell] = {
            "population_density": rng.uniform(50, 5000),
            "median_income_normalized": rng.uniform(0.3, 1.2),
            "critical_facility_density": rng.uniform(0, 10),
            "housing_age_median": rng.uniform(10, 60),
        }

    total_pairs = len(all_pairs)
    print(f"  {len(sampled_cells)} cells, {total_pairs:,} total pairs")

    rows = []
    t0 = time.time()

    for ci, cell in enumerate(sampled_cells):
        # Pre-filter data for this cell and neighbors
        neighbors = list(h3.grid_disk(cell, 2))
        cell_weather = weather_df[weather_df["h3_index_res7"].isin(neighbors)]
        cell_outage = outage_df[outage_df["h3_index_res7"].isin(neighbors)] if not outage_df.empty else pd.DataFrame()

        for ts in cell_timestamps.get(cell, []):
            ts_dt = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)

            # Temporal features
            feats = temporal.rolling_weather_stats(cell_weather, ts_dt)
            feats.update(temporal.cyclical_encoding(ts_dt))

            if not cell_outage.empty and "outage_fraction" in cell_outage.columns:
                feats.update(temporal.lag_outage_features(cell_outage, ts_dt))
                feats.update(temporal.trend_features(cell_outage, ts_dt))
            else:
                for lag in temporal.LAG_HOURS:
                    feats[f"lag_outage_{lag}h"] = 0.0
                for w in [3, 6, 12]:
                    feats[f"trend_outage_{w}h"] = 0.0

            feats.update(temporal.grid_load_features(load_df, ts_dt))

            # Spatial features
            feats.update(spatial.neighborhood_aggregation(
                cell, cell_weather, cell_outage if not cell_outage.empty else pd.DataFrame(), ts_dt
            ))
            feats.update(spatial.infrastructure_features(None))

            # Compound event features
            feats.update(compound.compute_all(cell_weather, ts_dt))

            # Socioeconomic (consistent per cell)
            socio = cell_socio.get(cell, {})
            feats["population_density"] = socio.get("population_density", 500)
            feats["median_income_normalized"] = socio.get("median_income_normalized", 0.6)
            feats["critical_facility_density"] = socio.get("critical_facility_density", 2)
            feats["housing_age_median"] = socio.get("housing_age_median", 30)

            # Target: outage in next 24 hours
            feats["h3_cell"] = cell
            feats["timestamp"] = ts

            if not outage_df.empty and "h3_index_res7" in outage_df.columns:
                target_start = pd.Timestamp(ts_dt)
                target_end = pd.Timestamp(ts_dt) + timedelta(hours=24)
                target_mask = (
                    (outage_df["h3_index_res7"] == cell)
                    & (outage_df["observed_at"] >= target_start)
                    & (outage_df["observed_at"] <= target_end)
                    & (outage_df["outage_fraction"] > 0.01)
                )
                feats["target_outage"] = int(target_mask.any())
                matching = outage_df.loc[target_mask]
                feats["target_max_outage_fraction"] = (
                    float(matching["outage_fraction"].max()) if not matching.empty else 0.0
                )
            else:
                feats["target_outage"] = 0
                feats["target_max_outage_fraction"] = 0.0

            rows.append(feats)

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = len(rows) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {ci+1}/{len(sampled_cells)} cells, {len(rows)} samples ({rate:.0f} samples/sec)")

    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"\n  Built {len(df):,} samples with {len(df.columns) - 4} features in {elapsed:.1f}s")
    print(f"  Positive rate: {df['target_outage'].mean():.4f}")
    return df


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build outage prediction training dataset")
    parser.add_argument("--states", nargs="+", default=["TX"], help="State codes (default: TX)")
    parser.add_argument("--sample-size", type=int, default=15_000, help="Target sample count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    data_dir = ROOT / "data" / "raw"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else out_dir / "training_dataset.parquet"

    print("=" * 70)
    print("OUTAGE PREDICTION - Dataset Builder")
    print("=" * 70)
    print(f"  States: {args.states}")
    print(f"  Sample size: {args.sample_size:,}")
    print(f"  Output: {output_path}")

    # Step 1: Load NOAA storms
    print("\n[1] Loading NOAA Storm Events...")
    storms_raw = load_noaa_storms(data_dir, args.states)
    weather_df = transform_storms(storms_raw)
    print(f"  Transformed: {len(weather_df):,} geocoded events")
    print(f"  Date range: {weather_df['event_time'].min()} to {weather_df['event_time'].max()}")
    print(f"  Unique H3 cells: {weather_df['h3_index_res7'].nunique():,}")
    print(f"  Event types: {weather_df['event_type'].value_counts().head(10).to_dict()}")

    # Step 2: Generate outage observations
    outage_df = generate_outage_observations(weather_df, rng)

    # Step 3: Generate grid load
    load_df = generate_grid_load(
        weather_df["event_time"].min(),
        weather_df["event_time"].max(),
        rng,
    )

    # Step 4: Compute features
    training_df = compute_training_features(
        weather_df, outage_df, load_df,
        sample_size=args.sample_size,
        rng=rng,
    )

    # Step 5: Save
    print(f"\n[5] Saving to {output_path}...")
    training_df.to_parquet(output_path, index=False)

    # Also save intermediate data for reproducibility
    weather_df.to_parquet(out_dir / "weather_events.parquet", index=False)
    outage_df.to_parquet(out_dir / "outage_observations.parquet", index=False)
    load_df.to_parquet(out_dir / "grid_load.parquet", index=False)

    print(f"\n{'=' * 70}")
    print("DATASET BUILD COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Training samples: {len(training_df):,}")
    print(f"  Features: {len([c for c in training_df.columns if c not in ('h3_cell', 'timestamp', 'target_outage', 'target_max_outage_fraction')])}")
    print(f"  Positive rate: {training_df['target_outage'].mean():.4f}")
    print(f"  Weather events: {len(weather_df):,}")
    print(f"  Outage observations: {len(outage_df):,}")
    print(f"  Files saved to: {out_dir}")


if __name__ == "__main__":
    main()

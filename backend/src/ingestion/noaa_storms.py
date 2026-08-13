"""NOAA Storm Events Database ingestor.

Processes bulk CSV downloads from:
https://www.ncdc.noaa.gov/stormevents/ftp.jsp

Handles damage string parsing (K/M/B suffixes), geocoding via
BEGIN_LAT/BEGIN_LON, and H3 index computation at resolutions 7 and 9.
"""

import re
from datetime import date
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseIngestor, ValidationResult

logger = structlog.get_logger(__name__)

SEVERE_EVENT_TYPES = frozenset(
    [
        "Thunderstorm Wind",
        "High Wind",
        "Heavy Snow",
        "Ice Storm",
        "Winter Storm",
        "Winter Weather",
        "Hurricane",
        "Hurricane (Typhoon)",
        "Tornado",
        "Hail",
        "Excessive Heat",
        "Heat",
        "Flash Flood",
        "Flood",
        "Coastal Flood",
        "Tropical Storm",
        "Blizzard",
        "Drought",
        "Wildfire",
        "Strong Wind",
    ]
)

DAMAGE_MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def parse_damage_value(val: str | float | None) -> float:
    """Parse NOAA damage strings like '25K', '1.5M', '0.00K' into numeric values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    val = str(val).strip().upper()
    if not val or val == "0" or val == "NAN":
        return 0.0
    match = re.match(r"^([\d.]+)([KMB])?$", val)
    if match:
        number = float(match.group(1))
        suffix = match.group(2)
        return number * DAMAGE_MULTIPLIERS.get(suffix, 1) if suffix else number
    try:
        return float(val)
    except ValueError:
        return 0.0


class NoaaStormsIngestor(BaseIngestor):
    """Ingestor for NOAA Storm Events bulk CSV data."""

    source_name = "noaa_storms"

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    async def fetch(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Load storm events from local CSV files.

        Looks for files matching pattern: *storm_events_details*.csv
        in data_dir/noaa_storms/ or data_dir/ directly.
        """
        search_dirs = [
            self.data_dir / "noaa_storms",
            self.data_dir,
        ]

        csv_files = []
        for d in search_dirs:
            if d.exists():
                csv_files.extend(d.glob("*storm_events*details*.csv"))
                csv_files.extend(d.glob("*StormEvents*details*.csv"))

        if not csv_files:
            logger.warning("noaa_storms.no_files", data_dir=str(self.data_dir))
            return pd.DataFrame()

        csv_files = list(set(csv_files))
        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                frames.append(df)
                logger.info("noaa_storms.loaded_file", path=str(f), records=len(df))
            except Exception as e:
                logger.error("noaa_storms.read_error", path=str(f), error=str(e))

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        col_map = {c: c.upper() for c in combined.columns}
        combined.rename(columns=col_map, inplace=True)

        if "BEGIN_DATE_TIME" in combined.columns:
            combined["BEGIN_DATE_TIME"] = pd.to_datetime(
                combined["BEGIN_DATE_TIME"], format="mixed", errors="coerce"
            )
            mask = (combined["BEGIN_DATE_TIME"].dt.date >= start_date) & (
                combined["BEGIN_DATE_TIME"].dt.date <= end_date
            )
            combined = combined.loc[mask]

        if region_code and "STATE_FIPS" in combined.columns:
            state_fips_map = {
                "TX": "48",
                "CA": "06",
                "FL": "12",
                "NY": "36",
                "PA": "42",
                "OH": "39",
                "IL": "17",
                "GA": "13",
                "NC": "37",
                "MI": "26",
                "NJ": "34",
                "VA": "51",
            }
            fips = state_fips_map.get(region_code)
            if fips:
                combined["STATE_FIPS"] = combined["STATE_FIPS"].astype(str).str.zfill(2)
                combined = combined[combined["STATE_FIPS"] == fips]

        return combined.reset_index(drop=True)

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate storm events data."""
        total = len(df)
        errors = []
        warnings = []

        required_cols = ["EVENT_TYPE", "BEGIN_DATE_TIME"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
                return df.head(0), ValidationResult(
                    valid=False,
                    total_records=total,
                    valid_records=0,
                    invalid_records=total,
                    errors=errors,
                )

        valid_mask = df["BEGIN_DATE_TIME"].notna() & df["EVENT_TYPE"].notna()

        if "BEGIN_LAT" in df.columns and "BEGIN_LON" in df.columns:
            geo_valid = (
                df["BEGIN_LAT"].notna()
                & df["BEGIN_LON"].notna()
                & (df["BEGIN_LAT"].between(-90, 90))
                & (df["BEGIN_LON"].between(-180, 180))
                & ~((df["BEGIN_LAT"] == 0) & (df["BEGIN_LON"] == 0))
            )
            geo_invalid_count = (~geo_valid & valid_mask).sum()
            if geo_invalid_count > 0:
                warnings.append(f"{geo_invalid_count} records with invalid/missing coordinates")
            valid_mask = valid_mask & geo_valid

        clean_df = df[valid_mask].copy()
        invalid_count = total - len(clean_df)

        return clean_df, ValidationResult(
            valid=len(clean_df) > 0,
            total_records=total,
            valid_records=len(clean_df),
            invalid_records=invalid_count,
            errors=errors,
            warnings=warnings,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform validated NOAA data into the weather_events schema."""
        result = pd.DataFrame()
        result["event_time"] = df["BEGIN_DATE_TIME"]
        result["source"] = "noaa_storms"
        result["event_type"] = df["EVENT_TYPE"].str.strip()

        result["magnitude"] = pd.to_numeric(df.get("MAGNITUDE"), errors="coerce").fillna(0)
        result["magnitude_type"] = df.get("MAGNITUDE_TYPE", pd.Series(dtype=str))

        result["lat"] = pd.to_numeric(df["BEGIN_LAT"], errors="coerce")
        result["lon"] = pd.to_numeric(df["BEGIN_LON"], errors="coerce")

        result["state_fips"] = df.get("STATE_FIPS", pd.Series(dtype=str))
        if "STATE_FIPS" in df.columns:
            result["state_fips"] = result["state_fips"].astype(str).str.zfill(2)

        result["county_fips"] = df.get("CZ_FIPS", pd.Series(dtype=str))
        if "CZ_FIPS" in df.columns:
            result["county_fips"] = result["county_fips"].astype(str).str.zfill(3)

        result["damage_property"] = df.get("DAMAGE_PROPERTY", pd.Series(dtype=str)).apply(
            parse_damage_value
        )
        result["damage_crops"] = df.get("DAMAGE_CROPS", pd.Series(dtype=str)).apply(
            parse_damage_value
        )
        result["injuries"] = pd.to_numeric(df.get("INJURIES_DIRECT", 0), errors="coerce").fillna(
            0
        ).astype(int) + pd.to_numeric(df.get("INJURIES_INDIRECT", 0), errors="coerce").fillna(
            0
        ).astype(int)

        result["deaths"] = pd.to_numeric(df.get("DEATHS_DIRECT", 0), errors="coerce").fillna(
            0
        ).astype(int) + pd.to_numeric(df.get("DEATHS_INDIRECT", 0), errors="coerce").fillna(
            0
        ).astype(int)

        result["narrative"] = df.get("EVENT_NARRATIVE", "")
        result["episode_id"] = pd.to_numeric(df.get("EPISODE_ID"), errors="coerce")

        h3_res7 = []
        h3_res9 = []
        for _, row in result.iterrows():
            lat, lon = row["lat"], row["lon"]
            if pd.notna(lat) and pd.notna(lon):
                try:
                    h3_res7.append(h3.latlng_to_cell(lat, lon, 7))
                    h3_res9.append(h3.latlng_to_cell(lat, lon, 9))
                except Exception:
                    h3_res7.append(None)
                    h3_res9.append(None)
            else:
                h3_res7.append(None)
                h3_res9.append(None)

        result["h3_index_res7"] = h3_res7
        result["h3_index_res9"] = h3_res9

        return result

    async def load(self, df: pd.DataFrame, session: AsyncSession) -> int:
        """Insert transformed records into the weather_events table."""
        if df.empty:
            return 0

        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "event_time": row["event_time"],
                    "source": row["source"],
                    "event_type": row["event_type"],
                    "magnitude": float(row["magnitude"]) if pd.notna(row["magnitude"]) else None,
                    "magnitude_type": row.get("magnitude_type"),
                    "h3_index_res7": row.get("h3_index_res7"),
                    "h3_index_res9": row.get("h3_index_res9"),
                    "state_fips": row.get("state_fips"),
                    "county_fips": row.get("county_fips"),
                    "damage_property": float(row["damage_property"])
                    if pd.notna(row["damage_property"])
                    else 0,
                    "damage_crops": float(row["damage_crops"])
                    if pd.notna(row["damage_crops"])
                    else 0,
                    "injuries": int(row.get("injuries", 0)),
                    "deaths": int(row.get("deaths", 0)),
                    "narrative": str(row.get("narrative", "")),
                    "episode_id": int(row["episode_id"])
                    if pd.notna(row.get("episode_id"))
                    else None,
                }
            )

        batch_size = 1000
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            placeholders = []
            params = {}
            for j, rec in enumerate(batch):
                keys = list(rec.keys())
                ph = ", ".join(f":v{i + j}_{k}" for k in keys)
                placeholders.append(f"({ph})")
                for k in keys:
                    params[f"v{i + j}_{k}"] = rec[k]

            cols = ", ".join(keys)
            values_str = ", ".join(placeholders)
            stmt = text(f"INSERT INTO weather_events ({cols}) VALUES {values_str}")
            await session.execute(stmt, params)
            total_inserted += len(batch)

        await session.commit()
        logger.info("noaa_storms.loaded", records=total_inserted)
        return total_inserted

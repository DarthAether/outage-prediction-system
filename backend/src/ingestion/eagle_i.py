"""EAGLE-I outage data ingestor.

Processes county-level power outage CSVs from the DOE EAGLE-I system.
This is the ground truth target variable for the outage prediction model.
Data source: data/raw/eagle_i/ directory with county-level outage snapshots.
"""

from datetime import date
from pathlib import Path

import h3
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseIngestor, ValidationResult

logger = structlog.get_logger(__name__)

COUNTY_CENTROIDS: dict[str, tuple[float, float]] = {
    # Texas counties (state FIPS 48)
    "48201": (29.7604, -95.3698),  # Harris (Houston)
    "48113": (32.7767, -96.7970),  # Dallas
    "48029": (29.4241, -98.4936),  # Bexar (San Antonio)
    "48439": (30.2672, -97.7431),  # Travis (Austin)
    "48141": (32.7555, -97.3308),  # Tarrant (Fort Worth)
    "48085": (33.0198, -96.6989),  # Collin
    "48157": (26.2034, -98.2300),  # Hidalgo
    "48215": (31.7619, -106.4850),  # El Paso
    "48339": (30.0802, -94.1266),  # Montgomery
    "48167": (29.3013, -94.7977),  # Galveston
    "48121": (33.1972, -96.6150),  # Denton
    "48453": (30.2672, -97.7431),  # Travis
    "48245": (32.3513, -95.3010),  # Jefferson
    "48491": (33.9137, -98.5264),  # Williamson
    "48303": (32.4487, -99.7331),  # Lubbock
    # California counties (state FIPS 06)
    "06037": (34.0522, -118.2437),  # Los Angeles
    "06073": (32.7157, -117.1611),  # San Diego
    "06059": (33.7175, -117.8311),  # Orange
    "06065": (33.9533, -117.3962),  # Riverside
    "06071": (34.9592, -116.4194),  # San Bernardino
    "06085": (37.3382, -121.8863),  # Santa Clara
    "06013": (37.9161, -122.3108),  # Contra Costa
    "06001": (37.6017, -122.0574),  # Alameda
    "06075": (37.7749, -122.4194),  # San Francisco
    "06067": (38.5816, -121.4944),  # Sacramento
    "06081": (37.4323, -122.3232),  # San Mateo
    "06029": (35.3733, -119.0187),  # Kern
    # Florida counties (state FIPS 12)
    "12086": (25.7617, -80.1918),  # Miami-Dade
    "12011": (26.1224, -80.1373),  # Broward
    "12099": (26.7153, -80.0534),  # Palm Beach
    "12057": (28.5383, -81.3792),  # Hillsborough (Tampa)
    "12095": (28.5383, -81.3792),  # Orange (Orlando)
    "12031": (30.3322, -81.6557),  # Duval (Jacksonville)
    "12103": (27.3364, -82.5307),  # Pinellas
    "12071": (26.1420, -81.7948),  # Lee
    "12009": (28.0222, -80.6250),  # Brevard
    "12117": (27.4467, -82.3453),  # Seminole
    "12105": (28.2920, -81.4076),  # Polk
    "12115": (27.3364, -82.5307),  # Sarasota
}


class EagleIIngestor(BaseIngestor):
    """Ingestor for EAGLE-I power outage data (ground truth target)."""

    source_name = "eagle_i"

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    async def fetch(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Load county-level outage CSVs from data/raw/eagle_i/."""
        search_dirs = [
            self.data_dir / "eagle_i",
            self.data_dir,
        ]

        csv_files: list[Path] = []
        for d in search_dirs:
            if d.exists():
                csv_files.extend(d.glob("*eagle*i*.csv"))
                csv_files.extend(d.glob("*outage*.csv"))

        if not csv_files:
            logger.warning("eagle_i.no_files", data_dir=str(self.data_dir))
            return pd.DataFrame()

        csv_files = list(set(csv_files))
        frames: list[pd.DataFrame] = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                frames.append(df)
                logger.info("eagle_i.loaded_file", path=str(f), records=len(df))
            except Exception as e:
                logger.error("eagle_i.read_error", path=str(f), error=str(e))

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        col_map: dict[str, str] = {}
        for c in combined.columns:
            lower = c.strip().lower().replace(" ", "_")
            col_map[c] = lower
        combined.rename(columns=col_map, inplace=True)

        ts_col = None
        for candidate in ["timestamp", "datetime", "date_time", "recorded_at", "time"]:
            if candidate in combined.columns:
                ts_col = candidate
                break

        if ts_col:
            combined[ts_col] = pd.to_datetime(combined[ts_col], format="mixed", errors="coerce")
            mask = (combined[ts_col].dt.date >= start_date) & (combined[ts_col].dt.date <= end_date)
            combined = combined.loc[mask]
            combined.rename(columns={ts_col: "timestamp"}, inplace=True)

        if region_code:
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
                "LA": "22",
                "WA": "53",
                "OR": "41",
                "CO": "08",
            }
            fips = state_fips_map.get(region_code)
            if fips and "state_fips" in combined.columns:
                combined["state_fips"] = combined["state_fips"].astype(str).str.zfill(2)
                combined = combined[combined["state_fips"] == fips]

        return combined.reset_index(drop=True)

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate outage records for required fields and value constraints."""
        total = len(df)
        errors: list[str] = []
        warnings: list[str] = []

        required_cols = ["county_fips", "customers_out", "total_customers"]
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

        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        df["customers_out"] = pd.to_numeric(df["customers_out"], errors="coerce")
        df["total_customers"] = pd.to_numeric(df["total_customers"], errors="coerce")

        valid_mask = (
            df["county_fips"].str.match(r"^\d{5}$")
            & df["customers_out"].notna()
            & (df["customers_out"] >= 0)
            & df["total_customers"].notna()
            & (df["total_customers"] > 0)
        )

        if "state_fips" in df.columns:
            df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
            fips_valid = df["state_fips"].str.match(r"^\d{2}$")
            invalid_fips = (~fips_valid & valid_mask).sum()
            if invalid_fips > 0:
                warnings.append(f"{invalid_fips} records with invalid state FIPS")
            valid_mask = valid_mask & fips_valid

        if "timestamp" in df.columns:
            ts_valid = df["timestamp"].notna()
            valid_mask = valid_mask & ts_valid

        overflow_mask = df["customers_out"] > df["total_customers"]
        overflow_count = (overflow_mask & valid_mask).sum()
        if overflow_count > 0:
            warnings.append(
                f"{overflow_count} records where customers_out > total_customers (capped)"
            )

        clean_df = df[valid_mask].copy()
        clean_df.loc[clean_df["customers_out"] > clean_df["total_customers"], "customers_out"] = (
            clean_df["total_customers"]
        )

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
        """Compute outage fraction, H3 index, and map to outage_observations schema."""
        result = pd.DataFrame()

        result["timestamp"] = df.get("timestamp", pd.Series(dtype="datetime64[ns]"))
        result["county_fips"] = df["county_fips"]
        result["state_fips"] = df.get("state_fips", df["county_fips"].str[:2])
        result["customers_out"] = df["customers_out"].astype(int)
        result["total_customers"] = df["total_customers"].astype(int)
        result["outage_fraction"] = df["customers_out"] / df["total_customers"]

        h3_res7: list[str | None] = []
        h3_res9: list[str | None] = []
        lats: list[float | None] = []
        lons: list[float | None] = []

        for fips in df["county_fips"]:
            centroid = COUNTY_CENTROIDS.get(fips)
            if centroid:
                lat, lon = centroid
                lats.append(lat)
                lons.append(lon)
                try:
                    h3_res7.append(h3.latlng_to_cell(lat, lon, 7))
                    h3_res9.append(h3.latlng_to_cell(lat, lon, 9))
                except Exception:
                    h3_res7.append(None)
                    h3_res9.append(None)
            else:
                lats.append(None)
                lons.append(None)
                h3_res7.append(None)
                h3_res9.append(None)

        result["lat"] = lats
        result["lon"] = lons
        result["h3_index_res7"] = h3_res7
        result["h3_index_res9"] = h3_res9
        result["source"] = "eagle_i"

        return result

    async def load(self, df: pd.DataFrame, session: AsyncSession) -> int:
        """Batch insert into outage_observations table."""
        if df.empty:
            return 0

        records: list[dict] = []
        for _, row in df.iterrows():
            records.append(
                {
                    "timestamp": row["timestamp"],
                    "county_fips": row["county_fips"],
                    "state_fips": row["state_fips"],
                    "customers_out": int(row["customers_out"]),
                    "total_customers": int(row["total_customers"]),
                    "outage_fraction": float(row["outage_fraction"]),
                    "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
                    "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
                    "h3_index_res7": row.get("h3_index_res7"),
                    "h3_index_res9": row.get("h3_index_res9"),
                    "source": row["source"],
                }
            )

        batch_size = 1000
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            placeholders = []
            params: dict = {}
            for j, rec in enumerate(batch):
                keys = list(rec.keys())
                ph = ", ".join(f":v{i + j}_{k}" for k in keys)
                placeholders.append(f"({ph})")
                for k in keys:
                    params[f"v{i + j}_{k}"] = rec[k]

            cols = ", ".join(keys)
            values_str = ", ".join(placeholders)
            stmt = text(
                f"INSERT INTO outage_observations ({cols}) VALUES {values_str} "
                f"ON CONFLICT (county_fips, timestamp) DO UPDATE SET "
                f"customers_out = EXCLUDED.customers_out, "
                f"total_customers = EXCLUDED.total_customers, "
                f"outage_fraction = EXCLUDED.outage_fraction"
            )
            await session.execute(stmt, params)
            total_inserted += len(batch)

        await session.commit()
        logger.info("eagle_i.loaded", records=total_inserted)
        return total_inserted

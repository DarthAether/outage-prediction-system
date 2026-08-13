"""ERCOT grid load data ingestor.

Processes grid load and capacity data from the Electric Reliability
Council of Texas. Reads from CSV files in data/raw/ercot/ or generates
realistic simulated data for development/testing when no files are found.
"""

import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseIngestor, ValidationResult

logger = structlog.get_logger(__name__)

ERCOT_REGIONS = {
    "COAST": {"lat": 29.76, "lon": -95.37},
    "EAST": {"lat": 31.76, "lon": -94.18},
    "FAR_WEST": {"lat": 31.43, "lon": -103.49},
    "NORTH": {"lat": 32.78, "lon": -96.80},
    "NORTH_CENTRAL": {"lat": 33.20, "lon": -97.13},
    "SOUTH": {"lat": 27.80, "lon": -97.40},
    "SOUTH_CENTRAL": {"lat": 29.42, "lon": -98.49},
    "WEST": {"lat": 31.99, "lon": -102.08},
}


def _generate_simulated_data(
    start_date: date, end_date: date, interval_minutes: int = 60
) -> pd.DataFrame:
    """Generate realistic ERCOT grid load data for development purposes.

    Models diurnal load patterns, seasonal variation, and random fluctuations
    with physically plausible values for the Texas grid.
    """
    timestamps = pd.date_range(
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date, datetime.max.time()),
        freq=f"{interval_minutes}min",
    )

    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []

    base_capacity_mw = 85_000.0
    base_load_mw = 45_000.0

    for ts in timestamps:
        hour = ts.hour
        day_of_year = ts.day_of_year

        diurnal = 1.0 + 0.25 * math.sin(math.pi * (hour - 6) / 12)
        if hour < 5 or hour > 22:
            diurnal = 0.75

        seasonal = 1.0 + 0.15 * math.sin(2 * math.pi * (day_of_year - 172) / 365)
        if 150 < day_of_year < 270:
            seasonal += 0.10

        noise = rng.normal(1.0, 0.03)
        load_mw = base_load_mw * diurnal * seasonal * noise
        load_mw = max(25_000.0, min(load_mw, 80_000.0))

        capacity_noise = rng.normal(1.0, 0.01)
        capacity_mw = base_capacity_mw * capacity_noise
        capacity_mw = max(load_mw * 1.05, capacity_mw)

        freq_deviation = rng.normal(0, 0.02)
        frequency_hz = 60.0 + freq_deviation

        rows.append(
            {
                "timestamp": ts,
                "load_mw": round(load_mw, 2),
                "capacity_mw": round(capacity_mw, 2),
                "frequency_hz": round(frequency_hz, 4),
                "region": rng.choice(list(ERCOT_REGIONS.keys())),
            }
        )

    return pd.DataFrame(rows)


class ErcotIngestor(BaseIngestor):
    """Ingestor for ERCOT grid load and capacity data."""

    source_name = "ercot"

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    async def fetch(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Load ERCOT data from CSV files or generate simulated data."""
        search_dirs = [
            self.data_dir / "ercot",
            self.data_dir,
        ]

        csv_files: list[Path] = []
        for d in search_dirs:
            if d.exists():
                csv_files.extend(d.glob("*ercot*.csv"))
                csv_files.extend(d.glob("*grid_load*.csv"))

        if csv_files:
            csv_files = list(set(csv_files))
            frames: list[pd.DataFrame] = []
            for f in csv_files:
                try:
                    df = pd.read_csv(f, low_memory=False)
                    frames.append(df)
                    logger.info("ercot.loaded_file", path=str(f), records=len(df))
                except Exception as e:
                    logger.error("ercot.read_error", path=str(f), error=str(e))

            if frames:
                combined = pd.concat(frames, ignore_index=True)
                col_map = {c: c.strip().lower().replace(" ", "_") for c in combined.columns}
                combined.rename(columns=col_map, inplace=True)

                ts_col = None
                for candidate in ["timestamp", "datetime", "interval_start", "time"]:
                    if candidate in combined.columns:
                        ts_col = candidate
                        break

                if ts_col:
                    combined[ts_col] = pd.to_datetime(
                        combined[ts_col], format="mixed", errors="coerce"
                    )
                    mask = (combined[ts_col].dt.date >= start_date) & (
                        combined[ts_col].dt.date <= end_date
                    )
                    combined = combined.loc[mask]
                    if ts_col != "timestamp":
                        combined.rename(columns={ts_col: "timestamp"}, inplace=True)

                return combined.reset_index(drop=True)

        logger.info(
            "ercot.generating_simulated",
            start_date=str(start_date),
            end_date=str(end_date),
        )
        return _generate_simulated_data(start_date, end_date)

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate grid load data for physical constraints."""
        total = len(df)
        errors: list[str] = []
        warnings: list[str] = []

        required_cols = ["load_mw", "frequency_hz"]
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

        df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
        df["frequency_hz"] = pd.to_numeric(df["frequency_hz"], errors="coerce")

        if "capacity_mw" in df.columns:
            df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")

        valid_mask = (
            df["load_mw"].notna()
            & (df["load_mw"] > 0)
            & df["frequency_hz"].notna()
            & (df["frequency_hz"] >= 59.5)
            & (df["frequency_hz"] <= 60.5)
        )

        if "capacity_mw" in df.columns:
            cap_valid = df["capacity_mw"].notna() & (df["capacity_mw"] > 0)
            cap_invalid = (~cap_valid & valid_mask).sum()
            if cap_invalid > 0:
                warnings.append(f"{cap_invalid} records with invalid capacity")
            valid_mask = valid_mask & cap_valid

        freq_extreme = (df["frequency_hz"] < 59.9) | (df["frequency_hz"] > 60.1)
        extreme_count = (freq_extreme & valid_mask).sum()
        if extreme_count > 0:
            warnings.append(f"{extreme_count} records with frequency deviation > 0.1 Hz")

        if "capacity_mw" in df.columns:
            overload = valid_mask & (df["load_mw"] > df["capacity_mw"])
            if overload.sum() > 0:
                warnings.append(f"{overload.sum()} records where load exceeds capacity")

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
        """Compute reserve margin and map to grid_load schema."""
        result = pd.DataFrame()

        result["timestamp"] = df.get("timestamp", pd.Series(dtype="datetime64[ns]"))
        result["load_mw"] = df["load_mw"]
        result["frequency_hz"] = df["frequency_hz"]

        if "capacity_mw" in df.columns:
            result["capacity_mw"] = df["capacity_mw"]
            result["reserve_margin_pct"] = (
                (df["capacity_mw"] - df["load_mw"]) / df["capacity_mw"] * 100
            ).round(4)
        else:
            result["capacity_mw"] = None
            result["reserve_margin_pct"] = None

        if "region" in df.columns:
            result["region"] = df["region"]
        else:
            result["region"] = "ERCOT_SYSTEM"

        result["source"] = "ercot"

        low_reserve = result["reserve_margin_pct"].notna() & (result["reserve_margin_pct"] < 6)
        result["grid_stress_flag"] = low_reserve.astype(int)

        return result

    async def load(self, df: pd.DataFrame, session: AsyncSession) -> int:
        """Insert grid load records into the grid_load table."""
        if df.empty:
            return 0

        records: list[dict] = []
        for _, row in df.iterrows():
            records.append(
                {
                    "timestamp": row["timestamp"],
                    "load_mw": float(row["load_mw"]),
                    "capacity_mw": float(row["capacity_mw"])
                    if pd.notna(row.get("capacity_mw"))
                    else None,
                    "frequency_hz": float(row["frequency_hz"]),
                    "reserve_margin_pct": (
                        float(row["reserve_margin_pct"])
                        if pd.notna(row.get("reserve_margin_pct"))
                        else None
                    ),
                    "region": str(row.get("region", "ERCOT_SYSTEM")),
                    "source": row["source"],
                    "grid_stress_flag": int(row.get("grid_stress_flag", 0)),
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
                f"INSERT INTO grid_load ({cols}) VALUES {values_str} "
                f"ON CONFLICT (timestamp, region) DO UPDATE SET "
                f"load_mw = EXCLUDED.load_mw, "
                f"capacity_mw = EXCLUDED.capacity_mw, "
                f"frequency_hz = EXCLUDED.frequency_hz, "
                f"reserve_margin_pct = EXCLUDED.reserve_margin_pct"
            )
            await session.execute(stmt, params)
            total_inserted += len(batch)

        await session.commit()
        logger.info("ercot.loaded", records=total_inserted)
        return total_inserted

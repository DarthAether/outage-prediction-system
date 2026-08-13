"""EIA grid infrastructure ingestor.

Fetches electric utility and transmission infrastructure data from the
U.S. Energy Information Administration. Reads from local CSV files in
data/raw/eia/ or queries the EIA API v2 for utility-level data including
transmission line kilometers, substation counts, and generation capacity.
"""

from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseIngestor, ValidationResult

logger = structlog.get_logger(__name__)

EIA_API_BASE = "https://api.eia.gov/v2"

STATE_FIPS_MAP: dict[str, str] = {
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


class EiaIngestor(BaseIngestor):
    """Ingestor for EIA grid infrastructure and utility data."""

    source_name = "eia"

    def __init__(
        self,
        data_dir: str = "data/raw",
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        self.timeout = timeout

    async def fetch(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Load infrastructure data from CSV files or EIA API."""
        df = await self._fetch_from_csv(start_date, end_date, region_code)
        if df is not None and not df.empty:
            return df

        if self.api_key:
            return await self._fetch_from_api(start_date, end_date, region_code)

        logger.warning("eia.no_data_source", data_dir=str(self.data_dir))
        return pd.DataFrame()

    async def _fetch_from_csv(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame | None:
        """Attempt to load from local CSV files."""
        search_dirs = [
            self.data_dir / "eia",
            self.data_dir,
        ]

        csv_files: list[Path] = []
        for d in search_dirs:
            if d.exists():
                csv_files.extend(d.glob("*eia*.csv"))
                csv_files.extend(d.glob("*infrastructure*.csv"))
                csv_files.extend(d.glob("*utility*.csv"))
                csv_files.extend(d.glob("*transmission*.csv"))

        if not csv_files:
            return None

        csv_files = list(set(csv_files))
        frames: list[pd.DataFrame] = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                frames.append(df)
                logger.info("eia.loaded_file", path=str(f), records=len(df))
            except Exception as e:
                logger.error("eia.read_error", path=str(f), error=str(e))

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)

        col_map = {c: c.strip().lower().replace(" ", "_") for c in combined.columns}
        combined.rename(columns=col_map, inplace=True)

        if region_code:
            state_fips = STATE_FIPS_MAP.get(region_code)
            for col in ["state_fips", "state", "state_code"]:
                if col in combined.columns:
                    if col == "state_fips" and state_fips:
                        combined[col] = combined[col].astype(str).str.zfill(2)
                        combined = combined[combined[col] == state_fips]
                    elif col in ("state", "state_code"):
                        combined = combined[combined[col].astype(str).str.upper() == region_code]
                    break

        return combined.reset_index(drop=True)

    async def _fetch_from_api(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Fetch utility data from EIA API v2."""
        all_rows: list[dict] = []

        headers = {"X-Api-Key": self.api_key} if self.api_key else {}

        async with httpx.AsyncClient(
            headers=headers, timeout=self.timeout, follow_redirects=True
        ) as client:
            params: dict[str, str] = {
                "frequency": "annual",
                "data[0]": "generation",
                "start": str(start_date.year),
                "end": str(end_date.year),
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "offset": "0",
                "length": "5000",
            }

            if region_code:
                params["facets[stateid][]"] = region_code

            try:
                url = f"{EIA_API_BASE}/electricity/state-electricity-profiles/source-disposition"
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                api_data = data.get("response", {}).get("data", [])
                all_rows.extend(api_data)

                logger.info("eia.api_fetched", records=len(api_data))
            except httpx.HTTPStatusError as e:
                logger.error("eia.api_http_error", status=e.response.status_code)
            except httpx.RequestError as e:
                logger.error("eia.api_request_error", error=str(e))

            try:
                plant_params: dict[str, str] = {
                    "frequency": "annual",
                    "data[0]": "total-consumption-btu",
                    "start": str(start_date.year),
                    "end": str(end_date.year),
                    "offset": "0",
                    "length": "5000",
                }
                if region_code:
                    plant_params["facets[stateid][]"] = region_code

                plant_url = f"{EIA_API_BASE}/electricity/facility-fuel"
                response = await client.get(plant_url, params=plant_params)
                response.raise_for_status()
                plant_data = response.json().get("response", {}).get("data", [])
                all_rows.extend(plant_data)
                logger.info("eia.api_plants_fetched", records=len(plant_data))
            except Exception as e:
                logger.warning("eia.api_plants_error", error=str(e))

        if not all_rows:
            return pd.DataFrame()

        return pd.DataFrame(all_rows)

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate infrastructure data for numeric fields."""
        total = len(df)
        errors: list[str] = []
        warnings: list[str] = []

        numeric_candidates = [
            "transmission_line_km",
            "transmission_line_miles",
            "substations_count",
            "generation_capacity_mw",
            "total_customers",
            "peak_demand_mw",
            "generation",
            "total-consumption-btu",
        ]

        found_numeric = False
        for col in numeric_candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                found_numeric = True

        if not found_numeric:
            warnings.append("No recognized numeric infrastructure columns found")

        valid_mask = pd.Series(True, index=df.index)

        for col in numeric_candidates:
            if col in df.columns:
                neg_mask = df[col].notna() & (df[col] < 0)
                neg_count = neg_mask.sum()
                if neg_count > 0:
                    warnings.append(f"{neg_count} records with negative {col}")
                    valid_mask = valid_mask & ~neg_mask

        all_null_mask = pd.Series(True, index=df.index)
        for col in numeric_candidates:
            if col in df.columns:
                all_null_mask = all_null_mask & df[col].isna()
        valid_mask = valid_mask & ~all_null_mask

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
        """Map to infrastructure schema with standardized columns."""
        result = pd.DataFrame()

        if "county_fips" in df.columns:
            result["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        elif "state_fips" in df.columns:
            result["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
        elif "stateid" in df.columns:
            result["state_code"] = df["stateid"]
            result["state_fips"] = df["stateid"].map(STATE_FIPS_MAP)

        if "utility_id" in df.columns:
            result["utility_id"] = df["utility_id"]
        elif "plantid" in df.columns:
            result["utility_id"] = df["plantid"]

        if "utility_name" in df.columns:
            result["utility_name"] = df["utility_name"]
        elif "plantName" in df.columns:
            result["utility_name"] = df["plantName"]

        if "transmission_line_km" in df.columns:
            result["transmission_line_km"] = df["transmission_line_km"]
        elif "transmission_line_miles" in df.columns:
            result["transmission_line_km"] = df["transmission_line_miles"] * 1.60934
        else:
            result["transmission_line_km"] = None

        result["substations_count"] = df.get("substations_count", pd.Series(dtype=float))

        if "generation_capacity_mw" in df.columns:
            result["generation_capacity_mw"] = df["generation_capacity_mw"]
        elif "generation" in df.columns:
            result["generation_capacity_mw"] = pd.to_numeric(df["generation"], errors="coerce")
        else:
            result["generation_capacity_mw"] = None

        result["total_customers"] = df.get("total_customers", pd.Series(dtype=float))
        result["peak_demand_mw"] = df.get("peak_demand_mw", pd.Series(dtype=float))

        if "period" in df.columns:
            result["data_year"] = pd.to_numeric(df["period"], errors="coerce")
        else:
            result["data_year"] = None

        result["source"] = "eia"

        return result

    async def load(self, df: pd.DataFrame, session: AsyncSession) -> int:
        """Upsert infrastructure records into the infrastructure table."""
        if df.empty:
            return 0

        records: list[dict] = []
        for _, row in df.iterrows():
            rec: dict = {"source": "eia"}

            for col in [
                "county_fips",
                "state_fips",
                "state_code",
                "utility_id",
                "utility_name",
            ]:
                if col in row.index and pd.notna(row.get(col)):
                    rec[col] = str(row[col])

            for col in [
                "transmission_line_km",
                "substations_count",
                "generation_capacity_mw",
                "total_customers",
                "peak_demand_mw",
            ]:
                if col in row.index and pd.notna(row.get(col)):
                    rec[col] = float(row[col])

            if "data_year" in row.index and pd.notna(row.get("data_year")):
                rec["data_year"] = int(row["data_year"])

            records.append(rec)

        batch_size = 500
        total_inserted = 0

        all_keys = set()
        for rec in records:
            all_keys.update(rec.keys())
        all_keys_list = sorted(all_keys)

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            placeholders = []
            params: dict = {}
            for j, rec in enumerate(batch):
                ph = ", ".join(f":v{i + j}_{k}" for k in all_keys_list)
                placeholders.append(f"({ph})")
                for k in all_keys_list:
                    params[f"v{i + j}_{k}"] = rec.get(k)

            cols = ", ".join(all_keys_list)
            values_str = ", ".join(placeholders)

            conflict_col = "utility_id" if "utility_id" in all_keys_list else "county_fips"
            update_cols = [
                k for k in all_keys_list if k not in (conflict_col, "source", "data_year")
            ]
            update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

            if update_clause:
                stmt = text(
                    f"INSERT INTO infrastructure ({cols}) VALUES {values_str} "
                    f"ON CONFLICT ({conflict_col}, data_year) DO UPDATE SET {update_clause}"
                )
            else:
                stmt = text(
                    f"INSERT INTO infrastructure ({cols}) VALUES {values_str} "
                    f"ON CONFLICT DO NOTHING"
                )

            await session.execute(stmt, params)
            total_inserted += len(batch)

        await session.commit()
        logger.info("eia.loaded", records=total_inserted)
        return total_inserted

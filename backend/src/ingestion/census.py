"""Census data ingestor.

Fetches socioeconomic data from the US Census Bureau American Community
Survey (ACS) 5-Year Estimates API. Retrieves population, median household
income, housing unit age, and other vulnerability indicators by county.
No API key required for basic access.
"""

from datetime import date

import h3
import httpx
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseIngestor, ValidationResult

logger = structlog.get_logger(__name__)

CENSUS_BASE_URL = "https://api.census.gov/data/2020/acs/acs5"

ACS_VARIABLES = {
    "B01003_001E": "total_population",
    "B19013_001E": "median_household_income",
    "B25035_001E": "median_year_structure_built",
    "B25001_001E": "total_housing_units",
    "B25002_003E": "vacant_housing_units",
    "B25064_001E": "median_gross_rent",
    "B01002_001E": "median_age",
    "B17001_002E": "population_below_poverty",
    "B09021_001E": "population_in_households",
}

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
    "AZ": "04",
    "MA": "25",
    "TN": "47",
    "IN": "18",
    "MO": "29",
    "MD": "24",
    "WI": "55",
    "MN": "27",
    "SC": "45",
    "AL": "01",
    "KY": "21",
    "OK": "40",
    "CT": "09",
    "IA": "19",
    "MS": "28",
    "AR": "05",
    "KS": "20",
    "NV": "32",
    "NE": "31",
    "NM": "35",
}

COUNTY_AREA_SQ_MI: dict[str, float] = {
    "48201": 1777.0,  # Harris
    "48113": 880.0,  # Dallas
    "48029": 1247.0,  # Bexar
    "48439": 1023.0,  # Travis
    "48141": 863.0,  # Tarrant
    "06037": 4751.0,  # Los Angeles
    "06073": 4526.0,  # San Diego
    "06059": 948.0,  # Orange
    "06065": 7303.0,  # Riverside
    "06071": 20105.0,  # San Bernardino
    "12086": 2431.0,  # Miami-Dade
    "12011": 1320.0,  # Broward
    "12099": 2383.0,  # Palm Beach
    "12057": 1266.0,  # Hillsborough
    "12095": 1003.0,  # Orange (FL)
}

COUNTY_CENTROIDS: dict[str, tuple[float, float]] = {
    "48201": (29.76, -95.37),
    "48113": (32.78, -96.80),
    "48029": (29.42, -98.49),
    "48439": (30.27, -97.74),
    "48141": (32.76, -97.33),
    "06037": (34.05, -118.24),
    "06073": (32.72, -117.16),
    "06059": (33.72, -117.83),
    "06065": (33.95, -117.40),
    "06071": (34.96, -116.42),
    "12086": (25.76, -80.19),
    "12011": (26.12, -80.14),
    "12099": (26.72, -80.05),
    "12057": (28.54, -81.38),
    "12095": (28.54, -81.38),
}


class CensusIngestor(BaseIngestor):
    """Ingestor for US Census Bureau ACS socioeconomic data."""

    source_name = "census"

    def __init__(
        self,
        api_key: str | None = None,
        target_states: list[str] | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.target_states = target_states or ["TX", "CA", "FL"]
        self.timeout = timeout

    async def fetch(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Fetch ACS data from the Census API by county."""
        states_to_query = [region_code] if region_code else self.target_states
        all_rows: list[dict] = []

        variable_list = ",".join(ACS_VARIABLES.keys())

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            for state_code in states_to_query:
                state_fips = STATE_FIPS_MAP.get(state_code)
                if not state_fips:
                    logger.warning("census.unknown_state", state=state_code)
                    continue

                params: dict[str, str] = {
                    "get": f"NAME,{variable_list}",
                    "for": "county:*",
                    "in": f"state:{state_fips}",
                }
                if self.api_key:
                    params["key"] = self.api_key

                try:
                    response = await client.get(CENSUS_BASE_URL, params=params)
                    response.raise_for_status()
                    data = response.json()

                    if len(data) < 2:
                        logger.warning("census.empty_response", state=state_code)
                        continue

                    headers = data[0]
                    for row_data in data[1:]:
                        row_dict = dict(zip(headers, row_data, strict=True))
                        row_dict["state_code"] = state_code
                        all_rows.append(row_dict)

                    logger.info(
                        "census.fetched_state",
                        state=state_code,
                        counties=len(data) - 1,
                    )
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "census.http_error",
                        state=state_code,
                        status=e.response.status_code,
                    )
                except httpx.RequestError as e:
                    logger.error("census.request_error", state=state_code, error=str(e))

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        rename_map: dict[str, str] = {}
        for api_var, friendly_name in ACS_VARIABLES.items():
            if api_var in df.columns:
                rename_map[api_var] = friendly_name
        df.rename(columns=rename_map, inplace=True)

        if "state" in df.columns and "county" in df.columns:
            df["state_fips"] = df["state"].astype(str).str.zfill(2)
            df["county_fips_3"] = df["county"].astype(str).str.zfill(3)
            df["county_fips"] = df["state_fips"] + df["county_fips_3"]

        return df.reset_index(drop=True)

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate census data for numeric fields and completeness."""
        total = len(df)
        errors: list[str] = []
        warnings: list[str] = []

        if "total_population" not in df.columns:
            errors.append("Missing total_population column")
            return df.head(0), ValidationResult(
                valid=False,
                total_records=total,
                valid_records=0,
                invalid_records=total,
                errors=errors,
            )

        numeric_cols = [
            "total_population",
            "median_household_income",
            "median_year_structure_built",
            "total_housing_units",
            "vacant_housing_units",
            "median_gross_rent",
            "median_age",
            "population_below_poverty",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        valid_mask = df["total_population"].notna() & (df["total_population"] > 0)

        if "median_household_income" in df.columns:
            income_valid = df["median_household_income"].notna() & (
                df["median_household_income"] > 0
            )
            income_invalid = (~income_valid & valid_mask).sum()
            if income_invalid > 0:
                warnings.append(f"{income_invalid} counties with missing/invalid income data")

        if "county_fips" in df.columns:
            fips_valid = df["county_fips"].astype(str).str.match(r"^\d{5}$")
            valid_mask = valid_mask & fips_valid

        neg_pop = (
            df.get("population_below_poverty", pd.Series(dtype=float)).fillna(0)
            > df["total_population"]
        )
        if neg_pop.sum() > 0:
            warnings.append(f"{neg_pop.sum()} counties with poverty population > total population")

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
        """Compute derived metrics and map to socioeconomic schema."""
        result = pd.DataFrame()

        result["county_fips"] = df["county_fips"]
        result["state_fips"] = df.get("state_fips", df["county_fips"].str[:2])
        result["county_name"] = df.get("NAME", "")
        result["state_code"] = df.get("state_code", "")

        result["total_population"] = df["total_population"].astype(int)

        result["population_density"] = df["county_fips"].apply(
            lambda fips: (
                (
                    df.loc[df["county_fips"] == fips, "total_population"].iloc[0]
                    / COUNTY_AREA_SQ_MI[fips]
                )
                if fips in COUNTY_AREA_SQ_MI
                else None
            )
        )

        result["median_household_income"] = df.get(
            "median_household_income", pd.Series(dtype=float)
        )
        result["median_year_structure_built"] = df.get(
            "median_year_structure_built", pd.Series(dtype=float)
        )
        result["total_housing_units"] = df.get("total_housing_units", pd.Series(dtype=float))
        result["vacant_housing_units"] = df.get("vacant_housing_units", pd.Series(dtype=float))
        result["median_gross_rent"] = df.get("median_gross_rent", pd.Series(dtype=float))
        result["median_age"] = df.get("median_age", pd.Series(dtype=float))
        result["population_below_poverty"] = df.get(
            "population_below_poverty", pd.Series(dtype=float)
        )

        if (
            result["total_population"].notna().any()
            and result["population_below_poverty"].notna().any()
        ):
            result["poverty_rate"] = (
                result["population_below_poverty"] / result["total_population"]
            ).round(6)
        else:
            result["poverty_rate"] = None

        if result["median_year_structure_built"].notna().any():
            result["avg_housing_age_years"] = (2020 - result["median_year_structure_built"]).clip(
                lower=0
            )
        else:
            result["avg_housing_age_years"] = None

        if (
            result["total_housing_units"].notna().any()
            and result["vacant_housing_units"].notna().any()
        ):
            result["vacancy_rate"] = (
                result["vacant_housing_units"] / result["total_housing_units"]
            ).round(6)
        else:
            result["vacancy_rate"] = None

        h3_res7: list[str | None] = []
        for fips in df["county_fips"]:
            centroid = COUNTY_CENTROIDS.get(fips)
            if centroid:
                try:
                    h3_res7.append(h3.latlng_to_cell(centroid[0], centroid[1], 7))
                except Exception:
                    h3_res7.append(None)
            else:
                h3_res7.append(None)
        result["h3_index_res7"] = h3_res7

        result["source"] = "census_acs5"
        result["data_year"] = 2020

        return result

    async def load(self, df: pd.DataFrame, session: AsyncSession) -> int:
        """Upsert census records into the socioeconomic table."""
        if df.empty:
            return 0

        records: list[dict] = []
        for _, row in df.iterrows():
            records.append(
                {
                    "county_fips": row["county_fips"],
                    "state_fips": row["state_fips"],
                    "county_name": str(row.get("county_name", "")),
                    "state_code": str(row.get("state_code", "")),
                    "total_population": int(row["total_population"]),
                    "population_density": (
                        float(row["population_density"])
                        if pd.notna(row.get("population_density"))
                        else None
                    ),
                    "median_household_income": (
                        float(row["median_household_income"])
                        if pd.notna(row.get("median_household_income"))
                        else None
                    ),
                    "median_year_structure_built": (
                        int(row["median_year_structure_built"])
                        if pd.notna(row.get("median_year_structure_built"))
                        else None
                    ),
                    "avg_housing_age_years": (
                        float(row["avg_housing_age_years"])
                        if pd.notna(row.get("avg_housing_age_years"))
                        else None
                    ),
                    "poverty_rate": (
                        float(row["poverty_rate"]) if pd.notna(row.get("poverty_rate")) else None
                    ),
                    "vacancy_rate": (
                        float(row["vacancy_rate"]) if pd.notna(row.get("vacancy_rate")) else None
                    ),
                    "median_age": (
                        float(row["median_age"]) if pd.notna(row.get("median_age")) else None
                    ),
                    "h3_index_res7": row.get("h3_index_res7"),
                    "source": row["source"],
                    "data_year": int(row.get("data_year", 2020)),
                }
            )

        batch_size = 500
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
                f"INSERT INTO socioeconomic ({cols}) VALUES {values_str} "
                f"ON CONFLICT (county_fips, data_year) DO UPDATE SET "
                f"total_population = EXCLUDED.total_population, "
                f"population_density = EXCLUDED.population_density, "
                f"median_household_income = EXCLUDED.median_household_income, "
                f"poverty_rate = EXCLUDED.poverty_rate, "
                f"vacancy_rate = EXCLUDED.vacancy_rate"
            )
            await session.execute(stmt, params)
            total_inserted += len(batch)

        await session.commit()
        logger.info("census.loaded", records=total_inserted)
        return total_inserted

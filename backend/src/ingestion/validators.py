"""Pandera-based validation schemas for all ingestion data sources.

Provides strict column-level type checks, nullable constraints, and
value range validations. Each schema corresponds to a target database
table and can be used for additional validation beyond what individual
ingestors perform.
"""

import pandera as pa
from pandera import Check, Column


class WeatherEventSchema(pa.DataFrameSchema):
    """Validation schema for weather_events table records.

    Covers data from NOAA storms, NWS alerts, and METAR observations.
    """

    def __init__(self):
        super().__init__(
            columns={
                "event_time": Column(
                    "datetime64[ns, UTC]",
                    nullable=False,
                    coerce=True,
                    description="Timestamp of the weather event in UTC",
                ),
                "source": Column(
                    str,
                    nullable=False,
                    checks=Check.isin(["noaa_storms", "nws", "metar"]),
                    description="Data source identifier",
                ),
                "event_type": Column(
                    str,
                    nullable=False,
                    checks=Check.str_length(min_value=1, max_value=200),
                    description="Type of weather event",
                ),
                "magnitude": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Event magnitude (units depend on source)",
                ),
                "magnitude_type": Column(
                    str,
                    nullable=True,
                    description="Units/type for the magnitude field",
                ),
                "lat": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(-90), Check.le(90)],
                    coerce=True,
                    description="Latitude in decimal degrees",
                ),
                "lon": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(-180), Check.le(180)],
                    coerce=True,
                    description="Longitude in decimal degrees",
                ),
                "h3_index_res7": Column(
                    str,
                    nullable=True,
                    checks=Check.str_length(min_value=15, max_value=15),
                    description="H3 hex index at resolution 7",
                ),
                "h3_index_res9": Column(
                    str,
                    nullable=True,
                    checks=Check.str_length(min_value=15, max_value=15),
                    description="H3 hex index at resolution 9",
                ),
                "state_fips": Column(
                    str,
                    nullable=True,
                    checks=Check.str_matches(r"^\d{2}$"),
                    description="Two-digit state FIPS code",
                ),
                "county_fips": Column(
                    str,
                    nullable=True,
                    checks=Check.str_matches(r"^\d{3,5}$"),
                    description="County FIPS code (3 or 5 digits)",
                ),
                "damage_property": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Property damage in USD",
                ),
                "damage_crops": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Crop damage in USD",
                ),
                "injuries": Column(
                    int,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Total injuries (direct + indirect)",
                ),
                "deaths": Column(
                    int,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Total deaths (direct + indirect)",
                ),
                "narrative": Column(
                    str,
                    nullable=True,
                    checks=Check.str_length(max_value=5000),
                    description="Event narrative or description",
                ),
            },
            coerce=True,
            strict=False,
        )


class OutageObservationSchema(pa.DataFrameSchema):
    """Validation schema for outage_observations table records.

    Ground truth target variable from EAGLE-I data.
    """

    def __init__(self):
        super().__init__(
            columns={
                "timestamp": Column(
                    "datetime64[ns]",
                    nullable=True,
                    coerce=True,
                    description="Observation timestamp",
                ),
                "county_fips": Column(
                    str,
                    nullable=False,
                    checks=Check.str_matches(r"^\d{5}$"),
                    description="Five-digit county FIPS code",
                ),
                "state_fips": Column(
                    str,
                    nullable=False,
                    checks=Check.str_matches(r"^\d{2}$"),
                    description="Two-digit state FIPS code",
                ),
                "customers_out": Column(
                    int,
                    nullable=False,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Number of customers without power",
                ),
                "total_customers": Column(
                    int,
                    nullable=False,
                    checks=Check.gt(0),
                    coerce=True,
                    description="Total number of customers in the area",
                ),
                "outage_fraction": Column(
                    float,
                    nullable=False,
                    checks=[Check.ge(0), Check.le(1)],
                    coerce=True,
                    description="Fraction of customers without power (0-1)",
                ),
                "lat": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(-90), Check.le(90)],
                    coerce=True,
                    description="County centroid latitude",
                ),
                "lon": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(-180), Check.le(180)],
                    coerce=True,
                    description="County centroid longitude",
                ),
                "h3_index_res7": Column(
                    str,
                    nullable=True,
                    description="H3 hex index at resolution 7",
                ),
                "h3_index_res9": Column(
                    str,
                    nullable=True,
                    description="H3 hex index at resolution 9",
                ),
                "source": Column(
                    str,
                    nullable=False,
                    checks=Check.isin(["eagle_i"]),
                    description="Data source identifier",
                ),
            },
            coerce=True,
            strict=False,
        )


class GridLoadSchema(pa.DataFrameSchema):
    """Validation schema for grid_load table records.

    ERCOT grid load, capacity, and frequency data.
    """

    def __init__(self):
        super().__init__(
            columns={
                "timestamp": Column(
                    "datetime64[ns]",
                    nullable=True,
                    coerce=True,
                    description="Observation timestamp",
                ),
                "load_mw": Column(
                    float,
                    nullable=False,
                    checks=Check.gt(0),
                    coerce=True,
                    description="System load in megawatts",
                ),
                "capacity_mw": Column(
                    float,
                    nullable=True,
                    checks=Check.gt(0),
                    coerce=True,
                    description="Available capacity in megawatts",
                ),
                "frequency_hz": Column(
                    float,
                    nullable=False,
                    checks=[Check.ge(59.0), Check.le(61.0)],
                    coerce=True,
                    description="Grid frequency in Hz (nominal 60 Hz)",
                ),
                "reserve_margin_pct": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(-50), Check.le(100)],
                    coerce=True,
                    description="Reserve margin as percentage",
                ),
                "region": Column(
                    str,
                    nullable=True,
                    description="ERCOT region or zone identifier",
                ),
                "source": Column(
                    str,
                    nullable=False,
                    checks=Check.isin(["ercot"]),
                    description="Data source identifier",
                ),
                "grid_stress_flag": Column(
                    int,
                    nullable=True,
                    checks=Check.isin([0, 1]),
                    coerce=True,
                    description="Flag indicating grid stress (reserve < 6%)",
                ),
            },
            coerce=True,
            strict=False,
        )


class InfrastructureSchema(pa.DataFrameSchema):
    """Validation schema for infrastructure table records.

    EIA utility and transmission infrastructure data.
    """

    def __init__(self):
        super().__init__(
            columns={
                "state_fips": Column(
                    str,
                    nullable=True,
                    checks=Check.str_matches(r"^\d{2}$"),
                    description="Two-digit state FIPS code",
                ),
                "county_fips": Column(
                    str,
                    nullable=True,
                    checks=Check.str_matches(r"^\d{5}$"),
                    description="Five-digit county FIPS code",
                ),
                "utility_id": Column(
                    str,
                    nullable=True,
                    description="EIA utility identifier",
                ),
                "utility_name": Column(
                    str,
                    nullable=True,
                    checks=Check.str_length(max_value=500),
                    description="Utility company name",
                ),
                "transmission_line_km": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Transmission line length in kilometers",
                ),
                "substations_count": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Number of substations",
                ),
                "generation_capacity_mw": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Generation capacity in megawatts",
                ),
                "total_customers": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Total customers served",
                ),
                "peak_demand_mw": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Peak demand in megawatts",
                ),
                "data_year": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(1990), Check.le(2030)],
                    coerce=True,
                    description="Year the data applies to",
                ),
                "source": Column(
                    str,
                    nullable=False,
                    checks=Check.isin(["eia"]),
                    description="Data source identifier",
                ),
            },
            coerce=True,
            strict=False,
        )


class SocioeconomicSchema(pa.DataFrameSchema):
    """Validation schema for socioeconomic table records.

    Census ACS demographic and economic data by county.
    """

    def __init__(self):
        super().__init__(
            columns={
                "county_fips": Column(
                    str,
                    nullable=False,
                    checks=Check.str_matches(r"^\d{5}$"),
                    description="Five-digit county FIPS code",
                ),
                "state_fips": Column(
                    str,
                    nullable=False,
                    checks=Check.str_matches(r"^\d{2}$"),
                    description="Two-digit state FIPS code",
                ),
                "county_name": Column(
                    str,
                    nullable=True,
                    description="County name from Census",
                ),
                "total_population": Column(
                    int,
                    nullable=False,
                    checks=Check.gt(0),
                    coerce=True,
                    description="Total population",
                ),
                "population_density": Column(
                    float,
                    nullable=True,
                    checks=Check.ge(0),
                    coerce=True,
                    description="Population per square mile",
                ),
                "median_household_income": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(0), Check.le(500_000)],
                    coerce=True,
                    description="Median household income in USD",
                ),
                "median_year_structure_built": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(1800), Check.le(2025)],
                    coerce=True,
                    description="Median year housing structures were built",
                ),
                "avg_housing_age_years": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(0), Check.le(250)],
                    coerce=True,
                    description="Average age of housing stock in years",
                ),
                "poverty_rate": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(0), Check.le(1)],
                    coerce=True,
                    description="Fraction of population below poverty line",
                ),
                "vacancy_rate": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(0), Check.le(1)],
                    coerce=True,
                    description="Fraction of housing units that are vacant",
                ),
                "median_age": Column(
                    float,
                    nullable=True,
                    checks=[Check.ge(0), Check.le(100)],
                    coerce=True,
                    description="Median age of the population",
                ),
                "h3_index_res7": Column(
                    str,
                    nullable=True,
                    description="H3 hex index at resolution 7",
                ),
                "source": Column(
                    str,
                    nullable=False,
                    checks=Check.isin(["census_acs5"]),
                    description="Data source identifier",
                ),
                "data_year": Column(
                    int,
                    nullable=True,
                    checks=[Check.ge(2000), Check.le(2030)],
                    coerce=True,
                    description="Year the data applies to",
                ),
            },
            coerce=True,
            strict=False,
        )


def validate_weather_events(df) -> pa.typing.DataFrame:
    """Validate a DataFrame against the WeatherEventSchema."""
    schema = WeatherEventSchema()
    return schema.validate(df, lazy=True)


def validate_outage_observations(df) -> pa.typing.DataFrame:
    """Validate a DataFrame against the OutageObservationSchema."""
    schema = OutageObservationSchema()
    return schema.validate(df, lazy=True)


def validate_grid_load(df) -> pa.typing.DataFrame:
    """Validate a DataFrame against the GridLoadSchema."""
    schema = GridLoadSchema()
    return schema.validate(df, lazy=True)


def validate_infrastructure(df) -> pa.typing.DataFrame:
    """Validate a DataFrame against the InfrastructureSchema."""
    schema = InfrastructureSchema()
    return schema.validate(df, lazy=True)


def validate_socioeconomic(df) -> pa.typing.DataFrame:
    """Validate a DataFrame against the SocioeconomicSchema."""
    schema = SocioeconomicSchema()
    return schema.validate(df, lazy=True)

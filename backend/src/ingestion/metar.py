"""METAR weather station observations ingestor.

Downloads Automated Surface Observing System (ASOS) data from the
Iowa State Mesonet archive. Provides high-frequency weather observations
including temperature, wind speed, gusts, precipitation, and visibility
from stations across the US.
"""

from datetime import date, datetime

import h3
import httpx
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseIngestor, ValidationResult

logger = structlog.get_logger(__name__)

IOWA_MESONET_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

TX_STATIONS: dict[str, tuple[float, float]] = {
    "KIAH": (29.9844, -95.3414),   # Houston Intercontinental
    "KHOU": (29.6454, -95.2789),   # Houston Hobby
    "KDFW": (32.8968, -97.0380),   # Dallas/Fort Worth
    "KDAL": (32.8471, -96.8518),   # Dallas Love Field
    "KSAT": (29.5337, -98.4698),   # San Antonio
    "KAUS": (30.1945, -97.6699),   # Austin-Bergstrom
    "KELP": (31.8072, -106.3776),  # El Paso
    "KMAF": (31.9425, -102.2019),  # Midland
    "KLBB": (33.6636, -101.8228),  # Lubbock
    "KAMA": (35.2194, -101.7060),  # Amarillo
    "KCRP": (27.7704, -97.5012),   # Corpus Christi
    "KMCJ": (26.1759, -97.9614),   # McAllen
    "KBPT": (29.9508, -94.0207),   # Beaumont/Port Arthur
    "KGGG": (32.3840, -94.7115),   # Longview
    "KACT": (31.6113, -97.2305),   # Waco
}

CA_STATIONS: dict[str, tuple[float, float]] = {
    "KLAX": (33.9425, -118.4081),  # Los Angeles
    "KSFO": (37.6197, -122.3647),  # San Francisco
    "KSAN": (32.7336, -117.1897),  # San Diego
    "KSJC": (37.3626, -121.9291),  # San Jose
    "KSAC": (38.5125, -121.4935),  # Sacramento
    "KOAK": (37.7213, -122.2208),  # Oakland
    "KONT": (34.0560, -117.6012),  # Ontario
    "KBUR": (34.2007, -118.3585),  # Burbank
}

FL_STATIONS: dict[str, tuple[float, float]] = {
    "KMIA": (25.7959, -80.2870),   # Miami
    "KFLL": (26.0726, -80.1527),   # Fort Lauderdale
    "KTPA": (27.9755, -82.5332),   # Tampa
    "KMCO": (28.4294, -81.3090),   # Orlando
    "KJAX": (30.4941, -81.6879),   # Jacksonville
    "KRSW": (26.5362, -81.7552),   # Fort Myers
    "KPBI": (26.6832, -80.0956),   # West Palm Beach
    "KTLH": (30.3965, -84.3503),   # Tallahassee
}

ALL_STATIONS: dict[str, tuple[float, float]] = {
    **TX_STATIONS, **CA_STATIONS, **FL_STATIONS
}

STATE_STATIONS: dict[str, dict[str, tuple[float, float]]] = {
    "TX": TX_STATIONS,
    "CA": CA_STATIONS,
    "FL": FL_STATIONS,
}


def _knots_to_mph(knots: float) -> float:
    return knots * 1.15078


def _fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9


class MetarIngestor(BaseIngestor):
    """Ingestor for METAR/ASOS weather station observations."""

    source_name = "metar"

    def __init__(
        self,
        target_states: list[str] | None = None,
        timeout: float = 120.0,
    ):
        self.target_states = target_states or ["TX"]
        self.timeout = timeout

    def _get_stations(self, region_code: str | None = None) -> dict[str, tuple[float, float]]:
        """Get the station dict for the target region(s)."""
        if region_code and region_code in STATE_STATIONS:
            return STATE_STATIONS[region_code]

        stations: dict[str, tuple[float, float]] = {}
        for state in self.target_states:
            stations.update(STATE_STATIONS.get(state, {}))
        return stations if stations else ALL_STATIONS

    async def fetch(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Download ASOS observations from Iowa State Mesonet archive."""
        stations = self._get_stations(region_code)
        if not stations:
            logger.warning("metar.no_stations")
            return pd.DataFrame()

        station_ids = list(stations.keys())
        all_frames: list[pd.DataFrame] = []

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            for batch_start in range(0, len(station_ids), 10):
                batch_stations = station_ids[batch_start : batch_start + 10]

                params: dict[str, str | int] = {
                    "data": "all",
                    "tz": "Etc/UTC",
                    "format": "onlycomma",
                    "latlon": "yes",
                    "elev": "no",
                    "missing": "M",
                    "trace": "T",
                    "direct": "no",
                    "report_type": "3",
                    "year1": start_date.year,
                    "month1": start_date.month,
                    "day1": start_date.day,
                    "year2": end_date.year,
                    "month2": end_date.month,
                    "day2": end_date.day,
                }

                for sid in batch_stations:
                    params[f"station"] = sid

                for sid in batch_stations:
                    try:
                        station_params = {**params, "station": sid}
                        response = await client.get(
                            IOWA_MESONET_BASE, params=station_params
                        )
                        response.raise_for_status()

                        text_content = response.text
                        if not text_content.strip() or "No results" in text_content:
                            logger.info("metar.no_data", station=sid)
                            continue

                        from io import StringIO
                        df = pd.read_csv(StringIO(text_content), low_memory=False)

                        if not df.empty:
                            df["station_id"] = sid
                            if sid in stations:
                                df["station_lat"] = stations[sid][0]
                                df["station_lon"] = stations[sid][1]
                            all_frames.append(df)
                            logger.info(
                                "metar.fetched_station",
                                station=sid,
                                records=len(df),
                            )

                    except httpx.HTTPStatusError as e:
                        logger.error(
                            "metar.http_error",
                            station=sid,
                            status=e.response.status_code,
                        )
                    except httpx.RequestError as e:
                        logger.error(
                            "metar.request_error", station=sid, error=str(e)
                        )
                    except Exception as e:
                        logger.error(
                            "metar.parse_error", station=sid, error=str(e)
                        )

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)

        col_map = {c: c.strip().lower() for c in combined.columns}
        combined.rename(columns=col_map, inplace=True)

        ts_col = None
        for candidate in ["valid", "valid(utc)", "timestamp", "datetime"]:
            if candidate in combined.columns:
                ts_col = candidate
                break

        if ts_col:
            combined[ts_col] = pd.to_datetime(combined[ts_col], errors="coerce", utc=True)
            mask = (combined[ts_col].dt.date >= start_date) & (
                combined[ts_col].dt.date <= end_date
            )
            combined = combined.loc[mask]
            if ts_col != "valid":
                combined.rename(columns={ts_col: "valid"}, inplace=True)

        return combined.reset_index(drop=True)

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate METAR observations for reasonable meteorological ranges."""
        total = len(df)
        errors: list[str] = []
        warnings: list[str] = []

        if "valid" not in df.columns and "timestamp" not in df.columns:
            errors.append("Missing timestamp column (valid or timestamp)")
            return df.head(0), ValidationResult(
                valid=False, total_records=total,
                valid_records=0, invalid_records=total, errors=errors,
            )

        ts_col = "valid" if "valid" in df.columns else "timestamp"
        valid_mask = df[ts_col].notna()

        for col, m_val in [("M", None)]:
            pass

        numeric_ranges: dict[str, tuple[float, float]] = {
            "tmpf": (-80.0, 140.0),
            "dwpf": (-80.0, 100.0),
            "sknt": (0.0, 200.0),
            "gust": (0.0, 250.0),
            "p01i": (0.0, 30.0),
            "vsby": (0.0, 100.0),
            "alti": (25.0, 35.0),
            "relh": (0.0, 100.0),
        }

        for col, (low, high) in numeric_ranges.items():
            if col in df.columns:
                df[col] = df[col].replace({"M": None, "T": 0.001, "": None})
                df[col] = pd.to_numeric(df[col], errors="coerce")
                range_valid = df[col].isna() | ((df[col] >= low) & (df[col] <= high))
                out_of_range = (~range_valid & valid_mask).sum()
                if out_of_range > 0:
                    warnings.append(f"{out_of_range} records with {col} out of range [{low}, {high}]")
                valid_mask = valid_mask & range_valid

        if "station" in df.columns or "station_id" in df.columns:
            station_col = "station" if "station" in df.columns else "station_id"
            valid_mask = valid_mask & df[station_col].notna()

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
        """Convert units, compute H3, and map to weather_events schema."""
        result = pd.DataFrame()

        ts_col = "valid" if "valid" in df.columns else "timestamp"
        result["event_time"] = df[ts_col]
        result["source"] = "metar"

        result["event_type"] = "surface_observation"

        station_col = "station" if "station" in df.columns else "station_id"
        result["station_id"] = df.get(station_col, "")

        if "tmpf" in df.columns:
            result["temperature_f"] = df["tmpf"]
            result["temperature_c"] = df["tmpf"].apply(
                lambda x: round(_fahrenheit_to_celsius(x), 2) if pd.notna(x) else None
            )
        if "dwpf" in df.columns:
            result["dewpoint_f"] = df["dwpf"]
            result["dewpoint_c"] = df["dwpf"].apply(
                lambda x: round(_fahrenheit_to_celsius(x), 2) if pd.notna(x) else None
            )

        if "sknt" in df.columns:
            result["wind_speed_kt"] = df["sknt"]
            result["wind_speed_mph"] = df["sknt"].apply(
                lambda x: round(_knots_to_mph(x), 2) if pd.notna(x) else None
            )
        if "gust" in df.columns:
            result["gust_speed_kt"] = df["gust"]
            result["gust_speed_mph"] = df["gust"].apply(
                lambda x: round(_knots_to_mph(x), 2) if pd.notna(x) else None
            )

        result["precip_in"] = df.get("p01i", pd.Series(dtype=float))
        result["visibility_mi"] = df.get("vsby", pd.Series(dtype=float))
        result["altimeter_inhg"] = df.get("alti", pd.Series(dtype=float))
        result["relative_humidity"] = df.get("relh", pd.Series(dtype=float))

        if "drct" in df.columns:
            result["wind_direction_deg"] = pd.to_numeric(df["drct"], errors="coerce")

        wind_mph = result.get("wind_speed_mph", pd.Series(dtype=float))
        gust_mph = result.get("gust_speed_mph", pd.Series(dtype=float))
        max_wind = pd.concat([wind_mph, gust_mph], axis=1).max(axis=1)
        result["magnitude"] = max_wind.fillna(0)
        result["magnitude_type"] = "wind_mph"

        lat_col = "station_lat" if "station_lat" in df.columns else "lat"
        lon_col = "station_lon" if "station_lon" in df.columns else "lon"

        if lat_col in df.columns and lon_col in df.columns:
            result["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
            result["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
        else:
            stn_col = "station" if "station" in df.columns else "station_id"
            result["lat"] = df[stn_col].map(
                lambda s: ALL_STATIONS.get(s, (None, None))[0]
            )
            result["lon"] = df[stn_col].map(
                lambda s: ALL_STATIONS.get(s, (None, None))[1]
            )

        h3_res7: list[str | None] = []
        h3_res9: list[str | None] = []
        for _, row in result.iterrows():
            lat, lon = row.get("lat"), row.get("lon")
            if pd.notna(lat) and pd.notna(lon):
                try:
                    h3_res7.append(h3.latlng_to_cell(float(lat), float(lon), 7))
                    h3_res9.append(h3.latlng_to_cell(float(lat), float(lon), 9))
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
        """Insert METAR records into the weather_events table."""
        if df.empty:
            return 0

        records: list[dict] = []
        for _, row in df.iterrows():
            records.append({
                "event_time": row["event_time"],
                "source": row["source"],
                "event_type": row["event_type"],
                "magnitude": float(row["magnitude"]) if pd.notna(row.get("magnitude")) else None,
                "magnitude_type": row.get("magnitude_type"),
                "h3_index_res7": row.get("h3_index_res7"),
                "h3_index_res9": row.get("h3_index_res9"),
                "station_id": str(row.get("station_id", "")),
                "temperature_f": float(row["temperature_f"]) if pd.notna(row.get("temperature_f")) else None,
                "temperature_c": float(row["temperature_c"]) if pd.notna(row.get("temperature_c")) else None,
                "wind_speed_mph": float(row["wind_speed_mph"]) if pd.notna(row.get("wind_speed_mph")) else None,
                "gust_speed_mph": float(row["gust_speed_mph"]) if pd.notna(row.get("gust_speed_mph")) else None,
                "wind_direction_deg": (
                    float(row["wind_direction_deg"])
                    if pd.notna(row.get("wind_direction_deg"))
                    else None
                ),
                "precip_in": float(row["precip_in"]) if pd.notna(row.get("precip_in")) else None,
                "visibility_mi": float(row["visibility_mi"]) if pd.notna(row.get("visibility_mi")) else None,
                "relative_humidity": (
                    float(row["relative_humidity"])
                    if pd.notna(row.get("relative_humidity"))
                    else None
                ),
            })

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
            stmt = text(f"INSERT INTO weather_events ({cols}) VALUES {values_str}")
            await session.execute(stmt, params)
            total_inserted += len(batch)

        await session.commit()
        logger.info("metar.loaded", records=total_inserted)
        return total_inserted

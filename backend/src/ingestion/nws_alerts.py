"""NWS Weather Alerts ingestor.

Fetches active weather alerts from the National Weather Service API
(api.weather.gov) in real-time. No authentication required.
Alerts are returned as GeoJSON features with severity, certainty,
event type, and polygon/point geometry.
"""

from datetime import date, datetime, timezone

import h3
import httpx
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseIngestor, ValidationResult

logger = structlog.get_logger(__name__)

NWS_BASE_URL = "https://api.weather.gov/alerts/active"

NWS_SEVERITY_MAP: dict[str, float] = {
    "Extreme": 100.0,
    "Severe": 75.0,
    "Moderate": 50.0,
    "Minor": 25.0,
    "Unknown": 10.0,
}

VALID_EVENT_TYPES = frozenset([
    "Tornado Warning",
    "Tornado Watch",
    "Severe Thunderstorm Warning",
    "Severe Thunderstorm Watch",
    "Flash Flood Warning",
    "Flash Flood Watch",
    "Flood Warning",
    "Flood Watch",
    "Flood Advisory",
    "Winter Storm Warning",
    "Winter Storm Watch",
    "Winter Weather Advisory",
    "Blizzard Warning",
    "Ice Storm Warning",
    "Wind Advisory",
    "High Wind Warning",
    "Extreme Wind Warning",
    "Hurricane Warning",
    "Hurricane Watch",
    "Tropical Storm Warning",
    "Tropical Storm Watch",
    "Storm Surge Warning",
    "Storm Surge Watch",
    "Excessive Heat Warning",
    "Excessive Heat Watch",
    "Heat Advisory",
    "Wind Chill Warning",
    "Wind Chill Watch",
    "Wind Chill Advisory",
    "Fire Weather Watch",
    "Red Flag Warning",
    "Dense Fog Advisory",
    "Freeze Warning",
    "Frost Advisory",
    "Coastal Flood Warning",
    "Coastal Flood Watch",
    "Coastal Flood Advisory",
    "Rip Current Statement",
    "Tsunami Warning",
    "Tsunami Watch",
    "Special Weather Statement",
    "Severe Weather Statement",
])

STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def _compute_centroid(geometry: dict | None) -> tuple[float | None, float | None]:
    """Extract centroid lat/lon from a GeoJSON geometry object."""
    if not geometry:
        return None, None

    geo_type = geometry.get("type", "")
    coords = geometry.get("coordinates")

    if not coords:
        return None, None

    try:
        if geo_type == "Point":
            lon, lat = coords[0], coords[1]
            return lat, lon
        elif geo_type == "Polygon":
            ring = coords[0]
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            return np.mean(lats), np.mean(lons)
        elif geo_type == "MultiPolygon":
            all_lats: list[float] = []
            all_lons: list[float] = []
            for polygon in coords:
                ring = polygon[0]
                all_lons.extend(p[0] for p in ring)
                all_lats.extend(p[1] for p in ring)
            return np.mean(all_lats), np.mean(all_lons)
    except (IndexError, TypeError, ValueError):
        pass

    return None, None


class NwsAlertsIngestor(BaseIngestor):
    """Ingestor for NWS active weather alerts via api.weather.gov."""

    source_name = "nws_alerts"

    def __init__(self, target_states: list[str] | None = None, timeout: float = 30.0):
        self.target_states = target_states or ["TX", "CA", "FL"]
        self.timeout = timeout

    async def fetch(
        self,
        start_date: date,
        end_date: date,
        region_code: str | None = None,
    ) -> pd.DataFrame:
        """Fetch active alerts from NWS API for target states."""
        states_to_query = [region_code] if region_code else self.target_states
        all_features: list[dict] = []

        headers = {
            "User-Agent": "(outage-prediction-system, contact@example.com)",
            "Accept": "application/geo+json",
        }

        async with httpx.AsyncClient(
            headers=headers, timeout=self.timeout, follow_redirects=True
        ) as client:
            for state in states_to_query:
                if state not in STATE_CODES:
                    logger.warning("nws_alerts.invalid_state", state=state)
                    continue

                try:
                    url = f"{NWS_BASE_URL}?area={state}"
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    features = data.get("features", [])
                    logger.info(
                        "nws_alerts.fetched_state",
                        state=state,
                        alerts=len(features),
                    )
                    all_features.extend(features)
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "nws_alerts.http_error",
                        state=state,
                        status=e.response.status_code,
                    )
                except httpx.RequestError as e:
                    logger.error("nws_alerts.request_error", state=state, error=str(e))

        if not all_features:
            return pd.DataFrame()

        rows: list[dict] = []
        for feature in all_features:
            props = feature.get("properties", {})
            geometry = feature.get("geometry")
            lat, lon = _compute_centroid(geometry)

            rows.append({
                "alert_id": props.get("id", ""),
                "event_type": props.get("event", ""),
                "severity": props.get("severity", "Unknown"),
                "certainty": props.get("certainty", "Unknown"),
                "urgency": props.get("urgency", "Unknown"),
                "headline": props.get("headline", ""),
                "description": props.get("description", ""),
                "instruction": props.get("instruction", ""),
                "onset": props.get("onset"),
                "expires": props.get("expires"),
                "effective": props.get("effective"),
                "sender_name": props.get("senderName", ""),
                "area_desc": props.get("areaDesc", ""),
                "affected_zones": ",".join(props.get("affectedZones", [])),
                "lat": lat,
                "lon": lon,
                "geometry_type": geometry.get("type") if geometry else None,
            })

        df = pd.DataFrame(rows)

        for col in ["onset", "expires", "effective"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        if "onset" in df.columns and df["onset"].notna().any():
            mask = (df["onset"].dt.date >= start_date) & (
                df["onset"].dt.date <= end_date
            )
            df = df.loc[mask | df["onset"].isna()]

        return df.reset_index(drop=True)

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate alert records for required fields and valid event types."""
        total = len(df)
        errors: list[str] = []
        warnings: list[str] = []

        required_cols = ["event_type", "severity"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
                return df.head(0), ValidationResult(
                    valid=False, total_records=total,
                    valid_records=0, invalid_records=total, errors=errors,
                )

        valid_mask = df["event_type"].notna() & (df["event_type"].str.len() > 0)
        valid_mask = valid_mask & df["severity"].notna()

        known_events = df["event_type"].isin(VALID_EVENT_TYPES)
        unknown_count = (~known_events & valid_mask).sum()
        if unknown_count > 0:
            unknown_types = df.loc[~known_events & valid_mask, "event_type"].unique()[:10]
            warnings.append(
                f"{unknown_count} alerts with unrecognized event types: "
                f"{', '.join(unknown_types)}"
            )

        no_geo = valid_mask & (df["lat"].isna() | df["lon"].isna())
        if no_geo.sum() > 0:
            warnings.append(f"{no_geo.sum()} alerts missing geometry data")

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
        """Map NWS alerts to weather_events schema."""
        result = pd.DataFrame()

        result["event_time"] = df.get("onset", df.get("effective"))
        if result["event_time"].isna().all() and "effective" in df.columns:
            result["event_time"] = df["effective"]

        result["source"] = "nws"
        result["event_type"] = df["event_type"]
        result["magnitude"] = df["severity"].map(NWS_SEVERITY_MAP).fillna(10.0)
        result["magnitude_type"] = "nws_severity"

        result["severity"] = df["severity"]
        result["certainty"] = df["certainty"]
        result["urgency"] = df["urgency"]
        result["headline"] = df["headline"]
        result["narrative"] = df["description"].fillna("").str[:2000]
        result["area_desc"] = df["area_desc"]

        result["lat"] = df["lat"]
        result["lon"] = df["lon"]

        h3_res7: list[str | None] = []
        h3_res9: list[str | None] = []
        for _, row in df.iterrows():
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

        result["expires"] = df.get("expires")
        result["alert_id"] = df.get("alert_id")

        return result

    async def load(self, df: pd.DataFrame, session: AsyncSession) -> int:
        """Insert alert records into weather_events table."""
        if df.empty:
            return 0

        records: list[dict] = []
        for _, row in df.iterrows():
            records.append({
                "event_time": row["event_time"],
                "source": row["source"],
                "event_type": row["event_type"],
                "magnitude": float(row["magnitude"]) if pd.notna(row["magnitude"]) else None,
                "magnitude_type": row.get("magnitude_type"),
                "h3_index_res7": row.get("h3_index_res7"),
                "h3_index_res9": row.get("h3_index_res9"),
                "narrative": str(row.get("narrative", ""))[:2000],
                "headline": str(row.get("headline", ""))[:500],
                "severity": row.get("severity"),
                "certainty": row.get("certainty"),
                "urgency": row.get("urgency"),
                "area_desc": str(row.get("area_desc", ""))[:1000],
                "alert_id": row.get("alert_id"),
                "expires": row.get("expires"),
            })

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
                f"INSERT INTO weather_events ({cols}) VALUES {values_str} "
                f"ON CONFLICT (alert_id) DO NOTHING"
            )
            await session.execute(stmt, params)
            total_inserted += len(batch)

        await session.commit()
        logger.info("nws_alerts.loaded", records=total_inserted)
        return total_inserted

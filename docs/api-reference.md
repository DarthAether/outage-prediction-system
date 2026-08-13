# API Reference

> **Prototype-document status:** This reference includes target and experimental endpoints that are not all backed by a loaded trained model or production services. Its sample model metadata and historical metrics are illustrative unless they match the verified results table in the root README. Treat the API as a research scaffold, not an operational outage service.

Reference for the prototype API surface: risk predictions, alerts, historical queries, health checks, and administrative configuration.

---

## Base URL

```
http://localhost:8000/api/v1
```

All endpoints are prefixed with `/api/v1` unless otherwise noted.

---

## Authentication

All API requests require an API key passed via the `X-API-Key` header.

```
X-API-Key: your-api-key-here
```

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | API key issued by the system administrator. Keys are scoped to read-only or read-write access. |

Unauthenticated requests receive a `401 Unauthorized` response. Requests with insufficient permissions receive `403 Forbidden`.

---

## Rate Limiting

The API enforces per-client-IP rate limiting using a token bucket algorithm:

- **Sustained rate**: 20 requests/second
- **Burst capacity**: 50 requests

When the rate limit is exceeded, the API returns `429 Too Many Requests` with a `Retry-After` header indicating the number of seconds to wait.

---

## Predictions

### POST /predictions/realtime

Run a real-time prediction for a single H3 cell. Computes features, runs the ensemble, estimates uncertainty, and classifies the risk level. If severity is YELLOW or above, an alert is created automatically.

**Request Body**

```json
{
  "h3_cell": "872830828ffffff",
  "region": "TX",
  "features_override": {
    "weather_count_24h": 5.0,
    "current_load_mw": 45000.0
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `h3_cell` | string | Yes | H3 cell index at resolution 7. |
| `region` | string | Yes | Region code (e.g., `TX`, `FL`, `CA`). |
| `features_override` | object | No | Optional feature overrides for what-if analysis. Keys are feature names, values are floats. |

**Response Body** (`200 OK`)

```json
{
  "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "h3_cell": "872830828ffffff",
  "region": "TX",
  "risk_probability": 0.72,
  "uncertainty": {
    "lower": 0.58,
    "upper": 0.86,
    "aleatoric": 0.05,
    "epistemic": 0.09,
    "confidence_level": 0.90
  },
  "risk_level": "ORANGE",
  "model_version": "v1.0.0",
  "top_features": [
    {"weather_count_24h": 0.32},
    {"neighbor_outage_max": 0.18},
    {"reserve_margin_pct": 0.14}
  ],
  "computed_at": "2025-03-15T14:30:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction_id` | string | Unique identifier for this prediction. |
| `h3_cell` | string | The H3 cell that was evaluated. |
| `region` | string | Region code. |
| `risk_probability` | float | Calibrated outage probability in [0, 1]. |
| `uncertainty` | object | Uncertainty decomposition with 90% confidence interval bounds, aleatoric (data noise), and epistemic (model structure) components. |
| `risk_level` | string | One of `GREEN`, `YELLOW`, `ORANGE`, `RED`. |
| `model_version` | string | Version identifier of the ensemble that produced this prediction. |
| `top_features` | array or null | Top contributing features with SHAP-based importance scores. |
| `computed_at` | string (ISO 8601) | Timestamp when the prediction was computed. |

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Prediction computed successfully. |
| 422 | Validation error (missing or malformed fields). |
| 503 | Model not loaded or service degraded. |

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/predictions/realtime \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "h3_cell": "872830828ffffff",
    "region": "TX"
  }'
```

---

### POST /predictions/batch

Submit a batch prediction job for an entire region or a specific set of H3 cells. The job is queued and processed asynchronously by the worker service.

**Request Body**

```json
{
  "region": "TX",
  "h3_cells": ["872830828ffffff", "87283082affffff"],
  "timestamp": "2025-03-15T14:00:00Z"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `region` | string | Yes | Region code. |
| `h3_cells` | array of strings | No | Specific H3 cells to process. If `null`, the entire region is processed. |
| `timestamp` | string (ISO 8601) | No | Override the prediction reference time. Defaults to current time. |

**Response Body** (`200 OK`)

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "queued",
  "results": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | UUID for tracking the batch job. |
| `status` | string | One of `queued`, `running`, `completed`, `failed`. |
| `results` | array or null | Array of `PredictionResult` objects when `status` is `completed`. |

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Job accepted and queued. |
| 422 | Validation error. |

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/predictions/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "region": "TX",
    "h3_cells": null,
    "timestamp": "2025-03-15T14:00:00Z"
  }'
```

---

### GET /predictions/history

Retrieve historical predictions filtered by region, H3 cell, and time range.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `region` | string | No | `TX` | Region code filter. |
| `h3_cell` | string | No | - | Specific H3 cell filter. |
| `start_time` | string (ISO 8601) | No | - | Start of time range. |
| `end_time` | string (ISO 8601) | No | - | End of time range. |
| `limit` | integer | No | 100 | Maximum results (1-5000). |

**Response Body** (`200 OK`)

```json
[
  {
    "prediction_id": "12345",
    "h3_cell": "872830828ffffff",
    "region": "TX",
    "risk_probability": 0.45,
    "uncertainty": {
      "lower": 0.30,
      "upper": 0.60,
      "aleatoric": 0.0,
      "epistemic": 0.15,
      "confidence_level": 0.90
    },
    "risk_level": "YELLOW",
    "model_version": "v1.0.0",
    "top_features": null,
    "computed_at": "2025-03-14T12:00:00Z"
  }
]
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Results returned (may be empty array). |

**Example**

```bash
curl "http://localhost:8000/api/v1/predictions/history?region=TX&limit=50&start_time=2025-03-01T00:00:00Z&end_time=2025-03-15T00:00:00Z" \
  -H "X-API-Key: your-api-key-here"
```

---

## Alerts

### GET /alerts/active

Return all unacknowledged, non-expired alerts.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `region` | string | No | - | Filter by region code. |
| `severity` | string | No | - | Filter by severity level (`YELLOW`, `ORANGE`, `RED`). |

**Response Body** (`200 OK`)

```json
[
  {
    "id": 42,
    "severity": "ORANGE",
    "region_code": "TX",
    "h3_cell": "872830828ffffff",
    "risk_probability": 0.72,
    "uncertainty_range": 0.28,
    "description": "High outage risk detected in Harris County area. Thunderstorm wind events with sustained intensity over the past 6 hours.",
    "recommended_actions": [
      "Pre-position line crews in affected area",
      "Notify mutual aid partners",
      "Verify backup generation at critical facilities"
    ],
    "created_at": "2025-03-15T14:30:00Z",
    "expires_at": "2025-03-16T14:30:00Z",
    "acknowledged": false
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Alert identifier. |
| `severity` | string | Alert severity: `YELLOW`, `ORANGE`, or `RED`. |
| `region_code` | string | Region where the alert was generated. |
| `h3_cell` | string | H3 cell with elevated risk. |
| `risk_probability` | float | The risk probability that triggered this alert. |
| `uncertainty_range` | float | Width of the confidence interval at the time of alert creation. |
| `description` | string | Human-readable description of the risk conditions. |
| `recommended_actions` | array of strings | Suggested operational responses. |
| `created_at` | string (ISO 8601) | When the alert was created. |
| `expires_at` | string (ISO 8601) or null | When the alert expires if not acknowledged. |
| `acknowledged` | boolean | Whether the alert has been acknowledged. |

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Active alerts returned (may be empty array). |

**Example**

```bash
curl "http://localhost:8000/api/v1/alerts/active?region=TX&severity=RED" \
  -H "X-API-Key: your-api-key-here"
```

---

### POST /alerts/{id}/acknowledge

Acknowledge an alert by its ID. Acknowledged alerts are excluded from future `/alerts/active` responses.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | integer | Alert ID to acknowledge. |

**Request Body**

```json
{
  "acknowledged_by": "operator.jsmith"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `acknowledged_by` | string | Yes | Identifier of the person or system acknowledging the alert. |

**Response Body** (`200 OK`)

Returns the updated `AlertResponse` object with `acknowledged: true`.

```json
{
  "id": 42,
  "severity": "ORANGE",
  "region_code": "TX",
  "h3_cell": "872830828ffffff",
  "risk_probability": 0.72,
  "uncertainty_range": 0.28,
  "description": "High outage risk detected in Harris County area.",
  "recommended_actions": ["Pre-position line crews in affected area"],
  "created_at": "2025-03-15T14:30:00Z",
  "expires_at": "2025-03-16T14:30:00Z",
  "acknowledged": true
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Alert acknowledged. |
| 404 | Alert not found. |
| 422 | Validation error. |

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/alerts/42/acknowledge \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"acknowledged_by": "operator.jsmith"}'
```

---

### WS /alerts/stream

WebSocket endpoint that pushes new alerts in real-time. Clients connect and receive JSON-serialized `AlertResponse` objects whenever a new alert is created.

**Connection**

```
ws://localhost:8000/api/v1/alerts/stream
```

**Message Format**

Each message is a JSON string containing an `AlertResponse` object:

```json
{
  "id": 43,
  "severity": "RED",
  "region_code": "TX",
  "h3_cell": "872830829ffffff",
  "risk_probability": 0.91,
  "uncertainty_range": 0.12,
  "description": "Critical outage risk. Hurricane-force winds approaching.",
  "recommended_actions": [
    "Activate emergency operations center",
    "Issue public safety notifications"
  ],
  "created_at": "2025-03-15T15:00:00Z",
  "expires_at": null,
  "acknowledged": false
}
```

**Notes**
- The connection uses an in-process asyncio broadcast queue. In production, this is backed by Redis Pub/Sub for multi-instance support.
- The server sends messages to the client only; it does not expect client-to-server messages after the initial handshake.

**Example (websocat)**

```bash
websocat ws://localhost:8000/api/v1/alerts/stream
```

---

## Historical

### GET /historical/outages

Return historical outage observations aggregated by county and time.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `region` | string | No | - | State FIPS code filter. |
| `county_fips` | string | No | - | County FIPS code filter. |
| `start_time` | string (ISO 8601) | No | 30 days ago | Start of time range. |
| `end_time` | string (ISO 8601) | No | Now | End of time range. |
| `limit` | integer | No | 500 | Maximum results (1-10,000). |

**Response Body** (`200 OK`)

```json
[
  {
    "id": 1001,
    "observed_at": "2025-03-14T18:00:00Z",
    "county_fips": "48201",
    "state_fips": "48",
    "customers_out": 12500,
    "total_customers": 250000,
    "outage_fraction": 0.05,
    "h3_cell": "872830828ffffff"
  }
]
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Results returned. |

**Example**

```bash
curl "http://localhost:8000/api/v1/historical/outages?region=48&county_fips=48201&limit=100" \
  -H "X-API-Key: your-api-key-here"
```

---

### GET /historical/weather

Return historical weather events filtered by region and type.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `region` | string | No | - | State FIPS code filter. |
| `event_type` | string | No | - | Weather event type (e.g., `Thunderstorm Wind`, `Ice Storm`, `Tornado`). |
| `start_time` | string (ISO 8601) | No | 30 days ago | Start of time range. |
| `end_time` | string (ISO 8601) | No | Now | End of time range. |
| `limit` | integer | No | 500 | Maximum results (1-10,000). |

**Response Body** (`200 OK`)

```json
[
  {
    "id": 5001,
    "event_time": "2025-03-14T16:30:00Z",
    "event_type": "Thunderstorm Wind",
    "magnitude": 65.0,
    "h3_cell": "872830828ffffff",
    "state_fips": "48",
    "county_fips": "48201",
    "damage_property": 50000.0,
    "injuries": 0,
    "deaths": 0
  }
]
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Results returned. |

**Example**

```bash
curl "http://localhost:8000/api/v1/historical/weather?region=48&event_type=Tornado&limit=200" \
  -H "X-API-Key: your-api-key-here"
```

---

### GET /historical/model-performance

Return model metrics over time from the model registry. Useful for monitoring model drift and comparing versions.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `region` | string | No | - | Region code filter. |
| `limit` | integer | No | 50 | Maximum results (1-200). |

**Response Body** (`200 OK`)

```json
[
  {
    "model_name": "xgboost",
    "version": "v1.0.0",
    "region_code": "TX",
    "metrics": {
      "auc_roc": 0.967,
      "f1": 0.471,
      "brier_score": 0.139,
      "ece": 0.360,
      "scope": "xgboost; physics-informed synthetic outage targets"
    },
    "is_active": true,
    "promoted_at": "2025-03-10T08:00:00Z",
    "created_at": "2025-03-10T07:45:00Z"
  }
]
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Results returned. |

**Example**

```bash
curl "http://localhost:8000/api/v1/historical/model-performance?region=TX" \
  -H "X-API-Key: your-api-key-here"
```

---

## Health

### GET /health

Comprehensive system health check. Reports database connectivity, Redis connectivity, model loading status, and uptime.

**Response Body** (`200 OK`)

```json
{
  "status": "degraded",
  "db_connected": true,
  "redis_connected": false,
  "model_loaded": false,
  "active_models": [],
  "uptime_seconds": 86423.15
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy` if DB is connected and model is loaded, `degraded` otherwise. |
| `db_connected` | boolean | PostgreSQL connectivity. |
| `redis_connected` | boolean | Redis connectivity. |
| `model_loaded` | boolean | Whether the ensemble model is loaded in memory. |
| `active_models` | array of strings | Names of currently active model components. |
| `uptime_seconds` | float | Seconds since the API server started. |

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Health status returned (may indicate degraded state). |

**Example**

```bash
curl http://localhost:8000/api/v1/health \
  -H "X-API-Key: your-api-key-here"
```

---

### GET /health/models

Return details about currently loaded models. The committed application scaffold does not load trained artifacts at startup, so the default response is:

**Response Body** (`200 OK`)

```json
{
  "loaded": false,
  "models": []
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Model information returned. |

**Example**

```bash
curl http://localhost:8000/api/v1/health/models \
  -H "X-API-Key: your-api-key-here"
```

---

## Admin

### PUT /admin/thresholds

Update risk classification thresholds for a region. Thresholds define the boundaries between GREEN, YELLOW, ORANGE, and RED risk levels.

**Request Body**

```json
{
  "region": "TX",
  "green_max": 0.25,
  "yellow_max": 0.55,
  "orange_max": 0.80
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `region` | string | Yes | - | Region code. |
| `green_max` | float | No | 0.25 | Upper bound of GREEN risk level. |
| `yellow_max` | float | No | 0.55 | Upper bound of YELLOW risk level. |
| `orange_max` | float | No | 0.80 | Upper bound of ORANGE risk level. Probability above this is RED. |

Thresholds must satisfy: `green_max < yellow_max < orange_max`.

**Response Body** (`200 OK`)

```json
{
  "region": "TX",
  "thresholds": {
    "GREEN": 0.25,
    "YELLOW": 0.55,
    "ORANGE": 0.80
  },
  "updated_at": "2025-03-15T14:30:00Z"
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Thresholds updated. |
| 422 | Invalid threshold ordering. |

**Example**

```bash
curl -X PUT http://localhost:8000/api/v1/admin/thresholds \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "region": "TX",
    "green_max": 0.20,
    "yellow_max": 0.50,
    "orange_max": 0.75
  }'
```

---

### POST /admin/models/promote

Promote a model version to active status. Deactivates any currently active model with the same name and region, then marks the specified version as active.

**Request Body**

```json
{
  "model_name": "xgboost",
  "version": "v1.1.0",
  "region": "TX"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_name` | string | Yes | Model name (e.g., `xgboost`, `lightgbm`, `lstm`). |
| `version` | string | Yes | Version string to promote. |
| `region` | string | No | Region scope. If null, promotes globally. |

**Response Body** (`200 OK`)

```json
{
  "model_name": "xgboost",
  "version": "v1.1.0",
  "region": "TX",
  "promoted_at": "2025-03-15T14:30:00Z"
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Model promoted. |
| 404 | Model name/version combination not found. |

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/admin/models/promote \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "model_name": "xgboost",
    "version": "v1.1.0",
    "region": "TX"
  }'
```

---

### GET /admin/config

Return the active runtime configuration, including active regions, region-specific risk thresholds, and model loading status.

**Response Body** (`200 OK`)

```json
{
  "active_regions": ["TX"],
  "region_configs": {
    "TX": {
      "risk_thresholds": {
        "GREEN": 0.25,
        "YELLOW": 0.55,
        "ORANGE": 0.80
      }
    }
  },
  "model_loaded": true,
  "active_model_names": ["xgboost", "lightgbm", "lstm"]
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Configuration returned. |

**Example**

```bash
curl http://localhost:8000/api/v1/admin/config \
  -H "X-API-Key: your-api-key-here"
```

---

## Error Responses

All error responses follow a consistent format:

```json
{
  "detail": "Human-readable error description",
  "code": "OPTIONAL_ERROR_CODE"
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| 400 | Bad request. |
| 401 | Missing or invalid API key. |
| 403 | Insufficient permissions. |
| 404 | Resource not found. |
| 422 | Validation error (Pydantic). Includes field-level detail. |
| 429 | Rate limit exceeded. Check `Retry-After` header. |
| 500 | Internal server error. |
| 503 | Service unavailable (model not loaded, DB unreachable). |

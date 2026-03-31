"""
GridShield AI - Demo API
Lightweight demo version that runs without ML dependencies (XGBoost, LightGBM, PyTorch).
All predictions are simulated with realistic distributions.
"""

import random
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GridShield AI - Demo",
    description="Power outage prediction system (demo mode — simulated predictions)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


@app.middleware("http")
async def add_demo_header(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Demo-Mode"] = "true"
    return response


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    h3_cell: str = Field(..., description="H3 cell index (resolution 7-9)")
    region: str = Field("TX", description="Region code (TX, CA, FL)")
    horizon_hours: int = Field(24, ge=1, le=168, description="Forecast horizon in hours")


class BatchPredictionRequest(BaseModel):
    predictions: list[PredictionRequest]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIONS = {
    "TX": {
        "name": "Texas (ERCOT)",
        "grid_operator": "ERCOT",
        "timezone": "America/Chicago",
        "h3_resolution": 7,
        "coverage_cells": 14_832,
        "model_version": "v2.4.1-ensemble",
    },
    "CA": {
        "name": "California (CAISO)",
        "grid_operator": "CAISO",
        "timezone": "America/Los_Angeles",
        "h3_resolution": 7,
        "coverage_cells": 11_247,
        "model_version": "v2.4.1-ensemble",
    },
    "FL": {
        "name": "Florida (FPL/Duke)",
        "grid_operator": "FPL",
        "timezone": "America/New_York",
        "h3_resolution": 7,
        "coverage_cells": 9_651,
        "model_version": "v2.4.1-ensemble",
    },
}

FEATURE_NAMES = [
    "wind_speed_10m_ms",
    "wind_gust_max_ms",
    "temperature_2m_c",
    "precipitation_mm",
    "lightning_density_km2",
    "soil_moisture_pct",
    "tree_canopy_density",
    "equipment_age_years",
    "transformer_load_pct",
    "line_voltage_deviation",
    "humidity_relative_pct",
    "pressure_msl_hpa",
    "vegetation_ndvi",
    "population_density_km2",
    "historical_outage_count",
    "distance_to_coast_km",
    "elevation_m",
    "ice_accretion_mm",
    "snow_depth_cm",
    "demand_forecast_mw",
]

FEATURE_IMPORTANCES = [
    0.142, 0.118, 0.097, 0.089, 0.081, 0.072, 0.064, 0.058, 0.053, 0.047,
    0.039, 0.033, 0.028, 0.024, 0.019, 0.014, 0.010, 0.007, 0.004, 0.001,
]

SAMPLE_H3_CELLS = [
    "872a1008fffffff", "872a1009fffffff", "872a100afffffff",
    "872a100bfffffff", "872a100cfffffff",
]

ALERT_DESCRIPTIONS = [
    "Elevated outage risk detected in coastal zone due to approaching tropical storm",
    "High wind speeds (>65 mph) forecast — overhead line exposure risk elevated",
    "Transformer overload warning in downtown substation cluster",
    "Ice accretion exceeding 0.5 in forecast on northern distribution lines",
    "Lightning density surge detected — wildfire ignition risk elevated",
    "Extreme heat advisory — demand-driven brownout probability rising",
    "Vegetation encroachment alert — trimming overdue in sector 14-B",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_level(prob: float) -> str:
    if prob < 0.25:
        return "GREEN"
    if prob < 0.50:
        return "YELLOW"
    if prob < 0.75:
        return "ORANGE"
    return "RED"


def _generate_prediction(req: PredictionRequest) -> dict:
    random.seed(hash(req.h3_cell + req.region) % 2**32)
    base = random.uniform(0.10, 0.95)
    # Re-seed with time component so repeated calls aren't identical
    random.seed(None)
    noise = random.uniform(-0.05, 0.05)
    prob = max(0.01, min(0.99, base + noise))
    level = _risk_level(prob)
    uncertainty = random.uniform(0.03, 0.12)

    top_k = 5
    indices = random.sample(range(len(FEATURE_NAMES)), top_k)
    top_features = [
        {"name": FEATURE_NAMES[i], "importance": round(random.uniform(0.05, 0.25), 4)}
        for i in sorted(indices, key=lambda i: -FEATURE_IMPORTANCES[i])
    ]

    now = datetime.now(timezone.utc)
    return {
        "prediction_id": str(uuid.uuid4()),
        "h3_cell": req.h3_cell,
        "region": req.region,
        "horizon_hours": req.horizon_hours,
        "risk_probability": round(prob, 4),
        "risk_level": level,
        "uncertainty_lower": round(max(0, prob - uncertainty), 4),
        "uncertainty_upper": round(min(1, prob + uncertainty), 4),
        "top_features": top_features,
        "model_version": REGIONS.get(req.region, REGIONS["TX"])["model_version"],
        "timestamp": now.isoformat(),
        "valid_until": (now + timedelta(hours=req.horizon_hours)).isoformat(),
    }


def _generate_alerts(n: int = 5) -> list[dict]:
    now = datetime.now(timezone.utc)
    severities = ["INFO", "WARNING", "CRITICAL", "WARNING", "INFO"]
    alerts = []
    for i in range(n):
        ts = now - timedelta(minutes=random.randint(5, 600))
        alerts.append({
            "alert_id": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "severity": severities[i % len(severities)],
            "region": random.choice(list(REGIONS.keys())),
            "h3_cell": random.choice(SAMPLE_H3_CELLS),
            "description": random.choice(ALERT_DESCRIPTIONS),
            "acknowledged": random.choice([True, False]),
        })
    return sorted(alerts, key=lambda a: a["timestamp"], reverse=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "GridShield AI",
        "version": "1.0.0",
        "status": "running",
        "mode": "demo",
        "docs_url": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "mode": "demo",
    }


@app.get("/api/v1/regions")
async def list_regions():
    return {"regions": REGIONS}


@app.post("/api/v1/predict")
async def predict(req: PredictionRequest):
    if req.region not in REGIONS:
        raise HTTPException(status_code=400, detail=f"Unknown region '{req.region}'. Valid: {list(REGIONS.keys())}")
    return _generate_prediction(req)


@app.post("/api/v1/predict/batch")
async def predict_batch(req: BatchPredictionRequest):
    if len(req.predictions) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
    results = [_generate_prediction(p) for p in req.predictions]
    return {"predictions": results, "count": len(results)}


@app.get("/api/v1/alerts")
async def get_alerts():
    return {"alerts": _generate_alerts(5), "total": 5}


@app.get("/api/v1/model/info")
async def model_info():
    return {
        "model_type": "Weighted Ensemble",
        "components": [
            {"name": "XGBoost", "weight": 0.40, "version": "1.7.6"},
            {"name": "LightGBM", "weight": 0.35, "version": "4.1.0"},
            {"name": "LSTM (PyTorch)", "weight": 0.25, "version": "2.1.0"},
        ],
        "feature_count": len(FEATURE_NAMES),
        "training_samples": 2_847_193,
        "metrics": {
            "auc_roc": 0.934,
            "precision_at_80_recall": 0.871,
            "brier_score": 0.062,
            "f1_score": 0.889,
        },
        "last_trained": "2025-12-15T08:30:00Z",
        "mlflow_run_id": "a1b2c3d4e5f6",
    }


@app.get("/api/v1/features/importance")
async def feature_importance():
    features = [
        {"rank": i + 1, "name": FEATURE_NAMES[i], "importance": FEATURE_IMPORTANCES[i]}
        for i in range(len(FEATURE_NAMES))
    ]
    return {"features": features, "model_version": "v2.4.1-ensemble"}

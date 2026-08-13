# System Architecture

> **Design-document status:** This is an aspirational deployment architecture, not the evaluated thesis system. The verified benchmark uses a two-tree XGBoost/LightGBM average with physics-informed synthetic outage targets. Several components below—including live EAGLE-I/ERCOT ingestion, the LSTM stack, a worker process, and production deployment—are incomplete or not validated in this repository. See the root README for the implemented benchmark and current limitations.

This document records the broader real-time platform design explored during the project.

---

## System Context (C4 Level 1)

The system sits at the intersection of multiple external data sources and serves two primary user groups.

### Users

| Actor | Description |
|-------|-------------|
| **Utility Operators** | Dispatch and operations center staff who monitor real-time outage risk maps, receive severity-graded alerts (GREEN/YELLOW/ORANGE/RED), and acknowledge alerts to coordinate field crew deployment. |
| **Researchers / Analysts** | Data scientists and grid reliability engineers who access historical predictions, model performance dashboards, and feature importance analysis for capacity planning and post-event review. |

### External Systems

| System | Integration | Data Provided |
|--------|-------------|---------------|
| **NOAA Climate Data Online (CDO)** | REST API (batch pull) | Historical storm events: event type, magnitude, location, damage, injuries, episode linkage |
| **National Weather Service (NWS)** | REST API (real-time) | Active weather alerts, watches, and warnings with polygons and severity |
| **EAGLE-I (DOE)** | REST API (polling) | Real-time county-level outage observations: customers out, total customers served |
| **ERCOT** | REST API / SFTP | Grid load (MW), capacity, reserve margin, frequency for the Texas Interconnection |
| **EIA (Energy Information Administration)** | REST API | Utility service territories, historical outage reports (Form 861), generation capacity |
| **US Census Bureau** | REST API / bulk download | Population density, median income, housing age, critical facility counts per tract |

---

## Container Diagram (C4 Level 2)

```
+------------------------------------------------------------------+
|                        Outage Prediction System                   |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+    +------------------+    +--------------+  |
|  |  Next.js         |    |  FastAPI          |    |  Worker      |  |
|  |  Frontend        |--->|  API Server       |<---|  Service     |  |
|  |  (port 3000)     |    |  (port 8000)      |    |              |  |
|  +------------------+    +--------+---------+    +------+-------+  |
|                                   |                     |          |
|                          +--------+---------+           |          |
|                          |                  |           |          |
|                  +-------v------+   +-------v-------+   |          |
|                  | PostgreSQL   |   |  Redis        |   |          |
|                  | + TimescaleDB|   |  (port 6379)  |<--+          |
|                  | + PostGIS    |   +---------------+              |
|                  | (port 5432)  |                                   |
|                  +--------------+   +---------------+              |
|                                     |  MLflow       |              |
|                                     |  (port 5000)  |              |
|                                     +---------------+              |
+------------------------------------------------------------------+
```

### Container Responsibilities

#### PostgreSQL + TimescaleDB + PostGIS
- **Role**: Primary data store for all time-series and geospatial data.
- **TimescaleDB** provides hypertable partitioning for weather events, outage observations, and grid load data, enabling fast time-range queries across millions of rows.
- **PostGIS** stores region boundaries as `MULTIPOLYGON` geometries (SRID 4326) and supports spatial queries for H3 cell assignment and neighbor lookups.
- **Tables**: `weather_events`, `outage_observations`, `grid_load`, `infrastructure`, `socioeconomic`, `feature_store`, `predictions`, `alerts`, `model_registry`, `regions`.
- **Port**: 5432

#### Redis
- **Role**: Caching layer, real-time pub/sub backbone, and task broker.
- Feature vectors for hot H3 cells are cached to avoid recomputation during burst inference. Alert broadcasts are distributed via Redis Pub/Sub channels to all connected WebSocket clients. In production deployments, Redis also serves as the Celery/ARQ broker for batch job queuing.
- **Port**: 6379

#### MLflow Tracking Server
- **Role**: Experiment tracking and model artifact storage.
- Every training run logs hyperparameters, evaluation metrics (AUC-ROC, F1, Brier Score, ECE), and model artifacts. The model registry tracks which version is promoted to active for each region. Backend store is PostgreSQL; artifact store is a mounted volume.
- **Port**: 5000

#### FastAPI API Server
- **Role**: Synchronous HTTP and WebSocket gateway for all client interactions.
- Handles real-time single-cell predictions, batch prediction submission, alert management, historical data queries, health checks, and admin configuration endpoints. Loads the ensemble model at startup and holds it in application state. Applies rate limiting (20 req/s sustained, 50 burst) and structured request logging via middleware.
- **Port**: 8000

#### Worker Service
- **Role**: Background processing for batch inference and scheduled data ingestion.
- Runs batch prediction jobs across all H3 cells in active regions on a configurable interval (default: 60 minutes). Executes data ingestion pipelines for NOAA, NWS, EAGLE-I, ERCOT, EIA, and Census sources on independent schedules. Shares the same codebase and Docker image as the API server with a different entrypoint.

#### Next.js Frontend
- **Role**: Interactive dashboard for outage risk monitoring.
- Renders a Mapbox-powered hexagonal risk map with H3 cells colored by risk level. Displays real-time alerts via WebSocket subscription, historical outage timelines, model performance charts, and feature importance breakdowns. Built with React, TypeScript, and Tailwind CSS.
- **Port**: 3000

---

## Data Flow

### Ingestion Pipeline

```
External Sources ──> Validators ──> Transformers ──> Database
```

1. **Source Connectors**: Dedicated ingestion modules (`noaa_storms.py`, `nws_alerts.py`, `eagle_i.py`, `ercot.py`, `eia.py`, `census.py`, `metar.py`) pull data from each external API using source-specific authentication and pagination logic.
2. **Validators** (`validators.py`): Incoming records pass through schema validation, deduplication checks (episode-level for NOAA, timestamp+county for EAGLE-I), and range/type enforcement. Malformed records are logged and rejected.
3. **Transformers**: Raw records are geocoded to H3 cells at resolution 7, FIPS codes are standardized, magnitudes are normalized to common scales, and timestamps are converted to UTC.
4. **Database Write**: Validated, transformed records are bulk-inserted into their respective TimescaleDB hypertables. The scheduler (`scheduler.py`) orchestrates pull frequencies per source.

### Feature Pipeline

```
Raw Data ──> Temporal Features ──> Spatial Features ──> Compound Features ──> Feature Store
```

1. **Temporal Features** (`temporal.py`): Multi-horizon rolling aggregations over weather events and outage history at windows of 1h, 3h, 6h, 12h, 24h, 48h, and 72h. Includes event counts, max/mean magnitude, distinct event types, lagged outage fractions, trend slopes, grid load metrics, and cyclical time encodings (hour, day-of-week, month).
2. **Spatial Features** (`spatial.py`): H3 k-ring neighbor aggregations for weather severity and outage spread. Infrastructure features: transmission/distribution line density, substation count, line age, and vegetation density per cell.
3. **Compound Event Features** (`compound_events.py`): Interaction terms that capture multi-hazard scenarios (e.g., wind + ice co-occurrence) and cascading event sequences within a temporal window.
4. **Socioeconomic Features**: Population density, normalized median income, critical facility density, and median housing age from Census data.
5. **Feature Store** (`feature_store.py`): Orchestrates all builders, materializes feature vectors as JSONB in the `feature_store` table, and caches hot vectors in Redis for real-time inference.

### Inference Pipeline

```
Feature Store ──> Ensemble ──> Uncertainty ──> Prediction ──> Alert ──> WebSocket
```

1. **Feature Retrieval**: For a given H3 cell and timestamp, the feature store retrieves or computes the full feature vector. Cached vectors are served from Redis when available.
2. **Ensemble Prediction**: The `OutageEnsemble` routes tabular features to XGBoost and LightGBM, sequential features to the LSTM, and combines outputs through a stacking meta-learner (logistic regression trained on validation-set predictions).
3. **Uncertainty Estimation**: The `UncertaintyEstimator` combines MC Dropout variance (50 stochastic forward passes through the LSTM) with ensemble member disagreement to produce decomposed aleatoric and epistemic uncertainty bounds. Post-hoc isotonic calibration adjusts the final probability.
4. **Risk Classification**: The calibrated probability is mapped to a risk level (GREEN/YELLOW/ORANGE/RED) using region-specific thresholds configurable via the admin API.
5. **Alert Generation**: Predictions at YELLOW severity or above automatically generate an alert record with recommended actions based on severity and weather context.
6. **WebSocket Broadcast**: New alerts are pushed to all connected clients via an in-process asyncio broadcast queue (Redis Pub/Sub in production).

---

## ASCII Architecture Diagram

```
                         +-------------------+
                         |    Utility        |
                         |    Operators      |
                         +--------+----------+
                                  |
                                  v
+----------+            +-------------------+            +----------+
|  NOAA    |            |   Next.js         |            |  NWS     |
|  CDO     |            |   Frontend        |            |  Alerts  |
+----+-----+            |   :3000           |            +----+-----+
     |                  +--------+----------+                 |
     |                           |                            |
     |                  +--------v----------+                 |
     |                  |   FastAPI         |                 |
     +----------------->|   API Server      |<----------------+
     |                  |   :8000           |                 |
     |                  +---+-----+-----+--+                 |
     |                      |     |     |                    |
+----+-----+          +-----+  +--+--+  +-----+      +------+-----+
| EAGLE-I  |          |       |      |        |      |  ERCOT     |
| (DOE)    +--------->| Pg    |Redis |  MLflow|      |  Grid Load |
+----------+          | +Ts   |:6379 |  :5000 |      +------+-----+
                      | +GIS  |      |        |             |
+----------+          |:5432  +------+--------+      +------+-----+
|  EIA     +--------->|       |                      |  Census    |
+----------+          +---+---+                      +------+-----+
                          ^                                 |
                          |     +-------------------+       |
                          +-----+   Worker          +<------+
                                |   Service         |
                                +-------------------+
```

---

## Key Design Decisions

### H3 Resolution 7 as the Spatial Unit

H3 hexagons at resolution 7 provide an average cell area of approximately 5.16 km², which aligns well with county-subdivision granularity while remaining computationally tractable. Resolution 7 was selected over finer resolutions (8 or 9) because EAGLE-I outage data is reported at the county level -- finer cells would produce duplicate values within the same county. Resolution 7 also keeps the total cell count per region manageable (Texas ~130k cells) for batch inference within a 60-minute cycle. The hexagonal grid avoids edge effects present in rectangular grids and supports efficient k-ring neighbor queries for spatial feature computation.

### Redis over Kafka for Event Streaming

Redis Pub/Sub was chosen over Apache Kafka for alert streaming and feature caching. The system operates with a small number of concurrent WebSocket clients (tens to low hundreds) and does not require persistent message replay or consumer group semantics. Redis provides sub-millisecond latency for both cache reads and pub/sub, and the single Redis instance already serves as the task broker and feature cache. Adding Kafka would introduce a separate JVM-based cluster with significant operational overhead for a throughput profile that Redis handles comfortably. If the system scales to thousands of concurrent consumers or requires guaranteed delivery with replay, Kafka can be introduced as a drop-in replacement for the pub/sub layer.

### Temporal Splits over Random Splits

All train/validation/test splits are strictly temporal: the training set contains only data before the validation period, and the validation set contains only data before the test period. This prevents data leakage from future weather events or outage patterns into the training signal. Random splits in time-series data would allow the model to "see" the temporal neighborhood of test samples during training, inflating evaluation metrics by 5-15% based on empirical testing. Temporal splits produce conservative but honest estimates of production performance, which is critical for a safety-adjacent system.

### Dual Uncertainty Estimation (MC Dropout + Ensemble Disagreement)

Uncertainty quantification uses two complementary sources rather than a single method. MC Dropout (50 stochastic forward passes through the LSTM with dropout active) captures uncertainty in the neural model's learned representations -- it responds to novel sequential patterns that the LSTM has not seen during training. Ensemble disagreement (variance across XGBoost, LightGBM, and LSTM predictions) captures epistemic uncertainty from structural model differences -- when the tree models and the neural model disagree, the system has lower confidence. The decomposition into aleatoric (data noise, from MC Dropout variance) and epistemic (model structure, from ensemble disagreement) uncertainty allows operators to distinguish between "this area is inherently unpredictable" and "the model is uncertain because conditions are unusual." Post-hoc isotonic calibration ensures the final confidence intervals match observed frequencies.

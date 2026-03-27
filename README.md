# Proactive Power Outage Prediction in the Texas Electrical Grid
### An Infrastructure-Aware Machine Learning Framework

A spatiotemporal prediction system for anticipating weather-induced power outages across ERCOT service territories, combining compound weather event modeling with grid topology features over H3 hexagonal spatial indices (resolution 7, ~5.16 km<sup>2</sup> per cell) to produce probabilistic 24-hour outage risk forecasts.

---

## Architecture

```
                        +---------------------+
                        |   NOAA CDO / ISD    |
                        |   EIA Form 861/OE   |
                        +--------+------------+
                                 |
                         ingestion workers
                                 |
                    +------------v------------+
                    |      Apache Kafka       |
                    |   (weather.raw topic)   |
                    +------------+------------+
                                 |
              +------------------+------------------+
              |                                     |
   +----------v----------+              +-----------v-----------+
   |  Feature Pipeline   |              |   TimescaleDB         |
   |  - H3 spatial join  |              |   (hypertable:        |
   |  - lag features     |              |    weather_obs,       |
   |  - rolling stats    |              |    outage_events)     |
   |  - compound indices |              +-----------+-----------+
   +----------+----------+                          |
              |                                     |
              v                                     |
   +---------------------+                         |
   |   MLflow Registry   |                         |
   |   - LightGBM        |                         |
   |   - CatBoost        |<------------------------+
   |   - MC Dropout NN   |       training pipeline
   +----------+----------+
              |
              v
   +---------------------+       +-------------------+
   |   FastAPI Service    +------>+  Redis (cache +   |
   |   /predict           |       |  pub/sub)        |
   |   /explain (SHAP)    |       +-------------------+
   +----------+-----------+
              |
              v
   +---------------------+
   |   Next.js Dashboard  |
   |   - Deck.gl H3 map  |
   |   - Risk heatmaps   |
   |   - SHAP waterfall   |
   +----------------------+
```

## Key Features

**Compound Weather Event Modeling**
Constructs joint severity indices from co-occurring hazards (e.g., ice + wind, heat + humidity) rather than treating weather variables independently. Storm severity is encoded using NOAA Storm Events magnitude bins cross-referenced against grid infrastructure exposure per H3 cell.

**Uncertainty Quantification via MC Dropout**
Inference runs 50 stochastic forward passes (MC Dropout, p=0.15) through the neural network head, yielding calibrated prediction intervals. The output is a full predictive distribution per cell, not a point estimate, enabling risk-stratified dispatch of field crews.

**State-Agnostic, Transferable Framework**
Feature engineering is parameterized by region configuration files (`config/regions/*.yaml`), making the pipeline portable to any U.S. state with NOAA ISD coverage. Texas is the reference implementation; adapting to a new region requires only a config file and a historical outage dataset.

**Real-Time Operational Dashboard**
Next.js frontend renders H3 hexagonal risk maps via Deck.gl with sub-second updates pushed over WebSocket. Each cell is clickable for a SHAP waterfall decomposition of the driving risk factors.

**H3 Spatial Indexing**
All spatial operations use Uber's H3 hierarchical hexagonal grid at resolution 7 (edge length ~1.22 km, cell area ~5.16 km<sup>2</sup>), providing uniform spatial binning that avoids the distortion artifacts of rectangular grids at scale.

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Apache Kafka, custom Python workers |
| Storage | TimescaleDB (PostgreSQL + hypertables), Redis |
| Feature Store | Feast (offline: Parquet, online: Redis) |
| ML Training | LightGBM, CatBoost, PyTorch (MC Dropout head) |
| Experiment Tracking | MLflow |
| Serving | FastAPI, uvicorn, gunicorn |
| Explainability | SHAP (TreeExplainer + DeepExplainer) |
| Frontend | Next.js 14, TypeScript, Deck.gl, Recharts |
| Spatial | H3-py, GeoPandas, PostGIS |
| Orchestration | Docker Compose (dev), Kubernetes (prod) |
| CI/CD | GitHub Actions |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/vijaysivanj/outage-prediction-system.git
cd outage-prediction-system

# Copy environment template and fill in API keys
cp .env.example .env
# Edit .env with your NOAA CDO token, EIA API key, etc.

# Start all services
docker-compose up -d

# Verify services are healthy
curl http://localhost:8000/health       # FastAPI backend
curl http://localhost:3000              # Next.js dashboard
```

The compose stack brings up TimescaleDB, Redis, Kafka + Zookeeper, the FastAPI backend, and the Next.js frontend. MLflow UI is available at `http://localhost:5000`.

## Data Sources

| Source | Description | Granularity |
|---|---|---|
| [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Hourly surface observations (temp, wind, precip, pressure) | Station-level, hourly |
| [NOAA Storm Events](https://www.ncdc.noaa.gov/stormevents/) | Historical severe weather episodes with damage estimates | Event-level |
| [EIA Form OE-417](https://www.eia.gov/electricity/data/disturbance/) | Electric disturbance events (outages) reported to DOE | Event-level |
| [EIA Form 861](https://www.eia.gov/electricity/data/eia861/) | Utility service territory boundaries and customer counts | Annual, utility-level |
| [HIFLD Transmission Lines](https://hifld-geoplatform.opendata.arcgis.com/) | Transmission infrastructure geospatial data | Vector geometry |

## Project Structure

```
outage-prediction-system/
├── backend/
│   ├── src/
│   │   ├── api/              # FastAPI routes and middleware
│   │   ├── db/               # SQLAlchemy models, Alembic migrations
│   │   ├── features/         # Feature engineering pipeline (H3, temporal, compound)
│   │   ├── ingestion/        # Kafka consumers, NOAA/EIA API clients
│   │   ├── ml/               # Model training, MC Dropout inference, SHAP
│   │   ├── services/         # Business logic layer
│   │   └── streaming/        # WebSocket event broadcasting
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── e2e/
│   │   └── load/             # Locust load testing scenarios
│   └── pyproject.toml
├── frontend/
│   └── src/
│       ├── app/              # Next.js App Router pages
│       ├── components/       # Deck.gl map, SHAP charts, risk panels
│       ├── hooks/            # WebSocket, data fetching hooks
│       └── lib/              # H3 utilities, API client
├── config/
│   ├── models/               # Hyperparameter configs per model
│   └── regions/              # Region-specific YAML (stations, grid bounds, thresholds)
├── data/
│   ├── raw/                  # Unprocessed downloads (gitignored)
│   └── processed/            # Feature-engineered datasets
├── docker/
│   └── init-db/              # TimescaleDB bootstrap SQL
├── docs/
│   └── model-cards/          # ML model documentation (per MLflow run)
├── notebooks/                # Exploratory analysis and prototyping
├── paper/
│   ├── figures/
│   └── tables/
├── scripts/                  # Data download, DB seeding, batch inference
├── legacy/                   # Previous iteration (sklearn baseline models)
├── .github/workflows/        # CI/CD pipeline definitions
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Research Contributions

This work makes three primary contributions to the power systems reliability literature:

1. **Compound Weather Index (CWI) formulation.** We introduce a multiplicative severity index that captures the joint effect of co-occurring weather hazards on grid infrastructure. Unlike additive approaches, CWI models the nonlinear interaction between simultaneous ice accretion and sustained wind, or between extreme heat and humidity, which are the dominant failure modes observed in Texas outage data from 2018--2024.

2. **Calibrated uncertainty quantification for operational outage prediction.** By combining gradient-boosted tree ensembles with an MC Dropout neural network head (50 forward passes, dropout p=0.15), the system produces well-calibrated predictive intervals (coverage probability within 2% of nominal across all risk quantiles on the held-out 2023--2024 test set). This enables risk-tiered resource allocation rather than binary outage/no-outage classification.

3. **Region-portable feature engineering framework.** The entire pipeline -- from spatial binning to feature construction to model retraining -- is parameterized by declarative region configuration files, requiring zero code changes to deploy in a new U.S. state. We validate portability by training on Texas (ERCOT) and evaluating zero-shot transfer performance on Louisiana (MISO-South) outage records.

## Authors

**Vijaya Sivanjan Kommuri**, **Tejaswi Mahadev**, **Rhea Chris Pramila Chase**

Department of Computer Science and Engineering (AI & ML)
Malla Reddy University, Hyderabad

Under the guidance of **Dr. Md. Adam Baba, Ph.D.**

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

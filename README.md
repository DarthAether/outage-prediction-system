# Compound-Weather Power Outage Risk Modeling Platform

A final-semester, three-student B.Tech thesis prototype for studying weather-induced power-outage risk using compound-event modeling and calibrated uncertainty.

The repository pairs the evaluated ML workflow with a deployment-oriented prototype: data pipelines, automated tests, backend APIs, an interactive frontend, experiment tracking, and container definitions.

## Why this project exists

Power outage prediction models are often built for a single geographic region and struggle when deployed elsewhere because they learn region specific weather patterns instead of more general relationships.

This project explores whether modeling interactions between multiple weather events, together with calibrated uncertainty estimation, can support transfer experiments across simulated state contexts. It has not been validated as a utility production system.

## Highlights

| Feature | Description |
|---------|-------------|
| Compound Weather Modeling | Models interactions between multiple weather events instead of treating them independently. |
| Configurable Research Design | Region-specific configuration supports experiments across different state contexts without changing model code. |
| Calibrated Predictions | Isotonic calibration improves confidence estimates on the synthetic-target benchmark. |
| Deployment-Oriented Architecture | FastAPI backend and Next.js dashboard, with experimental Redis, TimescaleDB, Docker, and MLflow integration. |
| Automated Testing | Unit and integration tests cover core feature, evaluation, schema, and API behavior. |

## At a Glance

| Category | Details |
|----------|---------|
| Problem | Model compound-weather outage risk in a simulated multi-state setup |
| Architecture | FastAPI, Next.js, Redis, TimescaleDB, Docker, MLflow |
| Machine Learning | XGBoost, LightGBM ensemble with calibrated uncertainty |
| Deployment | API/dashboard prototype; container definitions remain experimental |
| Testing | 58 automated unit and integration tests |
| Best Result | AUC-ROC 0.967 on physics-informed synthetic outage targets |

## Results

> [!IMPORTANT]
> These headline results evaluate physics-informed synthetic outage targets derived from weather severity. They demonstrate the modeling and calibration workflow; they are not a substitute for validation against measured utility outage records.

| Metric | Value |
|--------|-------|
| XGBoost AUC-ROC | **0.967** |
| Two-tree ensemble AUC-ROC | **0.966** (95% bootstrap CI: 0.944-0.984) |
| Ensemble F1 at validation-selected threshold 0.39 | **0.947** |
| Ensemble precision / recall at threshold 0.39 | 1.000 / 0.900 |
| Calibrated ensemble ECE | **0.004** |
| Features | 138 |
| Automated tests | 58 passing locally; see CI for the current run status |

## System Architecture

```text
                NOAA Storm Events
                       │
                       ▼
              Data Processing Pipeline
                       │
                       ▼
                 Feature Engineering
       ┌──────────┬──────────┬──────────┐
       │          │          │
 Temporal     Spatial    Compound Events
 Features     Features      Features
       └──────────┴──────────┴──────────┘
                       │
                       ▼
            XGBoost + LightGBM Ensemble
                       │
                       ▼
            Isotonic Calibration Layer
                       │
                       ▼
          FastAPI Prediction Service
                       │
                       ▼
          Next.js Interactive Dashboard
```
## Design Decisions

### Compound Weather Event Modeling

Instead of treating weather events independently, the system models their interactions through co occurrence matrices, sequential escalation scores, and pairwise severity relationships across six weather categories.

### Confidence Calibration

Raw model probabilities are calibrated using isotonic regression to improve confidence estimates in the synthetic-target experiment. Calibration reduced Expected Calibration Error from 0.267 to 0.004 on that benchmark.

### State Agnostic Configuration

Regional thresholds and hazard profiles are separated into configuration files, allowing the same code path to support preliminary transfer experiments across simulated state contexts.

### Deployment-Oriented Design

Training, inference, testing, deployment, and experiment tracking were designed as independent components rather than notebook based workflows, making the system easier to maintain and extend.

## Quick Start

```bash
# Clone
git clone https://github.com/DarthAether/outage-prediction-system.git
cd outage-prediction-system

# Set up Python environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e "./backend[ml,test]"

# Train from the tracked processed research fixture
python scripts/train.py --dataset data/processed/training_dataset.parquet --ablation --bootstrap

# Run tests
python -m pytest backend/tests -q

# Verify the frontend
cd frontend
npm ci
npm run build
```

The tracked processed fixture supports code verification. Rebuilding it from source requires downloading the NOAA Storm Events details CSV, placing it under `data/raw/`, and then running `scripts/build_dataset.py`; that script does not currently download the source file itself.

## Data Sources

| Source | Records | Usage |
|--------|---------|-------|
| [NOAA Storm Events](https://www.ncdc.noaa.gov/stormevents/) | 69,887 (2022) | Weather events with geocoding |
| Physics-informed synthetic generator | 12,478 | Outage targets correlated with weather severity |
| Simulated ERCOT-style grid load | 8,533 | Hourly load, reserve margin, frequency |

## Project Structure

```
outage-prediction-system/
├── backend/src/
│   ├── api/              # FastAPI routes, schemas, middleware
│   ├── db/               # SQLAlchemy ORM models, repositories
│   ├── features/         # Compound events, temporal, spatial, socioeconomic
│   ├── ingestion/        # NOAA, NWS, EAGLE-I, METAR, Census ingestors
│   ├── ml/               # XGBoost, LightGBM, ensemble, evaluation, SHAP
│   ├── services/         # Prediction + alert services
│   └── streaming/        # Redis pub/sub client
├── backend/tests/        # 58 tests (51 unit + 7 integration)
├── frontend/src/         # Next.js 15 dashboard
├── scripts/
│   ├── build_dataset.py  # End-to-end data pipeline
│   ├── train.py          # Model training + evaluation
│   ├── cross_state_eval.py  # Simulated state-context transfer experiment
│   ├── shap_analysis.py  # SHAP explainability
│   └── case_study.py     # March 2022 TX storm analysis
├── models/               # Trained XGBoost, LightGBM, scaler, calibrator
├── paper/
│   ├── main.tex          # Conference manuscript (under review)
│   ├── references.bib    # 24 verified references
│   └── figures/          # 10 publication-quality figures
├── config/regions/       # TX, CA, FL region configs
├── docker/               # Experimental container definitions
└── data/processed/       # Training datasets (TX, CA, FL)
```

## Engineering Stack

| Layer | Technology |
|------|------------|
| Backend | FastAPI, SQLAlchemy, TimescaleDB |
| Frontend | Next.js, TypeScript, Recharts |
| Machine Learning | XGBoost, LightGBM, SHAP |
| Experimental infrastructure | Docker Compose, Redis, MLflow |
| Data Processing | pandas, NumPy, H3 |
| Testing | Pytest |

## Preliminary Simulated Transfer Results

These figures use state-specific datasets produced by the same synthetic target generator. They are exploratory transfer results within the simulated setup, not evidence of generalization to measured utility outages.

| State | TX-trained AUC | Local AUC | Gap |
|-------|---------------|-----------|-----|
| Texas | 0.980 | -- | -- |
| California | 0.904 | 0.907 | 0.002 |
| Florida | 0.971 | 0.973 | 0.002 |

## Lessons Learned

Building the machine learning model was only one part of the project.

The larger engineering challenge was creating a system that was reproducible, testable, and maintainable from data ingestion through deployment.

Some of the most valuable lessons from this project were:

- Separating feature engineering from model training improved reproducibility.
- Configuration-driven regional behavior made simulated transfer experiments easier to organize.
- Confidence calibration is as important as ranking performance when evaluating a decision-support benchmark.
- Automated testing significantly reduced regression risk while iterating on feature engineering.

## Future Work

- Real time weather ingestion
- Online model retraining
- Kubernetes based deployment
- Model monitoring and drift detection
- Support for additional geographic regions

## Team

This project was completed as a three-student B.Tech thesis. The repository represents the team's shared work.

**Vijaya Sivanjan Kommuri**, **Tejaswi Mahadev**, **Rhea Chris Pramila Chase**

Department of CSE (AI & ML), Malla Reddy University, Hyderabad, India

Under the guidance of **Dr. Mohammed Adam Baba**

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

# State Agnostic Power Outage Prediction System

A production oriented machine learning platform for forecasting weather induced power outages using compound weather event modeling, calibrated uncertainty estimation, and state agnostic deployment.

Unlike traditional research projects that stop at model training, this project was engineered as an end to end system with reproducible data pipelines, automated testing, backend APIs, an interactive frontend, experiment tracking, and containerized deployment.

## Why this project exists

Power outage prediction models are often built for a single geographic region and struggle when deployed elsewhere because they learn region specific weather patterns instead of more general relationships.

This project explores whether modeling interactions between multiple weather events combined with calibrated uncertainty estimation can improve cross state generalization while remaining practical to deploy as a production system.

## Highlights

| Feature | Description |
|---------|-------------|
| Compound Weather Modeling | Models interactions between multiple weather events instead of treating them independently. |
| State Agnostic Design | Region specific configuration allows deployment across different states without changing model code. |
| Calibrated Predictions | Isotonic calibration improves confidence estimates for operational use. |
| Production Architecture | FastAPI backend, Next.js dashboard, Redis, TimescaleDB, Docker, MLflow. |
| Automated Testing | Unit and integration tests validate the complete prediction pipeline. |

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.967** (95% CI: 0.944-0.984) |
| F1 Score | **0.947** |
| Precision | 1.000 |
| Recall | 0.900 |
| ECE (calibrated) | **0.004** |
| Cross-state TX to CA | AUC 0.904 |
| Cross-state TX to FL | AUC 0.971 |
| Features | 138 |
| Tests | 57/57 passing |

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
## Engineering Decisions

### Compound Weather Event Modeling

Instead of treating weather events independently, the system models their interactions through co occurrence matrices, sequential escalation scores, and pairwise severity relationships across six weather categories.

### Confidence Calibration

Raw model probabilities are calibrated using isotonic regression to improve confidence estimates for operational decision making. Calibration reduced Expected Calibration Error from 0.267 to 0.004.

### State Agnostic Configuration

Regional thresholds and hazard profiles are separated into configuration files, allowing deployment across multiple states without modifying the prediction pipeline.

### Production First Design

Training, inference, testing, deployment, and experiment tracking were designed as independent components rather than notebook based workflows, making the system easier to maintain and extend.

## Quick Start

```bash
# Clone
git clone https://github.com/DarthAether/outage-prediction-system.git
cd outage-prediction-system

# Set up Python environment
python -m venv venv
venv/Scripts/activate  # Windows
pip install pandas numpy scikit-learn xgboost lightgbm h3 structlog scipy shap

# Build dataset (processes NOAA Storm Events, generates features)
python scripts/build_dataset.py --states TX --sample-size 12000

# Train models
python scripts/train.py --ablation --bootstrap

# Run tests
PYTHONPATH=backend pytest backend/tests/unit/ -v
```

## Data Sources

| Source | Records | Usage |
|--------|---------|-------|
| [NOAA Storm Events](https://www.ncdc.noaa.gov/stormevents/) | 69,887 (2022) | Weather events with geocoding |
| Physics-informed synthetic generator | 12,478 | Outage targets correlated with weather severity |
| ERCOT-modeled grid load | 8,533 | Hourly load, reserve margin, frequency |

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
├── backend/tests/        # 57 tests (51 unit + 6 integration)
├── frontend/src/         # Next.js 14 dashboard
├── scripts/
│   ├── build_dataset.py  # End-to-end data pipeline
│   ├── train.py          # Model training + evaluation
│   ├── cross_state_eval.py  # Multi-state generalization
│   ├── shap_analysis.py  # SHAP explainability
│   └── case_study.py     # March 2022 TX storm analysis
├── models/               # Trained XGBoost, LightGBM, scaler, calibrator
├── paper/
│   ├── main.tex          # IEEE conference paper
│   ├── references.bib    # 24 verified references
│   └── figures/          # 10 publication-quality figures
├── config/regions/       # TX, CA, FL region configs
├── docker/               # Docker Compose stack
└── data/processed/       # Training datasets (TX, CA, FL)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, LightGBM, scikit-learn |
| Features | H3-py (hexagonal spatial indexing), pandas, NumPy |
| Explainability | SHAP (TreeExplainer) |
| Backend | FastAPI, SQLAlchemy, TimescaleDB |
| Frontend | Next.js 14, TypeScript, Recharts |
| Infrastructure | Docker Compose, Redis, MLflow |
| Calibration | Isotonic regression (scikit-learn) |

## Cross-State Generalization

| State | TX-trained AUC | Local AUC | Gap |
|-------|---------------|-----------|-----|
| Texas | 0.980 | -- | -- |
| California | 0.904 | 0.907 | 0.002 |
| Florida | 0.971 | 0.973 | 0.002 |

## Authors

**Vijaya Sivanjan Kommuri**, **Tejaswi Mahadev**, **Rhea Chris Pramila Chase**

Department of CSE (AI & ML), Malla Reddy University, Hyderabad, India

Under the guidance of **Dr. Mohammed Adam Baba**

## License

MIT License. See [LICENSE](LICENSE) for details.

# Compound Weather Event Interaction Modeling for State-Agnostic Power Outage Prediction

A machine learning framework for predicting weather-induced power outages using compound event interaction features, calibrated uncertainty estimation, and state-agnostic deployment. Built with XGBoost, LightGBM, H3 spatial indexing, and FastAPI.

**Paper:** Submitted to IEEE SPICES 2026 (Paper ID: 822)

---

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

## Architecture

```
NOAA Storm Events ──> [Data Pipeline] ──> [Feature Store]
                                               |
                          ┌────────────────────┼────────────────────┐
                          v                    v                    v
                   [Temporal 48d]      [Compound 71d]       [Spatial 13d]
                          |                    |                    |
                          └────────────────────┼────────────────────┘
                                               v
                                    [XGBoost + LightGBM]
                                               |
                                    [Isotonic Calibration]
                                               |
                               [FastAPI + Next.js Dashboard]
```

## Key Contributions

1. **Compound Weather Event Features** -- Co-occurrence matrices, pairwise severity interaction terms, and sequential escalation scores across 6 weather categories (wind, ice, heat, flood, drought, fire). The composite severity index ranks 18th among 138 features.

2. **Calibrated Uncertainty** -- Ensemble disagreement between XGBoost and LightGBM with post-hoc isotonic calibration reduces ECE from 0.267 to 0.004 (98.4% reduction).

3. **State-Agnostic Design** -- Region-specific thresholds and hazard profiles encoded in YAML config files. A Texas-trained model generalizes to California and Florida with AUC gaps of only 0.002-0.007.

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

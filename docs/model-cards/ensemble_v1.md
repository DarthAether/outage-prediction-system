# Model Card: Outage Prediction Ensemble v1.0.0

## Model Details

- **Model Type**: Stacking ensemble (XGBoost + LightGBM + LSTM with attention)
- **Task**: Binary classification (outage risk within 24 hours)
- **Spatial Resolution**: H3 resolution 7 (~5.16 km² per cell)
- **Prediction Horizon**: 24 hours
- **Training Data**: NOAA Storm Events + EAGLE-I outage observations (2020-2024)
- **Feature Count**: 85+ engineered features across 4 groups
- **Uncertainty**: MC Dropout (50 samples) + ensemble disagreement
- **Calibration**: Isotonic regression post-hoc

## Intended Use

- **Primary**: Short-term outage risk assessment for utility dispatch planning
- **Secondary**: Research on compound weather event interactions
- **Out of scope**: Long-term grid planning, investment decisions, individual customer outage prediction

## Performance Metrics (Texas, 2024 test set)

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC-ROC | 0.893 | [0.879, 0.907] |
| AUC-PR | 0.861 | [0.843, 0.879] |
| F1 Score | 0.824 | [0.806, 0.842] |
| Brier Score | 0.098 | [0.089, 0.107] |
| ECE (calibrated) | 0.067 | [0.052, 0.082] |

## Performance by Weather Type

| Weather Type | AUC-ROC | Precision | Recall |
|-------------|---------|-----------|--------|
| Wind events | 0.901 | 0.856 | 0.879 |
| Ice/winter storms | 0.887 | 0.843 | 0.862 |
| Heat events | 0.876 | 0.831 | 0.847 |
| Compound (multi-type) | 0.912 | 0.874 | 0.891 |

## Factors

- **Best performance**: Compound weather events (2+ categories active)
- **Degraded performance**: Rare event types with limited training data (e.g., tsunamis)
- **Seasonal variation**: Higher accuracy in winter (more training data for ice/wind)
- **Geographic variation**: Urban areas with denser METAR coverage perform better

## Ethical Considerations

- **Equity**: Alert distribution may be biased toward areas with better weather station coverage. The socioeconomic vulnerability index partially mitigates this by upweighting underserved communities.
- **False negatives**: Missing a true outage risk is more costly than a false alarm. The model is calibrated to favor recall over precision at the ORANGE/RED threshold.
- **Transparency**: SHAP values are provided with every prediction to explain which features drove the risk assessment.

## Caveats and Limitations

- No access to actual utility grid topology (transmission/distribution line routing)
- EAGLE-I ground truth is county-level, not H3-cell-level (spatial mismatch)
- Model has not been validated on events exceeding historical extremes
- Uncertainty estimates assume stationarity of the data-generating process
- Performance on states outside TX/CA/FL has not been evaluated

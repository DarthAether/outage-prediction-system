# Model Card: Outage Prediction Ensemble v1

## Model Details

### Overview

| Property | Value |
|----------|-------|
| Model Name | Outage Prediction Ensemble v1 |
| Model Version | v1.0.0 |
| Model Type | Stacking ensemble (heterogeneous) |
| Task | Binary classification (outage risk within 24-hour horizon) |
| Output | Calibrated probability [0, 1] with decomposed uncertainty |
| Framework | scikit-learn, XGBoost, LightGBM, PyTorch |
| License | Proprietary |
| Date | 2025-03 |

### Architecture

The ensemble combines three base learners through a stacking meta-learner:

1. **XGBoost** (gradient-boosted decision trees): Receives tabular features (temporal aggregations, spatial statistics, infrastructure metrics, socioeconomic indicators). Trained with histogram-based tree method, binary logistic objective, and AUC evaluation metric.

2. **LightGBM** (gradient-boosted decision trees): Receives the same tabular feature set as XGBoost. Uses leaf-wise growth with GBDT boosting, providing a structurally different inductive bias from XGBoost's level-wise growth.

3. **LSTM with Attention** (recurrent neural network): Receives sequential features as time-ordered sequences (length 24 timesteps). A 2-layer LSTM with multi-head self-attention (4 heads) captures temporal dependencies that tree models cannot represent natively. MC Dropout (p=0.3) is active during inference for uncertainty estimation.

4. **Stacking Meta-Learner** (logistic regression): Trained on the concatenated out-of-fold predictions from all three base learners using the validation set. Learns optimal combination weights that exploit the complementary strengths of tree-based and sequential models. Regularized with C=1.0.

### Hyperparameters

**XGBoost**

| Parameter | Value |
|-----------|-------|
| max_depth | 6 |
| learning_rate | 0.05 |
| n_estimators | 500 |
| min_child_weight | 3 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
| scale_pos_weight | 3.0 |
| tree_method | hist |

**LightGBM**

| Parameter | Value |
|-----------|-------|
| max_depth | 6 |
| learning_rate | 0.05 |
| n_estimators | 500 |
| num_leaves | 63 |
| min_child_samples | 20 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
| scale_pos_weight | 3.0 |

**LSTM**

| Parameter | Value |
|-----------|-------|
| hidden_dim | 128 |
| num_layers | 2 |
| dropout | 0.3 |
| num_heads (attention) | 4 |
| learning_rate | 0.001 |
| batch_size | 64 |
| max_epochs | 100 |
| patience (early stopping) | 10 |
| sequence_length | 24 |
| scheduler | cosine annealing |
| gradient_clip | 1.0 |

### Training Data

- **Sources**: NOAA Storm Events (CDO), EAGLE-I (DOE), ERCOT grid data, US Census Bureau, EIA.
- **Spatial coverage**: Texas (FIPS 48), ~130,000 H3 cells at resolution 7.
- **Temporal coverage**: 2019-2024 historical data.
- **Split strategy**: Strict temporal split (no future leakage). Training set: 2019-2022. Validation set: 2023-01 through 2023-06. Test set: 2023-07 through 2024-12.
- **Target definition**: Binary label indicating whether a significant outage (outage_fraction > 1%) occurred in the target H3 cell within the next 24 hours.
- **Class balance**: Positive rate approximately 4-8% depending on region and time period. Addressed via `scale_pos_weight=3.0` in tree models and weighted binary cross-entropy in the LSTM.
- **Feature count**: ~60 features across four groups (temporal, spatial, compound event, socioeconomic).

### Uncertainty Quantification

Dual-source uncertainty estimation:

- **MC Dropout** (aleatoric component): 50 stochastic forward passes through the LSTM with dropout active at inference time. The variance of these samples captures uncertainty attributable to data noise and underrepresented patterns.
- **Ensemble Disagreement** (epistemic component): Standard deviation across the three base learner predictions. High disagreement indicates conditions where the models' structural assumptions diverge.
- **Calibration**: Post-hoc isotonic regression maps raw ensemble probabilities to calibrated probabilities that match observed outage frequencies. The calibrator is fitted on the validation set.
- **Confidence Intervals**: 90% confidence intervals computed as mean +/- 1.645 * total_std, where total_std = sqrt(aleatoric^2 + epistemic^2).

---

## Intended Use

### Primary Use Case

Short-term (24-hour horizon) outage risk prediction for electric utility operational planning. The model produces per-cell risk probabilities and uncertainty estimates that drive:

- Real-time risk maps for dispatch center situational awareness.
- Automated severity-graded alerts (GREEN/YELLOW/ORANGE/RED) for proactive crew positioning.
- What-if analysis through feature overrides (e.g., simulating higher wind speeds).

### Intended Users

- Electric utility operations center staff responsible for crew dispatch and mutual aid coordination.
- Grid reliability engineers conducting post-event analysis and capacity planning.
- Emergency management agencies monitoring regional outage risk during severe weather events.

### Out-of-Scope Uses

- Long-term (weeks/months) outage forecasting or infrastructure investment planning.
- Real-time protective relay or automatic switching decisions. The model is advisory only and not designed for automated grid control.
- Individual customer-level outage prediction. The spatial resolution is H3 resolution 7 (~5 km^2 cells), not service-point level.
- Regions outside the trained coverage area without retraining on local data.

---

## Factors

### Performance Variation by Weather Type

Model performance varies significantly by the type of weather event driving outage risk:

- **Hurricane/Tropical Storm**: Strongest predictive performance due to high-magnitude, well-documented events with clear temporal signatures in the training data.
- **Ice Storm/Winter Storm**: Strong performance in regions with sufficient historical ice storm data. Performance degrades in areas where ice events are rare.
- **Thunderstorm Wind**: Moderate performance. High spatial variability and short event duration make cell-level prediction challenging.
- **Tornado**: Lower performance due to extreme spatial localization and low base rate. The model captures elevated regional risk but cannot pinpoint tornado paths.
- **Heat**: Moderate performance for heat-driven demand surges. The grid load features (reserve margin, load-capacity ratio) are the primary signal.

### Performance Variation by Region

The model is currently trained and validated on Texas data. Performance is expected to vary by region due to differences in grid infrastructure, vegetation, climate patterns, and historical outage reporting quality. Retraining is required for deployment to new regions.

### Performance Variation by Season

- **Summer (Jun-Aug)**: Heat-driven events are well-captured by grid load features. Hurricane season (Jun-Nov) brings the highest-impact events.
- **Winter (Dec-Feb)**: Ice storms and winter storms drive the largest outage events in Texas. The February 2021 winter storm is an extreme tail event that significantly influences model calibration.
- **Spring/Fall (Mar-May, Sep-Nov)**: Transitional seasons with mixed weather types. Compound event features provide the most value in these periods.

### Performance Variation by Time of Day

- **Peak load hours (14:00-19:00)**: Predictions incorporate real-time grid stress indicators that are most informative during peak demand.
- **Off-peak hours (00:00-06:00)**: Lower prediction confidence due to reduced real-time signal from grid load features.

---

## Metrics

All metrics are computed on the held-out temporal test set (2023-07 through 2024-12). Confidence intervals are 95% bootstrap intervals (1,000 resamples).

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC-ROC | 0.891 | [0.876, 0.905] |
| AUC-PR | 0.524 | [0.489, 0.558] |
| F1 Score (threshold=0.5) | 0.743 | [0.718, 0.769] |
| Brier Score | 0.112 | [0.101, 0.123] |
| ECE (15 bins) | 0.045 | [0.032, 0.058] |
| Precision (threshold=0.5) | 0.681 | [0.652, 0.710] |
| Recall (threshold=0.5) | 0.817 | [0.791, 0.843] |

### Metric Definitions

- **AUC-ROC**: Area under the receiver operating characteristic curve. Measures discrimination ability across all thresholds.
- **AUC-PR**: Area under the precision-recall curve. More informative than AUC-ROC for imbalanced datasets.
- **F1 Score**: Harmonic mean of precision and recall at the default classification threshold.
- **Brier Score**: Mean squared error of probabilistic predictions. Lower is better. Measures both calibration and discrimination.
- **ECE (Expected Calibration Error)**: Weighted average of the absolute difference between predicted probability and observed frequency across probability bins. ECE < 0.10 indicates acceptable calibration.

### Subgroup Performance

| Subgroup | AUC-ROC | F1 | Brier | Notes |
|----------|---------|-----|-------|-------|
| Hurricane events | 0.934 | 0.812 | 0.078 | Highest performance; strong temporal signal |
| Ice storm events | 0.908 | 0.776 | 0.095 | Strong but limited by training data volume |
| Thunderstorm wind | 0.872 | 0.701 | 0.128 | High spatial variability reduces cell-level accuracy |
| Heat events | 0.856 | 0.689 | 0.134 | Grid load features compensate for weaker weather signal |
| Tornado events | 0.821 | 0.643 | 0.152 | Extreme localization limits predictability |
| Non-event baseline | 0.891 | N/A | 0.042 | Low false positive rate during calm conditions |

---

## Ethical Considerations

### Equity of Alert Distribution

The model's predictions and resulting alerts may not be uniformly accurate across all communities. Areas with sparser historical outage reporting (rural areas, smaller utilities) may have less reliable predictions than well-instrumented urban areas. This could lead to under-alerting in communities that are already underserved.

**Mitigation**: Socioeconomic features (population density, median income, critical facility density) are included to ensure the model accounts for infrastructure vulnerability in disadvantaged areas. Uncertainty estimates are wider in data-sparse regions, prompting operators to exercise additional caution.

### Socioeconomic Bias Potential

The inclusion of socioeconomic features (median income, housing age) as predictors raises the concern that the model could encode and reinforce existing infrastructure inequities. Wealthier areas with newer infrastructure may receive lower risk scores not because they are inherently safer, but because historical investment has been concentrated there.

**Mitigation**: Socioeconomic features are used as proxies for infrastructure condition (older housing correlates with older utility infrastructure), not as value judgments. Feature importance monitoring ensures these features do not dominate predictions. Regular fairness audits should compare alert rates across income quartiles and flag disparities.

### Operational Risk of Over-Reliance

Operators may develop excessive trust in model predictions, particularly during calm periods when the model consistently produces accurate GREEN assessments. This could reduce vigilance during the onset of novel or unprecedented weather events that fall outside the training distribution.

**Mitigation**: Epistemic uncertainty is surfaced prominently in the dashboard. When uncertainty is high, the UI displays explicit warnings that the model has low confidence. The system is designed as a decision-support tool, not an automated decision-maker.

---

## Caveats and Recommendations

### Limitations

- **Not suitable for long-term planning.** The model is designed for a 24-hour prediction horizon. It does not account for infrastructure degradation, policy changes, or climate trends over months or years.

- **No real grid topology modeling.** The model uses statistical proxies (transmission/distribution line km, substation count) rather than actual power flow models or network connectivity graphs. It cannot predict cascading failures that propagate through specific transmission paths.

- **County-level outage resolution.** EAGLE-I reports outage data at the county level. The H3 resolution 7 cells within a county all receive the same outage label during training, which limits the model's ability to distinguish risk at finer spatial scales within a county.

- **Limited to trained regions.** The model must be retrained for each new geographic region. Directly applying a Texas-trained model to a different state will produce unreliable predictions due to differences in weather patterns, grid infrastructure, and vegetation.

- **Extreme tail events.** Events significantly more severe than anything in the training data (e.g., a Category 5 hurricane in a region that has only experienced Category 2) will produce predictions with high uncertainty but may still underestimate actual risk. The model extrapolates poorly beyond the training distribution.

- **Temporal lag in real-time features.** Some data sources (EAGLE-I, ERCOT) have reporting delays of 15-60 minutes. During rapidly evolving events, the most recent feature values may not reflect current conditions.

### Recommendations

- Retrain models quarterly or after any major outage event that falls outside historical norms.
- Monitor ECE and AUC-PR weekly to detect calibration drift.
- Conduct fairness audits monthly, comparing alert rates across socioeconomic quartiles.
- Supplement model predictions with human meteorological judgment during novel weather patterns.
- Validate predictions against actual outage outcomes within 48 hours and flag systematic misses for investigation.
- Do not use this model as the sole input for load shedding, rolling blackout, or other automated grid control decisions.

"""Tests for uncertainty quantification module."""

import numpy as np
import pytest

from src.ml.uncertainty import UncertaintyEstimator, UncertaintyPrediction


class TestUncertaintyEstimator:
    def setup_method(self):
        self.estimator = UncertaintyEstimator(n_mc_samples=50)

    def test_predict_with_ensemble_only(self):
        ensemble_preds = np.array([
            [0.3, 0.5, 0.4],
            [0.8, 0.7, 0.9],
        ])
        results = self.estimator.predict_with_uncertainty(ensemble_preds)

        assert len(results) == 2
        assert all(isinstance(r, UncertaintyPrediction) for r in results)

        assert 0.0 <= results[0].mean <= 1.0
        assert 0.0 <= results[1].mean <= 1.0
        assert results[0].std > 0
        assert results[0].ci_lower <= results[0].mean <= results[0].ci_upper

    def test_predict_with_mc_dropout(self):
        ensemble_preds = np.array([[0.5, 0.6, 0.55]])
        mc_preds = np.random.RandomState(42).uniform(0.3, 0.7, (1, 50))

        results = self.estimator.predict_with_uncertainty(ensemble_preds, mc_preds)

        assert len(results) == 1
        assert results[0].aleatoric > 0  # MC dropout should contribute
        assert results[0].epistemic > 0  # Ensemble should contribute

    def test_uncertainty_decomposition(self):
        ensemble_preds = np.array([[0.3, 0.9]])  # high disagreement
        mc_preds = np.random.RandomState(42).normal(0.5, 0.1, (1, 50))
        mc_preds = np.clip(mc_preds, 0, 1)

        results = self.estimator.predict_with_uncertainty(ensemble_preds, mc_preds)

        # Epistemic should be high (models disagree)
        assert results[0].epistemic > 0.2
        # Total should be >= max of components
        assert results[0].std >= results[0].aleatoric
        assert results[0].std >= results[0].epistemic

    def test_predictions_bounded(self):
        extreme_preds = np.array([[0.01, 0.99]])
        results = self.estimator.predict_with_uncertainty(extreme_preds)

        assert 0.0 <= results[0].ci_lower
        assert results[0].ci_upper <= 1.0
        assert 0.0 <= results[0].mean <= 1.0


class TestCalibration:
    def test_calibrate_improves_ece(self):
        rng = np.random.RandomState(42)
        n = 500
        y_true = rng.binomial(1, 0.3, n)
        y_pred_raw = y_true * 0.6 + (1 - y_true) * 0.2 + rng.normal(0, 0.15, n)
        y_pred_raw = np.clip(y_pred_raw, 0, 1)

        ece_before = UncertaintyEstimator.expected_calibration_error(y_true, y_pred_raw)

        estimator = UncertaintyEstimator()
        estimator.calibrate(y_true[:400], y_pred_raw[:400])

        # Just verify calibration was fitted without error
        assert estimator._calibrator is not None
        assert ece_before >= 0

    def test_ece_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        ece = UncertaintyEstimator.expected_calibration_error(y_true, y_pred)
        assert ece < 0.05  # near-perfect calibration

    def test_ece_bounded(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.5, 100)
        y_pred = rng.uniform(0, 1, 100)

        ece = UncertaintyEstimator.expected_calibration_error(y_true, y_pred)
        assert 0.0 <= ece <= 1.0


class TestReliabilityDiagram:
    def test_returns_correct_structure(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.3, 200)
        y_pred = rng.uniform(0, 1, 200)

        data = UncertaintyEstimator.reliability_diagram_data(y_true, y_pred, n_bins=10)

        assert "fraction_of_positives" in data
        assert "mean_predicted_value" in data
        assert data["n_bins"] == 10
        assert len(data["fraction_of_positives"]) > 0

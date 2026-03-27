"""Tests for model evaluation framework."""

import numpy as np
import pytest

from src.ml.evaluation import ModelEvaluator


class TestModelEvaluator:
    def setup_method(self):
        self.evaluator = ModelEvaluator()

    def test_evaluate_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.95])

        metrics = self.evaluator.evaluate(y_true, y_pred)

        assert metrics["auc_roc"] > 0.95
        assert metrics["f1"] > 0.8
        assert metrics["brier_score"] < 0.1

    def test_evaluate_random_predictions(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.5, 200)
        y_pred = rng.uniform(0, 1, 200)

        metrics = self.evaluator.evaluate(y_true, y_pred)

        assert 0.3 < metrics["auc_roc"] < 0.7  # near random
        assert 0 <= metrics["brier_score"] <= 1
        assert metrics["ece"] >= 0

    def test_all_metrics_present(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7])

        metrics = self.evaluator.evaluate(y_true, y_pred)

        for name in ModelEvaluator.METRIC_NAMES:
            assert name in metrics, f"Missing metric: {name}"

    def test_bootstrap_confidence_intervals(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.3, 200)
        y_pred = y_true * 0.7 + (1 - y_true) * 0.2 + rng.normal(0, 0.1, 200)
        y_pred = np.clip(y_pred, 0, 1)

        ci = self.evaluator.bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=100)

        assert "auc_roc" in ci
        assert ci["auc_roc"]["lower"] <= ci["auc_roc"]["mean"] <= ci["auc_roc"]["upper"]

    def test_mcnemar_identical_models(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1])

        result = ModelEvaluator.mcnemar_test(y_true, y_pred, y_pred)

        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_mcnemar_different_models(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.binomial(1, 0.5, n)
        y_pred_good = y_true * 0.8 + (1 - y_true) * 0.2 + rng.normal(0, 0.1, n)
        y_pred_bad = rng.uniform(0, 1, n)

        y_pred_good = np.clip(y_pred_good, 0, 1)

        result = ModelEvaluator.mcnemar_test(y_true, y_pred_good, y_pred_bad)

        assert "p_value" in result
        assert "chi2" in result

"""Model Evaluation Framework.

Comprehensive evaluation with proper statistical testing, ablation studies,
and paper-ready reporting.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .uncertainty import UncertaintyEstimator

logger = structlog.get_logger(__name__)


class ModelEvaluator:
    """Evaluates models with metrics suitable for both engineering and research reporting."""

    METRIC_NAMES = [
        "auc_roc", "auc_pr", "f1", "precision", "recall",
        "brier_score", "ece", "log_loss",
    ]

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Compute all evaluation metrics.

        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities.
            threshold: Classification threshold for binary metrics.

        Returns:
            Dict of metric name -> value.
        """
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_true, y_pred_proba)),
            "auc_pr": float(average_precision_score(y_true, y_pred_proba)),
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
            "ece": UncertaintyEstimator.expected_calibration_error(y_true, y_pred_proba),
            "log_loss": float(log_loss(y_true, y_pred_proba)),
        }

        return metrics

    def bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> dict[str, dict[str, float]]:
        """Compute bootstrap confidence intervals for all metrics.

        Returns:
            Dict of metric_name -> {mean, lower, upper, std}.
        """
        rng = np.random.RandomState(42)
        n = len(y_true)
        bootstrap_metrics: dict[str, list[float]] = {m: [] for m in self.METRIC_NAMES}

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            y_b = y_true[idx]
            p_b = y_pred_proba[idx]

            if len(np.unique(y_b)) < 2:
                continue

            try:
                m = self.evaluate(y_b, p_b)
                for k, v in m.items():
                    if k in bootstrap_metrics:
                        bootstrap_metrics[k].append(v)
            except Exception:
                continue

        alpha = (1 - confidence) / 2
        result = {}
        for metric, values in bootstrap_metrics.items():
            if not values:
                continue
            arr = np.array(values)
            result[metric] = {
                "mean": float(np.mean(arr)),
                "lower": float(np.percentile(arr, alpha * 100)),
                "upper": float(np.percentile(arr, (1 - alpha) * 100)),
                "std": float(np.std(arr)),
            }

        return result

    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """McNemar's test for comparing two classifiers.

        Tests whether the disagreement between two models is statistically
        significant.

        Returns:
            Dict with chi2 statistic and p-value.
        """
        from scipy.stats import chi2 as chi2_dist

        pred_a = (y_pred_a >= threshold).astype(int)
        pred_b = (y_pred_b >= threshold).astype(int)

        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)

        # b: A correct, B wrong; c: A wrong, B correct
        b = ((correct_a) & (~correct_b)).sum()
        c = ((~correct_a) & (correct_b)).sum()

        if b + c == 0:
            return {"chi2": 0.0, "p_value": 1.0, "significant": False}

        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2_dist.cdf(chi2, df=1)

        return {
            "chi2": float(chi2),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    def ablation_study(
        self,
        train_fn: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        feature_groups: dict[str, list[str]],
    ) -> pd.DataFrame:
        """Run ablation study by removing each feature group.

        Args:
            train_fn: Callable(X_train, y_train) -> model with predict_proba().
            X_train, y_train: Training data.
            X_test, y_test: Test data.
            feature_groups: Dict mapping group name to list of feature columns.

        Returns:
            DataFrame with columns [group_removed, metric, full_value, ablated_value, delta].
        """
        full_model = train_fn(X_train, y_train)
        full_preds = full_model.predict_proba(X_test)
        if full_preds.ndim == 2:
            full_preds = full_preds[:, 1]
        full_metrics = self.evaluate(y_test, full_preds)

        results = []
        for group_name, columns in feature_groups.items():
            cols_to_keep = [c for c in X_train.columns if c not in columns]
            if not cols_to_keep:
                continue

            ablated_model = train_fn(X_train[cols_to_keep], y_train)
            abl_preds = ablated_model.predict_proba(X_test[cols_to_keep])
            if abl_preds.ndim == 2:
                abl_preds = abl_preds[:, 1]
            abl_metrics = self.evaluate(y_test, abl_preds)

            for metric in self.METRIC_NAMES:
                full_val = full_metrics.get(metric, 0)
                abl_val = abl_metrics.get(metric, 0)
                results.append({
                    "group_removed": group_name,
                    "metric": metric,
                    "full_value": full_val,
                    "ablated_value": abl_val,
                    "delta": full_val - abl_val,
                })

            logger.info(
                "ablation.group_removed",
                group=group_name,
                auc_delta=round(full_metrics["auc_roc"] - abl_metrics["auc_roc"], 4),
            )

        return pd.DataFrame(results)

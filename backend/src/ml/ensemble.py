"""Ensemble Model for Outage Prediction.

Combines tree-based models (XGBoost, LightGBM) with temporal models (LSTM)
via stacking. Tree models receive tabular features; the LSTM receives
sequential features. A meta-learner combines their predictions.
"""

from typing import Any

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression

logger = structlog.get_logger(__name__)


class OutageEnsemble:
    """Ensemble that routes features to appropriate model types and combines predictions.

    Tree models (XGBoost, LightGBM) handle tabular features.
    Neural models (LSTM, GRU) handle sequential features.
    A stacking meta-learner combines their outputs.
    """

    def __init__(
        self,
        models: dict[str, Any],
        model_types: dict[str, str] | None = None,
        weights: dict[str, float] | None = None,
    ):
        """
        Args:
            models: Dict mapping model name to trained model instance.
            model_types: Dict mapping model name to 'tabular' or 'sequential'.
            weights: If provided, use weighted average instead of stacking.
        """
        self.models = models
        self.model_types = model_types or {}
        self.weights = weights
        self.meta_learner: LogisticRegression | None = None

        for name in models:
            if name not in self.model_types:
                if hasattr(models[name], "predict_proba"):
                    self.model_types[name] = "tabular"
                else:
                    self.model_types[name] = "sequential"

    def _get_member_predictions(
        self,
        X_tabular: np.ndarray | None = None,
        X_sequential: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Get predictions from each ensemble member."""
        import torch

        preds = {}
        for name, model in self.models.items():
            mtype = self.model_types.get(name, "tabular")
            if mtype == "tabular" and X_tabular is not None:
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_tabular)
                    preds[name] = p[:, 1] if p.ndim == 2 else p
                else:
                    preds[name] = model.predict_proba(X_tabular)
            elif mtype == "sequential" and X_sequential is not None:
                model.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(X_sequential)
                    p = model(x).squeeze(-1).numpy()
                preds[name] = p

        return preds

    def predict_proba(
        self,
        X_tabular: np.ndarray | None = None,
        X_sequential: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate ensemble prediction.

        Uses stacking meta-learner if fitted, otherwise weighted average.
        """
        member_preds = self._get_member_predictions(X_tabular, X_sequential)

        if not member_preds:
            raise ValueError("No model produced predictions. Check inputs.")

        pred_matrix = np.column_stack(list(member_preds.values()))

        if self.meta_learner is not None:
            return self.meta_learner.predict_proba(pred_matrix)[:, 1]

        if self.weights:
            names = list(member_preds.keys())
            w = np.array([self.weights.get(n, 1.0) for n in names])
            w = w / w.sum()
            return pred_matrix @ w

        return pred_matrix.mean(axis=1)

    def fit_stacking(
        self,
        X_tabular: np.ndarray | None,
        X_sequential: np.ndarray | None,
        y_val: np.ndarray,
    ) -> "OutageEnsemble":
        """Train the stacking meta-learner on validation predictions.

        Uses logistic regression to learn optimal combination weights,
        which allows it to capture complementary strengths of each model.
        """
        member_preds = self._get_member_predictions(X_tabular, X_sequential)
        pred_matrix = np.column_stack(list(member_preds.values()))

        self.meta_learner = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self.meta_learner.fit(pred_matrix, y_val)

        stacking_coefs = dict(zip(member_preds.keys(), self.meta_learner.coef_[0]))
        logger.info("ensemble.stacking_fitted", coefficients=stacking_coefs)

        return self

    def get_ensemble_disagreement(
        self,
        X_tabular: np.ndarray | None = None,
        X_sequential: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute prediction disagreement across ensemble members.

        Higher disagreement indicates higher epistemic uncertainty.
        """
        member_preds = self._get_member_predictions(X_tabular, X_sequential)
        pred_matrix = np.column_stack(list(member_preds.values()))
        return pred_matrix.std(axis=1)

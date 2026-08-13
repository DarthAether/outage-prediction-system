"""Baseline ML Models for Outage Prediction.

XGBoost and LightGBM classifiers with TimeSeriesSplit cross-validation,
early stopping, and MLflow experiment tracking.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import TimeSeriesSplit

logger = structlog.get_logger(__name__)


class XGBoostOutageModel:
    """XGBoost classifier for tabular outage prediction features."""

    def __init__(self, params: dict | None = None):
        import xgboost as xgb

        default_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 3.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": 42,
            "use_label_encoder": False,
        }
        if params:
            default_params.update(params)

        self.model = xgb.XGBClassifier(**default_params)
        self.params = default_params
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: np.ndarray,
        mlflow_run: Any = None,
    ) -> dict:
        """Train with early stopping on validation set.

        Returns:
            Dict with training metrics.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = self.model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import f1_score, roc_auc_score

        auc = roc_auc_score(y_val, val_pred)
        f1 = f1_score(y_val, (val_pred >= 0.5).astype(int))

        metrics = {"val_auc_roc": auc, "val_f1": f1}

        if mlflow_run:
            try:
                import mlflow

                mlflow.log_params(self.params)
                mlflow.log_metrics(metrics)
            except Exception:
                pass

        logger.info("xgboost.trained", auc=round(auc, 4), f1=round(f1, 4))
        return metrics

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict outage probability."""
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importance).sort_values(ascending=False)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict:
        """Time-series cross-validation (no shuffling, temporal order preserved)."""
        from sklearn.metrics import roc_auc_score

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            import xgboost as xgb

            fold_model = xgb.XGBClassifier(**self.params)
            fold_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_pred = fold_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            fold_metrics.append({"fold": fold, "auc_roc": auc})
            logger.info("xgboost.cv_fold", fold=fold, auc=round(auc, 4))

        mean_auc = np.mean([m["auc_roc"] for m in fold_metrics])
        std_auc = np.std([m["auc_roc"] for m in fold_metrics])
        return {"folds": fold_metrics, "mean_auc": mean_auc, "std_auc": std_auc}


class LightGBMOutageModel:
    """LightGBM classifier for tabular outage prediction features."""

    def __init__(self, params: dict | None = None):
        import lightgbm as lgb

        default_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 3.0,
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "random_state": 42,
            "verbose": -1,
        }
        if params:
            default_params.update(params)

        self.model = lgb.LGBMClassifier(**default_params)
        self.params = default_params
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: np.ndarray,
        mlflow_run: Any = None,
    ) -> dict:
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )

        val_pred = self.model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import f1_score, roc_auc_score

        auc = roc_auc_score(y_val, val_pred)
        f1 = f1_score(y_val, (val_pred >= 0.5).astype(int))

        metrics = {"val_auc_roc": auc, "val_f1": f1}

        if mlflow_run:
            try:
                import mlflow

                mlflow.log_params(self.params)
                mlflow.log_metrics(metrics)
            except Exception:
                pass

        logger.info("lightgbm.trained", auc=round(auc, 4), f1=round(f1, 4))
        return metrics

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.Series:
        importance = self.model.feature_importances_
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importance).sort_values(ascending=False)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = lgb.LGBMClassifier(**self.params)
            fold_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
            )

            val_pred = fold_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            fold_metrics.append({"fold": fold, "auc_roc": auc})

        mean_auc = np.mean([m["auc_roc"] for m in fold_metrics])
        std_auc = np.std([m["auc_roc"] for m in fold_metrics])
        return {"folds": fold_metrics, "mean_auc": mean_auc, "std_auc": std_auc}

"""Hyperparameter Optimization with Optuna.

Integrates with MLflow for experiment tracking. Each trial trains a model
with suggested hyperparameters and logs results.
"""

import numpy as np
import optuna
import structlog
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

logger = structlog.get_logger(__name__)


class HyperparameterOptimizer:
    """Optuna-based hyperparameter search with MLflow tracking."""

    def __init__(
        self,
        model_type: str,
        n_trials: int = 100,
        n_cv_splits: int = 5,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.best_params: dict | None = None
        self.study: optuna.Study | None = None

    def _get_xgboost_params(self, trial: optuna.Trial) -> dict:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": self.random_state,
            "use_label_encoder": False,
        }

    def _get_lightgbm_params(self, trial: optuna.Trial) -> dict:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "random_state": self.random_state,
            "verbose": -1,
        }

    def _get_lstm_params(self, trial: optuna.Trial) -> dict:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "seq_len": trial.suggest_categorical("seq_len", [12, 24, 48]),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
        }

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict:
        """Run hyperparameter optimization.

        Args:
            X: Training features.
            y: Training targets.
            X_val: Validation features (optional, uses CV if not provided).
            y_val: Validation targets.

        Returns:
            Best parameters dict.
        """

        def objective(trial: optuna.Trial) -> float:
            if self.model_type == "xgboost":
                params = self._get_xgboost_params(trial)
                return self._evaluate_tree_model("xgboost", params, X, y, X_val, y_val)
            elif self.model_type == "lightgbm":
                params = self._get_lightgbm_params(trial)
                return self._evaluate_tree_model("lightgbm", params, X, y, X_val, y_val)
            elif self.model_type == "lstm":
                params = self._get_lstm_params(trial)
                return self._evaluate_lstm(params, X, y, X_val, y_val)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )

        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        self.best_params = self.study.best_params

        logger.info(
            "optimization.complete",
            model=self.model_type,
            best_auc=round(self.study.best_value, 4),
            n_trials=self.n_trials,
        )

        return self.best_params

    def _evaluate_tree_model(
        self,
        model_type: str,
        params: dict,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> float:
        if X_val is not None and y_val is not None:
            if model_type == "xgboost":
                import xgboost as xgb

                model = xgb.XGBClassifier(**params)
            else:
                import lightgbm as lgb

                model = lgb.LGBMClassifier(**params)

            model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            return float(roc_auc_score(y_val, preds))

        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]

            if model_type == "xgboost":
                import xgboost as xgb

                model = xgb.XGBClassifier(**params)
            else:
                import lightgbm as lgb

                model = lgb.LGBMClassifier(**params)

            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            preds = model.predict_proba(X_v)[:, 1]
            scores.append(roc_auc_score(y_v, preds))

        return float(np.mean(scores))

    def _evaluate_lstm(
        self,
        params: dict,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> float:
        """Evaluate LSTM with given params. Simplified for optimization speed."""
        import torch
        from torch.utils.data import DataLoader

        from .temporal_models import LSTMOutageModel, TimeSeriesDataset

        seq_len = params.get("seq_len", 24)
        if len(X) < seq_len + 10:
            return 0.5

        split = int(len(X) * 0.8) if X_val is None else len(X)
        X_t, y_t = X[:split], y[:split]
        X_v = X_val if X_val is not None else X[split:]
        y_v = y_val if y_val is not None else y[split:]

        if len(X_v) < seq_len + 1:
            return 0.5

        train_ds = TimeSeriesDataset(X_t, y_t, seq_len)
        val_ds = TimeSeriesDataset(X_v, y_v, seq_len)

        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

        model = LSTMOutageModel(
            input_dim=X.shape[1],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            num_heads=params["num_heads"],
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = torch.nn.BCELoss()

        model.train()
        for _epoch in range(20):  # reduced epochs for speed
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb.unsqueeze(1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).squeeze().numpy()
                all_preds.extend(pred if pred.ndim > 0 else [pred.item()])
                all_targets.extend(yb.numpy())

        if len(set(all_targets)) < 2:
            return 0.5

        return float(roc_auc_score(all_targets, all_preds))

"""Dataset management for outage prediction models.

Handles temporal splitting, class balancing, and conversion between
pandas DataFrames and PyTorch datasets.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


class OutageDataset:
    """Manages train/val/test splits with strict temporal ordering.

    Time-series data MUST be split temporally, never randomly, to
    prevent data leakage from temporal autocorrelation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "target_outage",
        timestamp_col: str = "timestamp",
        exclude_cols: tuple[str, ...] = ("h3_cell", "timestamp", "target_outage", "target_max_outage_fraction"),
    ):
        self.df = df.sort_values(timestamp_col).reset_index(drop=True)
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.exclude_cols = exclude_cols

        self.feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ("float64", "float32", "int64", "int32")
        ]

        self.scaler: StandardScaler | None = None

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def temporal_split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by time. No shuffling. No leakage.

        Returns:
            (train_df, val_df, test_df) in temporal order.
        """
        n = len(self.df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train = self.df.iloc[:train_end]
        val = self.df.iloc[train_end:val_end]
        test = self.df.iloc[val_end:]

        logger.info(
            "dataset.split",
            train=len(train),
            val=len(val),
            test=len(test),
            train_pos_rate=round(train[self.target_col].mean(), 4),
            test_pos_rate=round(test[self.target_col].mean(), 4),
        )

        return train, val, test

    def get_X_y(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and target vector.

        Args:
            df: DataFrame subset (train, val, or test).
            fit_scaler: If True, fit the scaler on this data (use for training only).

        Returns:
            (X, y) as numpy arrays.
        """
        available_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[available_cols].fillna(0).values.astype(np.float32)
        y = df[self.target_col].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)

        return X, y

    def get_feature_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get the feature columns as a DataFrame (for tree models that need names)."""
        available_cols = [c for c in self.feature_cols if c in df.columns]
        return df[available_cols].fillna(0)

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 24,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for temporal models.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            seq_len: Number of time steps per sequence.

        Returns:
            (X_seq, y_seq) where X_seq is (n_sequences, seq_len, n_features)
            and y_seq is (n_sequences,).
        """
        sequences = []
        targets = []
        for i in range(len(X) - seq_len):
            sequences.append(X[i : i + seq_len])
            targets.append(y[i + seq_len - 1])

        return np.array(sequences), np.array(targets)

    def get_class_weights(self, y: np.ndarray) -> dict[int, float]:
        """Compute class weights for imbalanced data."""
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        total = len(y)

        if n_pos == 0 or n_neg == 0:
            return {0: 1.0, 1: 1.0}

        return {
            0: total / (2 * n_neg),
            1: total / (2 * n_pos),
        }

    def summary(self) -> dict:
        """Dataset summary statistics."""
        return {
            "total_samples": len(self.df),
            "n_features": self.n_features,
            "positive_rate": float(self.df[self.target_col].mean()),
            "time_range": {
                "start": str(self.df[self.timestamp_col].min()),
                "end": str(self.df[self.timestamp_col].max()),
            },
            "feature_names": self.feature_cols,
        }

"""Temporal Deep Learning Models for Outage Prediction.

Implements LSTM and GRU architectures with multi-head self-attention
and MC Dropout support for uncertainty quantification.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for sliding-window time series sequences."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 24):
        self.seq_len = seq_len
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return x, y


class LSTMOutageModel(nn.Module):
    """LSTM with multi-head self-attention for outage prediction.

    Supports MC Dropout: when mc_dropout=True is passed to forward(),
    dropout remains active during inference for uncertainty estimation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self._dropout_rate = dropout

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        """Forward pass with optional MC Dropout for uncertainty.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            mc_dropout: If True, keeps dropout active during inference.

        Returns:
            Predictions of shape (batch, 1) in [0, 1].
        """
        if mc_dropout:
            self.train()

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)

        last_hidden = attn_out[:, -1, :]
        out = self.fc(last_hidden)
        return torch.sigmoid(out)


class GRUOutageModel(nn.Module):
    """GRU variant with the same architecture as LSTMOutageModel."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        if mc_dropout:
            self.train()

        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)

        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        attn_out = self.layer_norm(attn_out + gru_out)

        last_hidden = attn_out[:, -1, :]
        out = self.fc(last_hidden)
        return torch.sigmoid(out)


class TemporalModelTrainer:
    """Training loop for LSTM/GRU models with MLflow integration."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        grad_clip: float = 1.0,
        device: str | None = None,
    ):
        self.model = model
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )
        self.criterion = nn.BCELoss()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        mlflow_run=None,
    ) -> dict:
        """Train the model with early stopping and LR scheduling.

        Returns:
            Dict with training history (train_losses, val_losses, best_epoch).
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None
        history = {"train_losses": [], "val_losses": []}

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)

            # Validation
            self.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device).unsqueeze(1)
                    preds = self.model(x_batch)
                    loss = self.criterion(preds, y_batch)
                    val_loss += loss.item()
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            self.scheduler.step()

            history["train_losses"].append(avg_train_loss)
            history["val_losses"].append(avg_val_loss)

            if mlflow_run:
                try:
                    import mlflow

                    mlflow.log_metrics(
                        {"train_loss": avg_train_loss, "val_loss": avg_val_loss},
                        step=epoch,
                    )
                except Exception:
                    pass

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        if best_state:
            self.model.load_state_dict(best_state)

        history["best_epoch"] = len(history["train_losses"]) - self.patience
        history["best_val_loss"] = best_val_loss
        return history

    def predict(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """Generate predictions for the given data."""
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                preds = self.model(x_batch).cpu().numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds, axis=0).squeeze()

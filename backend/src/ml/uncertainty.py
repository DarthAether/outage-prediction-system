"""Uncertainty Quantification for Outage Predictions.

Implements dual-source uncertainty estimation:
1. MC Dropout: Multiple stochastic forward passes through the LSTM with
   dropout active, capturing model uncertainty in the neural component.
2. Ensemble Disagreement: Variance across ensemble member predictions,
   capturing epistemic uncertainty from model diversity.

The aleatoric/epistemic decomposition and post-hoc calibration via
isotonic regression form the second research contribution.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression


@dataclass
class UncertaintyPrediction:
    """Container for a prediction with full uncertainty information."""

    mean: float
    std: float
    aleatoric: float
    epistemic: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.90
    calibrated: bool = False


class UncertaintyEstimator:
    """Estimates prediction uncertainty from MC Dropout and ensemble disagreement.

    The key insight is that MC Dropout variance captures uncertainty in the
    neural model's learned representation, while ensemble disagreement captures
    uncertainty from structural model differences. Their combination yields
    more reliable confidence intervals than either source alone.
    """

    def __init__(self, n_mc_samples: int = 50, confidence_level: float = 0.90):
        self.n_mc_samples = n_mc_samples
        self.confidence_level = confidence_level
        self._calibrator: IsotonicRegression | None = None

    def mc_dropout_predictions(
        self,
        model,  # LSTMOutageModel
        x_sequential: "np.ndarray",
    ) -> np.ndarray:
        """Run the LSTM model n_mc_samples times with dropout active.

        Args:
            model: An LSTMOutageModel instance with mc_dropout support.
            x_sequential: Input tensor of shape (batch_size, seq_len, features).

        Returns:
            Array of shape (batch_size, n_mc_samples) with sampled predictions.
        """
        import torch

        model.train()
        x_tensor = torch.FloatTensor(x_sequential)
        samples = []

        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                preds = model(x_tensor, mc_dropout=True).squeeze(-1).numpy()
                samples.append(preds)

        model.eval()
        return np.column_stack(samples)

    def ensemble_predictions(
        self,
        models: dict[str, object],
        x_tabular: np.ndarray,
        x_sequential: np.ndarray | None = None,
    ) -> np.ndarray:
        """Get predictions from each ensemble member.

        Args:
            models: Dict mapping model name to model instance.
            x_tabular: Tabular features for tree models.
            x_sequential: Sequential features for LSTM (optional).

        Returns:
            Array of shape (n_samples, n_models) with each model's predictions.
        """
        all_preds = []

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                # Tree-based model
                proba = model.predict_proba(x_tabular)
                if proba.ndim == 2:
                    preds = proba[:, 1]
                else:
                    preds = proba
                all_preds.append(preds)
            elif x_sequential is not None:
                # Neural model - single forward pass
                import torch

                model.eval()
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x_sequential)
                    preds = model(x_tensor).squeeze(-1).numpy()
                all_preds.append(preds)

        return np.column_stack(all_preds)

    def predict_with_uncertainty(
        self,
        ensemble_preds: np.ndarray,
        mc_preds: np.ndarray | None = None,
    ) -> list[UncertaintyPrediction]:
        """Compute full uncertainty estimates by combining ensemble and MC Dropout.

        Uncertainty decomposition:
        - Aleatoric: Mean of per-sample MC Dropout variance (data noise)
        - Epistemic: Variance of ensemble member means (model structure)
        - Total: sqrt(aleatoric^2 + epistemic^2)

        Args:
            ensemble_preds: (n_samples, n_models) ensemble member predictions.
            mc_preds: (n_samples, n_mc_samples) MC dropout predictions, optional.

        Returns:
            List of UncertaintyPrediction objects, one per sample.
        """
        n_samples = ensemble_preds.shape[0]
        z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(self.confidence_level, 1.645)

        results = []
        for i in range(n_samples):
            ens_mean = float(np.mean(ensemble_preds[i]))
            epistemic = float(np.std(ensemble_preds[i]))

            if mc_preds is not None and mc_preds.shape[0] > i:
                aleatoric = float(np.std(mc_preds[i]))
            else:
                aleatoric = 0.0

            total_std = float(np.sqrt(aleatoric**2 + epistemic**2))

            mean_pred = ens_mean
            if mc_preds is not None:
                mc_mean = float(np.mean(mc_preds[i]))
                mean_pred = 0.6 * ens_mean + 0.4 * mc_mean

            if self._calibrator is not None:
                mean_pred = float(self._calibrator.predict([mean_pred])[0])

            ci_lower = float(np.clip(mean_pred - z * total_std, 0.0, 1.0))
            ci_upper = float(np.clip(mean_pred + z * total_std, 0.0, 1.0))
            mean_pred = float(np.clip(mean_pred, 0.0, 1.0))

            results.append(UncertaintyPrediction(
                mean=mean_pred,
                std=total_std,
                aleatoric=aleatoric,
                epistemic=epistemic,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence_level=self.confidence_level,
                calibrated=self._calibrator is not None,
            ))

        return results

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> "UncertaintyEstimator":
        """Fit post-hoc calibration using isotonic regression.

        Isotonic regression maps predicted probabilities to calibrated
        probabilities that better match observed frequencies. This is
        critical for uncertainty estimates to be trustworthy.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Raw predicted probabilities.

        Returns:
            self (for chaining).
        """
        self._calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._calibrator.fit(y_pred, y_true)
        return self

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        ECE measures how well predicted probabilities match observed
        frequencies. Lower is better; ECE < 0.10 is generally considered
        well-calibrated.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted probabilities.
            n_bins: Number of bins for calibration assessment.

        Returns:
            ECE value in [0, 1].
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(y_true)

        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])

            bin_count = mask.sum()
            if bin_count == 0:
                continue

            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            ece += (bin_count / total) * abs(bin_acc - bin_conf)

        return float(ece)

    @staticmethod
    def reliability_diagram_data(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 15,
    ) -> dict:
        """Generate data for plotting reliability diagrams.

        Returns:
            Dict with 'fraction_of_positives', 'mean_predicted_value',
            'bin_counts' for each bin.
        """
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy="uniform"
        )
        return {
            "fraction_of_positives": fraction_pos.tolist(),
            "mean_predicted_value": mean_pred.tolist(),
            "n_bins": n_bins,
        }

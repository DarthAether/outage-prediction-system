"""Model Registry for tracking and promoting trained models.

Supplements MLflow with region-specific model management,
active model selection, and deployment tracking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import structlog
import torch

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Manages trained model artifacts, versioning, and deployment state."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._active_models: dict[str, dict] = {}

    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        region: str,
        metrics: dict,
        model_type: str = "tabular",
    ) -> str:
        """Save a trained model to disk.

        Args:
            model: Trained model instance.
            model_name: e.g., 'xgboost', 'lightgbm', 'lstm'.
            version: Version string, e.g., 'v1.0.0'.
            region: Region code, e.g., 'TX'.
            metrics: Dict of evaluation metrics.
            model_type: 'tabular' or 'sequential'.

        Returns:
            Path to saved model.
        """
        save_dir = self.model_dir / region / model_name / version
        save_dir.mkdir(parents=True, exist_ok=True)

        if model_type == "sequential":
            model_path = save_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
        else:
            model_path = save_dir / "model.pkl"
            joblib.dump(model, model_path)

        metadata = {
            "model_name": model_name,
            "version": version,
            "region": region,
            "model_type": model_type,
            "metrics": metrics,
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "registry.saved",
            model=model_name,
            version=version,
            region=region,
            path=str(model_path),
        )
        return str(model_path)

    def load_model(
        self,
        model_name: str,
        version: str,
        region: str,
        model_class: type | None = None,
        model_kwargs: dict | None = None,
    ) -> Any:
        """Load a model from disk.

        Args:
            model_name: Model name.
            version: Version string.
            region: Region code.
            model_class: For PyTorch models, the class to instantiate.
            model_kwargs: Constructor kwargs for PyTorch models.

        Returns:
            Loaded model instance.
        """
        save_dir = self.model_dir / region / model_name / version

        pt_path = save_dir / "model.pt"
        pkl_path = save_dir / "model.pkl"

        if pt_path.exists() and model_class is not None:
            model = model_class(**(model_kwargs or {}))
            model.load_state_dict(torch.load(pt_path, weights_only=True))
            model.eval()
            return model
        elif pkl_path.exists():
            return joblib.load(pkl_path)
        else:
            raise FileNotFoundError(f"No model found at {save_dir}")

    def promote(self, model_name: str, version: str, region: str) -> None:
        """Set a model version as the active model for a region."""
        key = f"{region}:{model_name}"
        self._active_models[key] = {
            "model_name": model_name,
            "version": version,
            "region": region,
            "promoted_at": datetime.utcnow().isoformat(),
        }
        logger.info("registry.promoted", model=model_name, version=version, region=region)

    def get_active_model(self, model_name: str, region: str) -> dict | None:
        """Get the currently active model info for a region."""
        key = f"{region}:{model_name}"
        return self._active_models.get(key)

    def list_models(self, region: str | None = None) -> list[dict]:
        """List all saved models, optionally filtered by region."""
        models = []
        search_dir = self.model_dir / region if region else self.model_dir

        if not search_dir.exists():
            return models

        for metadata_path in search_dir.rglob("metadata.json"):
            with open(metadata_path) as f:
                meta = json.load(f)
            if region is None or meta.get("region") == region:
                models.append(meta)

        return models

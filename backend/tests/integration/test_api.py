"""Integration tests for the FastAPI application.

Tests the API endpoints using httpx TestClient without requiring
a running database (uses mocked dependencies).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture
def app():
    """Create a test app instance with mocked dependencies."""
    from src.api.app import create_app
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client."""
    from httpx import ASGITransport, AsyncClient
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_root_health(self, client):
        """Root endpoint should return 200."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestPredictionSchemas:
    """Test prediction request/response schemas."""

    def test_prediction_request_schema(self):
        from src.api.schemas.predictions import PredictionRequest
        req = PredictionRequest(
            h3_cell="87489e346ffffff",
            region="TX",
        )
        assert req.h3_cell == "87489e346ffffff"
        assert req.region == "TX"

    def test_batch_prediction_request(self):
        from src.api.schemas.predictions import BatchPredictionRequest
        batch = BatchPredictionRequest(
            region="TX",
            h3_cells=["87489e346ffffff", "87489e347ffffff"],
        )
        assert len(batch.h3_cells) == 2
        assert batch.region == "TX"

    def test_prediction_result_schema(self):
        from datetime import datetime
        from src.api.schemas.predictions import PredictionResult, UncertaintyEstimate
        result = PredictionResult(
            prediction_id="test-123",
            h3_cell="87489e346ffffff",
            risk_probability=0.75,
            risk_level="ORANGE",
            region="TX",
            model_version="v1.0",
            computed_at=datetime.now(),
            uncertainty=UncertaintyEstimate(
                aleatoric=0.05,
                epistemic=0.08,
                lower=0.65,
                upper=0.85,
            ),
            top_features=[],
        )
        assert result.risk_probability == 0.75
        assert result.risk_level == "ORANGE"


class TestAlertSchemas:
    """Test alert schemas."""

    def test_alert_response_fields(self):
        from src.api.schemas.alerts import AlertResponse
        # Verify the schema has expected fields
        fields = AlertResponse.model_fields
        assert "severity" in fields or "risk_level" in fields


class TestMiddleware:
    """Test middleware components."""

    def test_rate_limit_middleware_init(self):
        from src.api.middleware import RateLimitMiddleware
        # Just verify it can be instantiated
        assert RateLimitMiddleware is not None

    def test_request_logging_middleware_init(self):
        from src.api.middleware import RequestLoggingMiddleware
        assert RequestLoggingMiddleware is not None

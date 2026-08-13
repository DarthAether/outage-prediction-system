from src.api.v1.admin import router as admin_router
from src.api.v1.alerts import router as alerts_router
from src.api.v1.health import router as health_router
from src.api.v1.historical import router as historical_router
from src.api.v1.predictions import router as predictions_router

__all__ = [
    "admin_router",
    "alerts_router",
    "health_router",
    "historical_router",
    "predictions_router",
]

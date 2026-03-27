from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class HealthResponse(BaseModel):
    status: str = Field(description="Overall system status")
    db_connected: bool = Field(description="Database connectivity")
    redis_connected: bool = Field(description="Redis connectivity")
    model_loaded: bool = Field(description="Whether the ensemble model is loaded")
    active_models: list[str] = Field(default_factory=list)
    uptime_seconds: float = Field(description="Seconds since server start")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    per_page: int = Field(ge=1)

    @property
    def total_pages(self) -> int:
        return max(1, -(-self.total // self.per_page))

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None

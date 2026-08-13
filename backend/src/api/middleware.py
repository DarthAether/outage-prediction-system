from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status code, and duration."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            status = response.status_code if response else 500
            logger.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=status,
                duration_ms=round(duration_ms, 2),
                client=request.client.host if request.client else "unknown",
            )


class _TokenBucket:
    """Simple token bucket for a single client."""

    __slots__ = ("capacity", "tokens", "refill_rate", "_last_refill")

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self._last_refill = time.monotonic()

    def consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory per-client-IP token bucket rate limiter.

    Args:
        app: ASGI application.
        requests_per_second: Sustained request rate per client.
        burst: Maximum burst size (bucket capacity).
    """

    def __init__(self, app, requests_per_second: float = 10.0, burst: int = 30) -> None:
        super().__init__(app)
        self.requests_per_second = requests_per_second
        self.burst = burst
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(capacity=burst, refill_rate=requests_per_second)
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "0.0.0.0"
        bucket = self._buckets[client_ip]
        if not bucket.consume():
            logger.warning("rate_limit.exceeded", client=client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
                headers={"Retry-After": str(int(1 / self.requests_per_second))},
            )
        return await call_next(request)

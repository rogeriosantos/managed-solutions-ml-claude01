"""
Custom middleware for logging, rate limiting, and request/response handling.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
import asyncio
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# In-memory store for rate limiting (fallback)
_rate_limit_store: Dict[str, deque] = defaultdict(deque)
_request_metrics: Dict[str, Any] = defaultdict(
    lambda: {
        "count": 0,
        "total_time": 0.0,
        "errors": 0,
        "last_reset": datetime.utcnow(),
    }
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""

    def __init__(self, app):
        super().__init__(app)
        self.sensitive_headers = {"authorization", "x-api-key", "cookie"}
        self.max_body_size = 1024 * 10  # 10KB max for logging

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()

        # Generate request ID
        request_id = f"req_{int(time.time() * 1000)}_{id(request)}"

        # Log request
        await self._log_request(request, request_id)

        # Process request
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time

            # Log response
            await self._log_response(response, request_id, processing_time)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Request {request_id} failed: {str(e)} | "
                f"Method: {request.method} | URL: {request.url} | "
                f"Time: {processing_time:.3f}s"
            )
            raise

    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details."""
        try:
            # Get client info
            client_host = (
                getattr(request.client, "host", "unknown")
                if request.client
                else "unknown"
            )
            user_agent = request.headers.get("user-agent", "unknown")

            # Filter sensitive headers
            headers = {
                k: v if k.lower() not in self.sensitive_headers else "[REDACTED]"
                for k, v in request.headers.items()
            }

            # Get request body (if small enough)
            body_info = "not_logged"
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if len(body) <= self.max_body_size:
                        body_info = f"size:{len(body)}"
                    else:
                        body_info = f"size:{len(body)}_truncated"
                except Exception:
                    body_info = "read_error"

            logger.info(
                f"Request {request_id} started | "
                f"Method: {request.method} | URL: {request.url} | "
                f"Client: {client_host} | Agent: {user_agent[:50]}... | "
                f"Headers: {len(headers)} | Body: {body_info}"
            )

        except Exception as e:
            logger.error(f"Error logging request {request_id}: {e}")

    async def _log_response(
        self, response: Response, request_id: str, processing_time: float
    ):
        """Log response details."""
        try:
            logger.info(
                f"Request {request_id} completed | "
                f"Status: {response.status_code} | "
                f"Time: {processing_time:.3f}s | "
                f"Headers: {len(response.headers)}"
            )

            # Log slow requests
            if processing_time > 2.0:
                logger.warning(
                    f"Slow request detected: {request_id} | "
                    f"Processing time: {processing_time:.3f}s"
                )

        except Exception as e:
            logger.error(f"Error logging response for {request_id}: {e}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for API rate limiting with Redis backend."""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
        redis_client: Optional[redis.Redis] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.redis_client = redis_client
        self.window_size = 60  # seconds

        # Exempt paths from rate limiting
        self.exempt_paths = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        is_allowed, retry_after = await self._check_rate_limit(client_id)

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(retry_after))},
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining, reset_time = await self._get_rate_limit_info(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Try to get API key first
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key:{api_key[:8]}..."

        # Fall back to IP address
        if request.client and request.client.host:
            return f"ip:{request.client.host}"

        # Ultimate fallback
        return "unknown_client"

    async def _check_rate_limit(self, client_id: str) -> tuple[bool, float]:
        """Check if client is within rate limits."""
        current_time = time.time()

        if self.redis_client:
            return await self._check_rate_limit_redis(client_id, current_time)
        else:
            return self._check_rate_limit_memory(client_id, current_time)

    async def _check_rate_limit_redis(
        self, client_id: str, current_time: float
    ) -> tuple[bool, float]:
        """Redis-based rate limiting with sliding window."""
        try:
            pipe = self.redis_client.pipeline()
            key = f"rate_limit:{client_id}"

            # Remove old entries
            cutoff = current_time - self.window_size
            pipe.zremrangebyscore(key, 0, cutoff)

            # Count current requests
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(current_time): current_time})

            # Set expiration
            pipe.expire(key, self.window_size + 10)

            results = pipe.execute()
            request_count = results[1]

            if request_count >= self.requests_per_minute:
                # Get oldest request time to calculate retry after
                oldest_requests = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_requests:
                    oldest_time = oldest_requests[0][1]
                    retry_after = self.window_size - (current_time - oldest_time)
                    return False, max(retry_after, 1)

            return True, 0

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fallback to memory-based rate limiting
            return self._check_rate_limit_memory(client_id, current_time)

    def _check_rate_limit_memory(
        self, client_id: str, current_time: float
    ) -> tuple[bool, float]:
        """Memory-based rate limiting (fallback)."""
        requests = _rate_limit_store[client_id]

        # Remove old requests
        cutoff_time = current_time - self.window_size
        while requests and requests[0] < cutoff_time:
            requests.popleft()

        # Check limits
        if len(requests) >= self.requests_per_minute:
            retry_after = self.window_size - (current_time - requests[0])
            return False, max(retry_after, 1)

        # Add current request
        requests.append(current_time)
        return True, 0

    async def _get_rate_limit_info(self, client_id: str) -> tuple[int, float]:
        """Get current rate limit status."""
        current_time = time.time()

        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}"
                count = self.redis_client.zcard(key)
                remaining = max(0, self.requests_per_minute - count)
                reset_time = current_time + self.window_size
                return remaining, reset_time
            except Exception:
                pass

        # Memory fallback
        requests = _rate_limit_store[client_id]
        remaining = max(0, self.requests_per_minute - len(requests))
        reset_time = current_time + self.window_size
        return remaining, reset_time


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics."""

    def __init__(self, app):
        super().__init__(app)
        self.reset_interval = 300  # Reset metrics every 5 minutes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"

        try:
            response = await call_next(request)
            processing_time = time.time() - start_time

            # Update metrics
            self._update_metrics(endpoint, processing_time, response.status_code)

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(endpoint, processing_time, 500, error=True)
            raise

    def _update_metrics(
        self,
        endpoint: str,
        processing_time: float,
        status_code: int,
        error: bool = False,
    ):
        """Update endpoint metrics."""
        current_time = datetime.utcnow()
        metrics = _request_metrics[endpoint]

        # Reset metrics if needed
        if (current_time - metrics["last_reset"]).total_seconds() > self.reset_interval:
            metrics.update(
                {"count": 0, "total_time": 0.0, "errors": 0, "last_reset": current_time}
            )

        # Update metrics
        metrics["count"] += 1
        metrics["total_time"] += processing_time
        if error or status_code >= 400:
            metrics["errors"] += 1

        # Add additional metrics
        if "min_time" not in metrics or processing_time < metrics["min_time"]:
            metrics["min_time"] = processing_time
        if "max_time" not in metrics or processing_time > metrics["max_time"]:
            metrics["max_time"] = processing_time

        metrics["avg_time"] = metrics["total_time"] / metrics["count"]
        metrics["error_rate"] = metrics["errors"] / metrics["count"]
        metrics["last_status"] = status_code
        metrics["last_updated"] = current_time


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and basic protection."""

    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value

        return response


def get_metrics_summary() -> Dict[str, Any]:
    """Get current metrics summary."""
    summary = {
        "endpoints": {},
        "total_requests": 0,
        "total_errors": 0,
        "collection_period_minutes": 5,
    }

    for endpoint, metrics in _request_metrics.items():
        summary["endpoints"][endpoint] = {
            "requests": metrics["count"],
            "avg_response_time": round(metrics.get("avg_time", 0), 3),
            "min_response_time": round(metrics.get("min_time", 0), 3),
            "max_response_time": round(metrics.get("max_time", 0), 3),
            "error_count": metrics["errors"],
            "error_rate": round(metrics.get("error_rate", 0), 3),
            "last_status": metrics.get("last_status", "unknown"),
            "last_updated": metrics.get("last_updated", datetime.utcnow()).isoformat(),
        }

        summary["total_requests"] += metrics["count"]
        summary["total_errors"] += metrics["errors"]

    if summary["total_requests"] > 0:
        summary["overall_error_rate"] = round(
            summary["total_errors"] / summary["total_requests"], 3
        )
    else:
        summary["overall_error_rate"] = 0.0

    return summary


async def reset_metrics():
    """Reset all collected metrics."""
    global _request_metrics
    _request_metrics.clear()
    logger.info("API metrics reset")

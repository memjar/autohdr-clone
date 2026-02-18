"""
Structured JSON Logging Middleware for AXE Backend
===================================================
Request ID tracking, structured logs, rotation.
"""

import json
import logging
import os
import time
import uuid
from logging.handlers import RotatingFileHandler
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


LOG_DIR = os.path.expanduser("~/.axe/logs")
os.makedirs(LOG_DIR, exist_ok=True)

# JSON structured logger
_logger = logging.getLogger("axe.requests")
_logger.setLevel(logging.INFO)
_logger.propagate = False

# Rotating file handler: 10MB max, 5 backups
_file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "backend.jsonl"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_file_handler)

# Error-only logger for quick scanning
_error_logger = logging.getLogger("axe.errors")
_error_logger.setLevel(logging.ERROR)
_error_logger.propagate = False
_error_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "errors.jsonl"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_error_handler.setFormatter(logging.Formatter("%(message)s"))
_error_logger.addHandler(_error_handler)


# In-memory metrics for /metrics endpoint
_metrics = {
    "total_requests": 0,
    "errors_4xx": 0,
    "errors_5xx": 0,
    "by_endpoint": {},
    "latencies": [],  # last 1000 request durations in ms
    "started_at": time.time(),
}
_MAX_LATENCIES = 1000


def get_metrics() -> dict:
    """Return current request metrics."""
    latencies = _metrics["latencies"]
    sorted_lat = sorted(latencies) if latencies else [0]
    p50_idx = int(len(sorted_lat) * 0.5)
    p95_idx = int(len(sorted_lat) * 0.95)
    p99_idx = int(len(sorted_lat) * 0.99)

    return {
        "total_requests": _metrics["total_requests"],
        "errors_4xx": _metrics["errors_4xx"],
        "errors_5xx": _metrics["errors_5xx"],
        "uptime_seconds": round(time.time() - _metrics["started_at"]),
        "latency_ms": {
            "p50": round(sorted_lat[p50_idx], 1),
            "p95": round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 1),
            "p99": round(sorted_lat[min(p99_idx, len(sorted_lat) - 1)], 1),
        },
        "top_endpoints": dict(
            sorted(_metrics["by_endpoint"].items(), key=lambda x: x[1], reverse=True)[:10]
        ),
    }


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request as structured JSON with request ID and duration."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        start = time.time()

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.time() - start) * 1000, 1)
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rid": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": 500,
                "ms": duration_ms,
                "error": str(exc)[:200],
                "ip": request.client.host if request.client else "unknown",
            }
            _logger.info(json.dumps(entry))
            _error_logger.error(json.dumps(entry))
            _metrics["total_requests"] += 1
            _metrics["errors_5xx"] += 1
            raise

        duration_ms = round((time.time() - start) * 1000, 1)
        status = response.status_code

        # Skip noisy health checks from logs
        path = request.url.path
        if path not in ("/health", "/ollama-health", "/favicon.ico"):
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rid": request_id,
                "method": request.method,
                "path": path,
                "status": status,
                "ms": duration_ms,
                "ip": request.client.host if request.client else "unknown",
            }
            _logger.info(json.dumps(entry))

            if status >= 500:
                _error_logger.error(json.dumps(entry))

        # Update metrics
        _metrics["total_requests"] += 1
        if 400 <= status < 500:
            _metrics["errors_4xx"] += 1
        elif status >= 500:
            _metrics["errors_5xx"] += 1

        endpoint_key = f"{request.method} {path}"
        _metrics["by_endpoint"][endpoint_key] = _metrics["by_endpoint"].get(endpoint_key, 0) + 1

        _metrics["latencies"].append(duration_ms)
        if len(_metrics["latencies"]) > _MAX_LATENCIES:
            _metrics["latencies"] = _metrics["latencies"][-_MAX_LATENCIES:]

        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        return response

"""
Structured Error Handler Middleware
====================================
Returns consistent JSON errors with request IDs.
Catches unhandled exceptions that would otherwise return raw 500s.
"""

import json
import time
import traceback
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catches unhandled exceptions → structured JSON error response."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            request_id = getattr(request.state, "request_id", "unknown")
            tb = traceback.format_exc()

            # Log to stderr for uvicorn capture
            print(f"[ERROR] {request_id} {request.method} {request.url.path}: {exc}")
            print(tb)

            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": str(exc)[:500],
                    "request_id": request_id,
                    "path": request.url.path,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )

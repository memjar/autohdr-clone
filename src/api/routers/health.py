"""
Health & Observability Router
==============================
/health (deep), /metrics, /security, /ollama-health
"""

import shutil
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Deep health check — probes Ollama, disk, ngrok."""
    # Lazy import to get current state from main module
    from src.api.main import (
        HAS_PROCESSOR, PROCESSOR_VERSION, HAS_RAWPY, RAWPY_VERSION,
        CV2_VERSION, HAS_SECURITY, HAS_PRO_PROCESSOR, HAS_AI_PROCESSOR,
    )

    ollama_ok = False
    ollama_models = 0
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                ollama_models = len(resp.json().get("models", []))
    except Exception:
        pass

    disk = shutil.disk_usage("/")
    disk_free_gb = round(disk.free / (1024 ** 3), 1)
    disk_ok = disk_free_gb > 5

    ngrok_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://127.0.0.1:4040/api/tunnels")
            if resp.status_code == 200:
                ngrok_ok = len(resp.json().get("tunnels", [])) > 0
    except Exception:
        pass

    all_ok = ollama_ok and disk_ok
    return {
        "status": "healthy" if all_ok else "degraded",
        "components": {
            "ollama": {"healthy": ollama_ok, "models_loaded": ollama_models},
            "disk": {"healthy": disk_ok, "free_gb": disk_free_gb},
            "ngrok": {"healthy": ngrok_ok},
            "processor": {"installed": HAS_PROCESSOR, "version": PROCESSOR_VERSION},
            "rawpy": {"installed": HAS_RAWPY, "version": RAWPY_VERSION},
            "opencv": {"installed": True, "version": CV2_VERSION},
            "security": {"enabled": HAS_SECURITY},
        },
        "quality_level": "95%+ AutoHDR" if HAS_PRO_PROCESSOR else ("90% AutoHDR" if HAS_AI_PROCESSOR else "60% AutoHDR"),
    }


@router.get("/metrics")
async def metrics():
    """Request metrics — counts, latency percentiles, top endpoints."""
    try:
        from src.api.middleware.logging import get_metrics
        return get_metrics()
    except ImportError:
        return {"error": "observability not loaded"}


@router.get("/ollama-health")
async def ollama_health():
    """Check if Ollama is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                return {"status": "healthy", "ollama": True}
    except Exception:
        pass
    return {"status": "unhealthy", "ollama": False}


@router.get("/security")
async def security_status():
    """Security configuration status."""
    from src.api.main import HAS_SECURITY
    if HAS_SECURITY:
        from src.api.security_middleware import ALLOWED_ORIGINS, rate_limiter
        return {
            "status": "enabled",
            "cors": {"mode": "whitelist", "allowed_origins": ALLOWED_ORIGINS, "credentials": True},
            "rate_limiting": {"enabled": True, "limits": rate_limiter.limits},
            "headers": {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
                "Strict-Transport-Security": "max-age=31536000 (HTTPS only)",
            },
        }
    return {
        "status": "disabled",
        "warning": "Security middleware not loaded. Running with permissive settings.",
        "cors": {"mode": "permissive", "allows_all": True},
        "rate_limiting": {"enabled": False},
        "headers": {"enabled": False},
    }

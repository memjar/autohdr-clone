"""Audit log endpoints — query volume, popular datasets, response times."""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from imi_audit import get_audit_log, get_audit_stats

logger = logging.getLogger("imi_audit_router")

router = APIRouter(prefix="/klaus/imi", tags=["imi-audit"])


@router.get("/audit")
async def audit_log(limit: int = 100, offset: int = 0):
    """Return recent audit log entries."""
    entries = get_audit_log(limit=min(limit, 1000), offset=offset)
    return JSONResponse({"entries": entries, "count": len(entries)})


@router.get("/audit/stats")
async def audit_stats():
    """Return audit statistics — query volume, popular datasets, avg response time."""
    stats = get_audit_stats()
    return JSONResponse(stats)

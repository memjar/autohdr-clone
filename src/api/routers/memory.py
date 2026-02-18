"""
IMI Memory Router — /klaus/imi/memory/* endpoints.

6 endpoints for querying and managing the team memory engine.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from imi_memory import (
    get_insights,
    get_recurring_patterns,
    search_memory,
    get_stats,
    add_feedback,
    get_context_for_query,
)

logger = logging.getLogger("imi-memory")

router = APIRouter(prefix="/klaus/imi", tags=["imi-memory"])


class FeedbackRequest(BaseModel):
    memory_id: str
    feedback: str


@router.get("/memory/insights")
async def memory_insights(limit: int = Query(20, ge=1, le=100)):
    """Return recent memories/insights."""
    return {"insights": get_insights(limit=limit)}


@router.get("/memory/patterns")
async def memory_patterns():
    """Return recurring patterns analysis."""
    return get_recurring_patterns()


@router.get("/memory/search")
async def memory_search(q: str = Query(..., min_length=1)):
    """Search past analyses by keyword."""
    results = search_memory(q)
    return {"query": q, "results": results, "count": len(results)}


@router.get("/memory/stats")
async def memory_stats():
    """Return memory statistics."""
    return get_stats()


@router.post("/memory/feedback")
async def memory_feedback(req: FeedbackRequest):
    """Add user feedback to a memory entry."""
    ok = add_feedback(req.memory_id, req.feedback)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory entry not found")
    return {"status": "ok", "memory_id": req.memory_id}


@router.get("/memory/context")
async def memory_context(q: str = Query(..., min_length=1)):
    """Get relevant past context for a new query."""
    results = get_context_for_query(q)
    return {"query": q, "context": results, "count": len(results)}

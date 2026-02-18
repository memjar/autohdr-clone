"""
Case Studies Router — browse, search, and ingest IMI case studies.
"""

import logging
from collections import Counter
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from imi_case_studies import load_studies, search_studies, get_study, add_studies

logger = logging.getLogger("imi_case_studies_router")

router = APIRouter(prefix="/klaus/imi", tags=["imi-case-studies"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CaseStudy(BaseModel):
    id: str
    title: str
    client: str
    industry: str
    methodology: str
    year: int
    pillar: str
    summary: str
    key_findings: list[str]


class PaginatedResponse(BaseModel):
    total: int
    skip: int
    limit: int
    studies: list[dict]


class IngestResponse(BaseModel):
    added: int
    total: int


class StatsResponse(BaseModel):
    total: int
    by_pillar: dict[str, int]
    by_industry: dict[str, int]
    by_methodology: dict[str, int]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/case-studies", response_model=PaginatedResponse)
def list_case_studies(skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
    """List all case studies with pagination."""
    studies = load_studies()
    return PaginatedResponse(
        total=len(studies),
        skip=skip,
        limit=limit,
        studies=studies[skip : skip + limit],
    )


@router.get("/case-studies/search")
def search_case_studies(q: str = Query(..., min_length=1), n: int = Query(5, ge=1, le=50)):
    """Keyword search across case studies."""
    results = search_studies(q, n=n)
    return {"query": q, "count": len(results), "results": results}


@router.get("/case-studies/stats", response_model=StatsResponse)
def case_study_stats():
    """Aggregate counts by pillar, industry, and methodology."""
    studies = load_studies()
    return StatsResponse(
        total=len(studies),
        by_pillar=dict(Counter(s.get("pillar", "unknown") for s in studies)),
        by_industry=dict(Counter(s.get("industry", "unknown") for s in studies)),
        by_methodology=dict(Counter(s.get("methodology", "unknown") for s in studies)),
    )


@router.get("/case-studies/{study_id}")
def get_case_study(study_id: str):
    """Retrieve a single case study by ID."""
    study = get_study(study_id)
    if not study:
        raise HTTPException(status_code=404, detail=f"Case study {study_id} not found")
    return study


@router.post("/case-studies/ingest", response_model=IngestResponse)
def ingest_case_studies(studies: list[dict]):
    """Bulk ingest case studies from a JSON array."""
    count = add_studies(studies)
    total = len(load_studies())
    return IngestResponse(added=count, total=total)

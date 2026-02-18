"""
Search-first interface for Klaus IMI.
Instant RAG retrieval + autocomplete + entity index — NO LLM calls.
"""

import csv
import logging
import os

from fastapi import APIRouter, Query

from config import IMI_DATA_DIR
from imi_rag import query as rag_query

logger = logging.getLogger("imi_search")

router = APIRouter(prefix="/klaus/imi", tags=["imi-search"])

# ---------------------------------------------------------------------------
# Entity index — built lazily on first call, cached in module-level dict
# ---------------------------------------------------------------------------

_entity_index: dict[str, set[str]] = {}  # category -> set of names

_BRAND_COLUMNS = {"brand", "Brand", "brand_name", "Brand_Name", "brand_name_", "BRAND"}


def _build_entity_index() -> dict[str, set[str]]:
    """Scan all CSVs: extract headers as metrics, unique values from brand columns as brands, filenames as datasets."""
    global _entity_index
    if _entity_index:
        return _entity_index

    brands: set[str] = set()
    metrics: set[str] = set()
    datasets: set[str] = set()

    if not os.path.isdir(IMI_DATA_DIR):
        logger.warning("IMI_DATA_DIR not found: %s", IMI_DATA_DIR)
        return _entity_index

    for fname in sorted(os.listdir(IMI_DATA_DIR)):
        if not fname.lower().endswith(".csv"):
            continue
        datasets.add(os.path.splitext(fname)[0])
        path = os.path.join(IMI_DATA_DIR, fname)
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    metrics.update(reader.fieldnames)
                    # Find brand columns present in this file
                    brand_cols = _BRAND_COLUMNS & set(reader.fieldnames)
                    if brand_cols:
                        for row in reader:
                            for col in brand_cols:
                                val = row.get(col, "").strip()
                                if val:
                                    brands.add(val)
        except Exception as e:
            logger.error("Failed to index %s: %s", fname, e)

    _entity_index["brands"] = brands
    _entity_index["metrics"] = metrics
    _entity_index["datasets"] = datasets
    logger.info(
        "Entity index built: %d brands, %d metrics, %d datasets",
        len(brands), len(metrics), len(datasets),
    )
    return _entity_index


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/search")
def search(q: str = Query(..., min_length=1, description="Search query")):
    """Instant RAG search — retrieval only, no LLM call."""
    results = rag_query(q, n=10)
    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "dataset": r.get("dataset", ""),
                "category": r.get("category", "general"),
                "score": r.get("score", 0.0),
                "snippet": r.get("text", "")[:300],
                "text": r.get("text", ""),
            }
            for r in results
        ],
    }


@router.get("/search/suggest")
def suggest(q: str = Query(..., min_length=1, description="Autocomplete prefix")):
    """Autocomplete suggestions from entity index."""
    index = _build_entity_index()
    q_lower = q.lower()
    matches: list[dict] = []

    for category, entities in index.items():
        for entity in entities:
            if q_lower in entity.lower():
                matches.append({"entity": entity, "category": category})
            if len(matches) >= 20:
                break
        if len(matches) >= 20:
            break

    return {"query": q, "suggestions": matches}


@router.get("/search/entities")
def list_entities():
    """List all known entities: brands, metrics, datasets."""
    index = _build_entity_index()
    return {
        "brands": sorted(index.get("brands", set())),
        "metrics": sorted(index.get("metrics", set())),
        "datasets": sorted(index.get("datasets", set())),
    }

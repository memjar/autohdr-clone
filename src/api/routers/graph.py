"""
Knowledge Graph endpoints for Klaus IMI.
Exposes the NetworkX-based knowledge graph via REST API.
"""

import logging

from fastapi import APIRouter, Query

from imi_knowledge_graph import get_entity, get_connections, find_path, get_stats, search_nodes

logger = logging.getLogger("imi_graph")

router = APIRouter(prefix="/klaus/imi", tags=["imi-graph"])


@router.get("/graph/stats")
def graph_stats():
    """Graph statistics: node counts by type, edge counts, components."""
    return get_stats()


@router.get("/graph/search")
def graph_search(q: str = Query(..., min_length=1, description="Search substring")):
    """Search nodes by name substring."""
    results = search_nodes(q)
    return {"query": q, "count": len(results), "results": results}


@router.get("/graph/path")
def graph_path(
    from_entity: str = Query(..., description="Source entity name"),
    to_entity: str = Query(..., description="Target entity name"),
):
    """Shortest path between two entities."""
    return find_path(from_entity, to_entity)


@router.get("/graph/entity/{name}")
def graph_entity(name: str, depth: int = Query(1, ge=0, le=3)):
    """Get entity details and connections up to `depth` hops."""
    entity = get_entity(name)
    if entity is None:
        return {"error": f"Entity '{name}' not found"}
    if depth > 1:
        expanded = get_connections(name, depth=depth)
        if expanded:
            entity["graph"] = expanded
    return entity

"""
Knowledge Graph module for Klaus IMI.
Builds a NetworkX graph from CSV data files — brands, metrics, datasets,
methodologies, and industries as nodes with typed edges.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any

import networkx as nx

from config import IMI_DATA_DIR

logger = logging.getLogger("imi_knowledge_graph")

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_graph: nx.Graph | None = None

# Heuristic keyword lists for node-type inference
_METHODOLOGY_KEYWORDS = {
    "nps", "csat", "ces", "top2box", "likert", "index", "score",
    "satisfaction", "loyalty", "awareness", "consideration", "preference",
    "intent", "equity", "perception", "sentiment", "recommendation",
}

_INDUSTRY_KEYWORDS = {
    "automotive", "retail", "tech", "technology", "finance", "financial",
    "healthcare", "telecom", "telecommunications", "fmcg", "cpg",
    "insurance", "banking", "energy", "media", "travel", "hospitality",
    "pharma", "pharmaceutical", "apparel", "food", "beverage",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    return headers, rows


def _is_numeric(val: str) -> bool:
    try:
        float(val.replace(",", "").replace("%", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def _classify_header(h: str) -> str:
    """Guess whether a header represents a metric or methodology."""
    lower = h.lower().strip()
    for kw in _METHODOLOGY_KEYWORDS:
        if kw in lower:
            return "methodology"
    return "metric"


def _extract_industry(filename: str) -> str | None:
    lower = filename.lower()
    for kw in _INDUSTRY_KEYWORDS:
        if kw in lower:
            return kw.title()
    return None


def _ensure_node(g: nx.Graph, name: str, ntype: str) -> None:
    key = name.lower().strip()
    if not key:
        return
    if key not in g:
        g.add_node(key, label=name.strip(), type=ntype)


def _ensure_edge(g: nx.Graph, src: str, dst: str, rel: str) -> None:
    s, d = src.lower().strip(), dst.lower().strip()
    if s and d and s != d:
        if not g.has_edge(s, d):
            g.add_edge(s, d, relation=rel)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_from_csv(g: nx.Graph, path: str) -> None:
    fname = os.path.basename(path)
    dataset_name = os.path.splitext(fname)[0]
    headers, rows = _load_csv(path)
    if not headers or not rows:
        return

    # Dataset node
    _ensure_node(g, dataset_name, "dataset")

    # Industry from filename
    industry = _extract_industry(fname)
    if industry:
        _ensure_node(g, industry, "industry")
        _ensure_edge(g, dataset_name, industry, "belongs_to_industry")

    # Identify brand column (first non-numeric column) vs metric columns
    brand_col_idx: int | None = None
    metric_cols: list[tuple[int, str]] = []

    for i, h in enumerate(headers):
        # Check if majority of values in this column are non-numeric → brand col
        sample = [rows[r][i] for r in range(min(5, len(rows))) if i < len(rows[r])]
        numeric_count = sum(1 for v in sample if _is_numeric(v))
        if brand_col_idx is None and numeric_count < len(sample) / 2:
            brand_col_idx = i
        else:
            htype = _classify_header(h)
            metric_cols.append((i, h))
            _ensure_node(g, h.strip(), htype)
            _ensure_edge(g, dataset_name, h.strip(), "has_metric")

    # Extract brands
    brands_in_dataset: list[str] = []
    if brand_col_idx is not None:
        for row in rows:
            if brand_col_idx < len(row):
                brand = row[brand_col_idx].strip()
                if brand and not _is_numeric(brand):
                    _ensure_node(g, brand, "brand")
                    _ensure_edge(g, brand, dataset_name, "measured_in")
                    brands_in_dataset.append(brand)
                    # Link brand to metrics
                    for mi, mh in metric_cols:
                        _ensure_edge(g, brand, mh.strip(), "has_metric")

    # related_to edges between brands in the same dataset
    for i, b1 in enumerate(brands_in_dataset):
        for b2 in brands_in_dataset[i + 1:]:
            _ensure_edge(g, b1, b2, "related_to")

    # competes_with: brands sharing many metrics → competitors
    if len(brands_in_dataset) >= 2 and len(metric_cols) >= 2:
        for i, b1 in enumerate(brands_in_dataset):
            for b2 in brands_in_dataset[i + 1:]:
                _ensure_edge(g, b1, b2, "competes_with")


def build_graph(data_dir: str | None = None) -> nx.Graph:
    """Build the full knowledge graph from all CSV files in data_dir."""
    global _graph
    data_dir = data_dir or IMI_DATA_DIR
    g = nx.Graph()

    if not os.path.isdir(data_dir):
        logger.warning("Data dir not found: %s — returning empty graph", data_dir)
        _graph = g
        return g

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            _build_from_csv(g, path)
        except Exception:
            logger.exception("Error processing %s", path)

    _graph = g
    logger.info("Knowledge graph built: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
    return g


def _get_graph() -> nx.Graph:
    """Lazy init — build on first access."""
    global _graph
    if _graph is None:
        build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_entity(name: str) -> dict[str, Any] | None:
    """Return node data + immediate connections for an entity."""
    g = _get_graph()
    key = name.lower().strip()
    if key not in g:
        return None
    node = dict(g.nodes[key])
    neighbors = []
    for n in g.neighbors(key):
        edge = g.edges[key, n]
        neighbors.append({
            "name": g.nodes[n].get("label", n),
            "type": g.nodes[n].get("type", "unknown"),
            "relation": edge.get("relation", "connected"),
        })
    return {
        "name": node.get("label", key),
        "type": node.get("type", "unknown"),
        "connections": neighbors,
    }


def get_connections(name: str, depth: int = 1) -> dict[str, Any] | None:
    """BFS out to `depth` hops from the named node."""
    g = _get_graph()
    key = name.lower().strip()
    if key not in g:
        return None

    visited: set[str] = set()
    nodes: list[dict] = []
    edges: list[dict] = []
    frontier = {key}

    for d in range(depth + 1):
        next_frontier: set[str] = set()
        for n in frontier:
            if n in visited:
                continue
            visited.add(n)
            nd = g.nodes[n]
            nodes.append({"name": nd.get("label", n), "type": nd.get("type", "unknown"), "depth": d})
            for nb in g.neighbors(n):
                if nb not in visited:
                    next_frontier.add(nb)
                edge = g.edges[n, nb]
                edge_key = tuple(sorted([n, nb]))
                edges.append({
                    "source": nd.get("label", n),
                    "target": g.nodes[nb].get("label", nb),
                    "relation": edge.get("relation", "connected"),
                })
        frontier = next_frontier

    # Dedupe edges
    seen_edges: set[tuple] = set()
    unique_edges = []
    for e in edges:
        k = (e["source"], e["target"], e["relation"])
        if k not in seen_edges:
            seen_edges.add(k)
            unique_edges.append(e)

    return {"root": name, "depth": depth, "nodes": nodes, "edges": unique_edges}


def find_path(from_node: str, to_node: str) -> dict[str, Any]:
    """Shortest path between two entities."""
    g = _get_graph()
    src, dst = from_node.lower().strip(), to_node.lower().strip()
    if src not in g:
        return {"error": f"Entity '{from_node}' not found"}
    if dst not in g:
        return {"error": f"Entity '{to_node}' not found"}
    try:
        path_keys = nx.shortest_path(g, src, dst)
    except nx.NetworkXNoPath:
        return {"error": f"No path between '{from_node}' and '{to_node}'", "path": []}

    path = []
    for k in path_keys:
        nd = g.nodes[k]
        path.append({"name": nd.get("label", k), "type": nd.get("type", "unknown")})

    edges = []
    for i in range(len(path_keys) - 1):
        e = g.edges[path_keys[i], path_keys[i + 1]]
        edges.append({
            "source": path[i]["name"],
            "target": path[i + 1]["name"],
            "relation": e.get("relation", "connected"),
        })

    return {"path": path, "edges": edges, "length": len(path_keys) - 1}


def get_stats() -> dict[str, Any]:
    """Graph statistics: node/edge counts by type."""
    g = _get_graph()
    type_counts: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        t = data.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    rel_counts: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        r = data.get("relation", "connected")
        rel_counts[r] = rel_counts.get(r, 0) + 1

    return {
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "node_types": type_counts,
        "edge_types": rel_counts,
        "connected_components": nx.number_connected_components(g),
    }


def search_nodes(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search nodes by substring match on label."""
    g = _get_graph()
    q = query.lower().strip()
    results = []
    for key, data in g.nodes(data=True):
        label = data.get("label", key)
        if q in label.lower():
            results.append({
                "name": label,
                "type": data.get("type", "unknown"),
                "connections": g.degree(key),
            })
            if len(results) >= limit:
                break
    results.sort(key=lambda x: x["connections"], reverse=True)
    return results

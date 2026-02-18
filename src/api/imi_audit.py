"""
IMI Audit Logger — tracks every query, response, and action.
Enterprise-grade logging for compliance and analytics.
"""

import json
import os
import time
import logging
from functools import wraps
from config import AUDIT_LOG_PATH

logger = logging.getLogger("imi_audit")


def log_event(endpoint: str, query: str = "", response_summary: str = "",
              datasets: list = None, processing_ms: float = 0, user: str = "anonymous",
              extra: dict = None):
    """Log an audit event."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": endpoint,
        "user": user,
        "query": query[:500] if query else "",
        "response_summary": response_summary[:200] if response_summary else "",
        "datasets_accessed": datasets or [],
        "processing_time_ms": round(processing_ms, 1),
    }
    if extra:
        entry.update(extra)
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Audit log write failed: {e}")


def get_audit_log(limit: int = 100, offset: int = 0) -> list[dict]:
    """Read recent audit entries."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    entries = []
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Return most recent first
    entries.reverse()
    return entries[offset:offset + limit]


def get_audit_stats() -> dict:
    """Compute audit statistics."""
    entries = get_audit_log(limit=10000)
    if not entries:
        return {"total_queries": 0, "endpoints": {}, "datasets": {}, "avg_processing_ms": 0}

    endpoints = {}
    datasets = {}
    total_ms = 0

    for e in entries:
        ep = e.get("endpoint", "unknown")
        endpoints[ep] = endpoints.get(ep, 0) + 1
        for ds in e.get("datasets_accessed", []):
            datasets[ds] = datasets.get(ds, 0) + 1
        total_ms += e.get("processing_time_ms", 0)

    # Queries per day
    days = {}
    for e in entries:
        day = e.get("ts", "")[:10]
        if day:
            days[day] = days.get(day, 0) + 1

    return {
        "total_queries": len(entries),
        "endpoints": dict(sorted(endpoints.items(), key=lambda x: -x[1])),
        "popular_datasets": dict(sorted(datasets.items(), key=lambda x: -x[1])[:10]),
        "avg_processing_ms": round(total_ms / len(entries), 1) if entries else 0,
        "queries_per_day": dict(sorted(days.items())),
        "first_entry": entries[-1].get("ts", "") if entries else "",
        "last_entry": entries[0].get("ts", "") if entries else "",
    }

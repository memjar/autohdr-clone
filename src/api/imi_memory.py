"""
IMI Team Memory Engine — Persistent append-only JSONL memory store.

Stores query history, analysis results, insights, and feedback
in ~/.axe/klaus_data/team_memory.jsonl for cross-session learning.
"""

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

MEMORY_DIR = os.path.expanduser("~/.axe/klaus_data")
MEMORY_FILE = os.path.join(MEMORY_DIR, "team_memory.jsonl")


def _ensure_dir():
    os.makedirs(MEMORY_DIR, exist_ok=True)


def _read_all() -> list[dict]:
    """Read all memory entries from the JSONL file."""
    _ensure_dir()
    if not os.path.exists(MEMORY_FILE):
        return []
    entries = []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def log_memory(
    type: str,
    query: str,
    findings: str,
    datasets_used: list[str] | None = None,
    user: str = "anonymous",
    tags: list[str] | None = None,
) -> dict:
    """Append a memory entry. Returns the created entry."""
    _ensure_dir()
    entry = {
        "id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": type,
        "query": query,
        "findings": findings,
        "datasets_used": datasets_used or [],
        "user": user,
        "tags": tags or [],
    }
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def search_memory(query: str, limit: int = 10) -> list[dict]:
    """Search memories by keyword matching in query + findings + tags."""
    keywords = set(re.findall(r"\w+", query.lower()))
    if not keywords:
        return []

    entries = _read_all()
    scored = []
    for e in entries:
        text = f"{e.get('query', '')} {e.get('findings', '')} {' '.join(e.get('tags', []))}".lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, e))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


def get_insights(limit: int = 20) -> list[dict]:
    """Return the most recent memories."""
    entries = _read_all()
    return entries[-limit:][::-1]


def get_recurring_patterns() -> dict:
    """Analyze memories to find recurring topics, datasets, and query patterns."""
    entries = _read_all()
    if not entries:
        return {"most_queried_topics": [], "most_used_datasets": [], "common_patterns": [], "total_analyzed": 0}

    # Extract words from queries for topic analysis
    all_words: list[str] = []
    dataset_counter: Counter = Counter()
    type_counter: Counter = Counter()
    tag_counter: Counter = Counter()

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
                  "when", "where", "which", "who", "and", "or", "but", "in", "on", "at",
                  "to", "for", "of", "with", "by", "from", "it", "this", "that", "do", "does"}

    for e in entries:
        words = re.findall(r"\w+", e.get("query", "").lower())
        all_words.extend(w for w in words if w not in stop_words and len(w) > 2)
        for ds in e.get("datasets_used", []):
            dataset_counter[ds] += 1
        type_counter[e.get("type", "unknown")] += 1
        for tag in e.get("tags", []):
            tag_counter[tag] += 1

    word_counter = Counter(all_words)

    return {
        "most_queried_topics": [{"topic": w, "count": c} for w, c in word_counter.most_common(15)],
        "most_used_datasets": [{"dataset": d, "count": c} for d, c in dataset_counter.most_common(10)],
        "common_patterns": [{"tag": t, "count": c} for t, c in tag_counter.most_common(10)],
        "query_types": dict(type_counter),
        "total_analyzed": len(entries),
    }


def get_context_for_query(query: str) -> list[dict]:
    """Find relevant past analyses for a new query via keyword overlap."""
    return search_memory(query, limit=5)


def add_feedback(memory_id: str, feedback: str) -> bool:
    """Add user feedback to a memory entry. Rewrites file with updated entry."""
    entries = _read_all()
    found = False
    for e in entries:
        if e.get("id") == memory_id:
            e["feedback"] = feedback
            found = True
            break

    if not found:
        return False

    _ensure_dir()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return True


def get_stats() -> dict:
    """Return memory statistics."""
    entries = _read_all()
    if not entries:
        return {"total": 0, "by_type": {}, "by_dataset": {}, "date_range": None}

    type_counter: Counter = Counter()
    dataset_counter: Counter = Counter()
    timestamps = []

    for e in entries:
        type_counter[e.get("type", "unknown")] += 1
        for ds in e.get("datasets_used", []):
            dataset_counter[ds] += 1
        ts = e.get("timestamp")
        if ts:
            timestamps.append(ts)

    timestamps.sort()
    return {
        "total": len(entries),
        "by_type": dict(type_counter),
        "by_dataset": dict(dataset_counter),
        "date_range": {"earliest": timestamps[0], "latest": timestamps[-1]} if timestamps else None,
    }

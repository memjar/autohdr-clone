"""
IMI RAG Pipeline — JSON Vector Store + Ollama Embeddings
Embeds all IMI datasets and provides semantic search for Klaus.
No ChromaDB dependency — uses a simple JSON file with cosine similarity.
"""

import os
import csv
import json
import hashlib
import logging
import math
import httpx

from config import IMI_DATA_DIR, VECTOR_STORE_PATH as STORE_PATH, OLLAMA_BASE, EMBED_MODEL

logger = logging.getLogger("imi_rag")

# In-memory store
_store = None  # {"documents": [{"text": str, "embedding": list, "metadata": dict}]}


def _load_store() -> dict:
    global _store
    if _store is not None:
        return _store
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, "r") as f:
            _store = json.load(f)
        logger.info(f"Loaded {len(_store.get('documents', []))} documents from {STORE_PATH}")
    else:
        _store = {"documents": []}
    return _store


def _save_store():
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump(_store, f)
    logger.info(f"Saved {len(_store['documents'])} documents to {STORE_PATH}")


def _embed_text(text: str) -> list[float]:
    """Get embedding from Ollama nomic-embed-text."""
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30.0
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _chunk_csv(filepath: str) -> list[dict]:
    filename = os.path.basename(filepath)
    dataset_name = filename.replace(".csv", "").replace("_", " ").title()
    chunks = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Schema chunk
        chunks.append({
            "text": f"Dataset: {dataset_name} | File: {filename} | Columns: {', '.join(headers)}",
            "metadata": {"dataset": filename, "category": _categorize(filename), "type": "schema"}
        })

        for i, row in enumerate(reader):
            parts = [f"{h}: {row.get(h, '').strip()}" for h in headers if row.get(h, "").strip()]
            text = f"[{dataset_name}] {' | '.join(parts)}"
            chunks.append({
                "text": text,
                "metadata": {"dataset": filename, "category": _categorize(filename), "row": i}
            })

    return chunks


def _chunk_json(filepath: str) -> list[dict]:
    filename = os.path.basename(filepath)
    dataset_name = filename.replace(".json", "").replace("_", " ").title()

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            text = f"[{dataset_name}] {json.dumps(item)}"
            chunks.append({
                "text": text,
                "metadata": {"dataset": filename, "category": "advertising", "row": i}
            })
    elif isinstance(data, dict):
        def flatten(obj, prefix=""):
            items = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    items.extend(flatten(v, f"{prefix}{k}."))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    items.extend(flatten(v, f"{prefix}[{i}]."))
            else:
                items.append(f"{prefix.rstrip('.')}: {obj}")
            return items

        flat_items = flatten(data)
        for i in range(0, len(flat_items), 10):
            batch = flat_items[i:i + 10]
            text = f"[{dataset_name}] {' | '.join(batch)}"
            chunks.append({
                "text": text,
                "metadata": {"dataset": filename, "category": "advertising", "row": i}
            })

    return chunks


def _categorize(filename: str) -> str:
    name = filename.lower()
    if "brand_health" in name: return "brand_health"
    if "competitive" in name: return "competitive"
    if "consumer_sentiment" in name: return "sentiment"
    if "genz" in name: return "segmentation"
    if "promotion" in name: return "roi"
    if "purchase_driver" in name: return "drivers"
    if "say_do" in name: return "behavioral"
    if "sponsorship" in name: return "roi"
    if "ad_pretest" in name: return "advertising"
    if "csfimi" in name or "sports_viewership" in name: return "sports"
    return "general"


def ingest_datasets(data_dir: str = IMI_DATA_DIR, force: bool = False) -> dict:
    """Ingest all IMI datasets — embed and store as JSON."""
    store = _load_store()

    if store["documents"] and not force:
        return {"status": "already_ingested", "documents": len(store["documents"])}

    if force:
        store["documents"] = []

    all_chunks = []
    files_processed = []

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".csv"):
            chunks = _chunk_csv(filepath)
            # For consumer_sentiment (2500 rows), sample every 5th row for denser coverage
            if "consumer_sentiment" in filename and len(chunks) > 500:
                sampled = [chunks[0]]  # Keep schema
                sampled.extend(chunks[1::5])  # Every 5th data row (~500 chunks)
                chunks = sampled
            all_chunks.extend(chunks)
            files_processed.append(filename)
            logger.info(f"Chunked {filename}: {len(chunks)} documents")
        elif filename.endswith(".json"):
            chunks = _chunk_json(filepath)
            all_chunks.extend(chunks)
            files_processed.append(filename)
            logger.info(f"Chunked {filename}: {len(chunks)} documents")

    # Ingest IMI knowledge base (methodology, frameworks, expertise)
    kb_path = os.path.join(os.path.dirname(__file__), "imi_knowledge_base.md")
    if os.path.exists(kb_path):
        with open(kb_path, "r") as f:
            kb_text = f.read()
        # Split by ## headings into logical chunks
        sections = kb_text.split("\n## ")
        for section in sections:
            if len(section.strip()) < 20:
                continue
            chunk_text = section if sections.index(section) == 0 else f"## {section}"
            all_chunks.append({
                "text": chunk_text.strip()[:1000],
                "metadata": {"dataset": "imi_knowledge_base", "category": "methodology", "type": "reference"}
            })
        files_processed.append("imi_knowledge_base.md")
        logger.info(f"Chunked imi_knowledge_base.md: {len(sections)} documents")

    if not all_chunks:
        return {"status": "no_data", "documents": 0}

    # Embed all chunks
    total = len(all_chunks)
    for i, chunk in enumerate(all_chunks):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"Embedding {i + 1}/{total}...")
        embedding = _embed_text(chunk["text"])
        store["documents"].append({
            "text": chunk["text"],
            "embedding": embedding,
            "metadata": chunk["metadata"]
        })

    _save_store()
    return {"status": "ingested", "documents": len(store["documents"]), "files": files_processed}


def ingest_document(filename: str, text_chunks: list[str]) -> dict:
    """Ingest pre-chunked document text into the vector store."""
    store = _load_store()
    dataset_name = f"uploaded_{filename}"
    count = 0
    for i, chunk_text in enumerate(text_chunks):
        if not chunk_text.strip():
            continue
        embedding = _embed_text(chunk_text)
        store["documents"].append({
            "text": f"[{dataset_name}] {chunk_text}",
            "embedding": embedding,
            "metadata": {"dataset": dataset_name, "category": "uploaded", "row": i}
        })
        count += 1
    _save_store()
    return {"status": "ingested", "documents": count, "dataset": dataset_name}


def query(text: str, n: int = 10, category: str = None) -> list[dict]:
    """Hybrid search: semantic similarity + keyword boosting + entity/metric matching."""
    store = _load_store()

    if not store["documents"]:
        logger.warning("Vector store is empty — run ingest_datasets() first")
        return []

    query_embedding = _embed_text(text)
    query_lower = text.lower()
    query_words = set(query_lower.split())

    # Detect specific metrics and entities in query for targeted boosting
    metric_terms = {
        "nps": ["nps", "net promoter"],
        "awareness": ["awareness", "unaided", "aided"],
        "loyalty": ["loyalty", "loyalty score"],
        "trust": ["trust", "trust score"],
        "market_share": ["market share", "share"],
        "purchase": ["purchase", "purchase intent", "buying"],
    }
    query_metrics = set()
    for metric, triggers in metric_terms.items():
        if any(t in query_lower for t in triggers):
            query_metrics.add(metric)

    # Detect brand names for entity boosting
    brand_names = ["tim hortons", "starbucks", "mcdonald", "a&w", "burger king",
                   "wendy", "subway", "taco bell", "popeyes", "dairy queen",
                   "pizza pizza", "domino", "little caesars", "kfc", "chick-fil-a"]
    query_brands = [b for b in brand_names if b in query_lower]

    # Score all documents with hybrid approach
    scored = []
    for doc in store["documents"]:
        if category and doc["metadata"].get("category") != category:
            continue

        # Semantic score
        sem_score = _cosine_sim(query_embedding, doc["embedding"])

        # Keyword boost: if query words appear in the document text, boost score
        doc_lower = doc["text"].lower()
        keyword_hits = sum(1 for w in query_words if len(w) > 2 and w in doc_lower)
        keyword_boost = min(keyword_hits * 0.05, 0.2)  # Max 0.2 boost

        # Entity boost: if queried brand appears in document
        entity_boost = 0.0
        if query_brands:
            brand_hits = sum(1 for b in query_brands if b in doc_lower)
            entity_boost = min(brand_hits * 0.1, 0.2)

        # Metric boost: if queried metric appears in document
        metric_boost = 0.0
        if query_metrics:
            for m in query_metrics:
                for trigger in metric_terms[m]:
                    if trigger in doc_lower:
                        metric_boost = 0.1
                        break

        # Category boost: brand_health docs get priority for brand metric queries
        cat_boost = 0.0
        doc_cat = doc["metadata"].get("category", "")
        if query_metrics and doc_cat == "brand_health":
            cat_boost = 0.05
        if "competitive" in query_lower and doc_cat == "competitive":
            cat_boost = 0.05

        # Dataset name boost: if query terms appear in the dataset filename
        ds_name = doc["metadata"].get("dataset", "").lower().replace("_", " ").replace(".csv", "").replace(".json", "")
        ds_hits = sum(1 for w in query_words if len(w) > 2 and w in ds_name)
        ds_boost = min(ds_hits * 0.08, 0.15)

        final_score = sem_score + keyword_boost + entity_boost + metric_boost + cat_boost + ds_boost
        scored.append((final_score, doc))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Dataset diversity: max 3 results per dataset to avoid dominance
    max_per_dataset = 3
    dataset_counts = {}
    results = []
    for score, doc in scored:
        if len(results) >= n:
            break
        ds = doc["metadata"].get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        if dataset_counts[ds] > max_per_dataset:
            continue
        results.append({
            "text": doc["text"],
            "dataset": ds,
            "category": doc["metadata"].get("category", "general"),
            "score": round(score, 4)
        })

    return results


def format_context(chunks: list[dict]) -> str:
    """Format RAG results as context for the LLM system message."""
    if not chunks:
        return ""

    # Group by dataset for cleaner presentation
    by_dataset = {}
    for c in chunks:
        ds = c['dataset']
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(c['text'])

    lines = ["RELEVANT DATA FROM IMI DATASETS:\n"]
    for ds, texts in by_dataset.items():
        lines.append(f"=== {ds} ===")
        for t in texts:
            # Convert pipe-delimited to cleaner format
            if " | " in t:
                # Extract dataset prefix [Name] and fields
                prefix_end = t.find("]")
                if prefix_end > 0:
                    fields = t[prefix_end + 2:].split(" | ")
                    for field in fields:
                        lines.append(f"  {field}")
                    lines.append("")
                else:
                    lines.append(f"  {t}")
            else:
                lines.append(f"  {t}")
        lines.append("")

    lines.append(f"Data sources: {', '.join(sorted(by_dataset.keys()))}")
    lines.append("Use ONLY the data above to answer. Cite the dataset name when referencing specific numbers.")
    return "\n".join(lines)

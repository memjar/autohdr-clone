"""
IMI Training Data Generator — Converts learning log + live queries into QLoRA training pairs.
Reads from imi_learning_log.jsonl and generates instruction-tuning JSONL for fine-tuning.
"""

import json
import os
import logging
import time
import httpx

logger = logging.getLogger("imi_training")

LEARNING_LOG = os.path.expanduser("~/.axe/klaus_data/imi_learning_log.jsonl")
TRAINING_OUTPUT = os.path.expanduser("~/.axe/klaus_training/imi_finetune.jsonl")
OLLAMA_BASE = "http://localhost:11434"
MODEL = "klaus-imi"


def _call_ollama(system: str, user: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
    for attempt in range(2):
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": max_tokens}
                },
                timeout=300.0
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except httpx.TimeoutException:
            if attempt == 0:
                logger.warning("Ollama timeout, retrying...")
                continue
            raise


# Seed queries covering all 13 datasets — these generate high-quality training pairs
SEED_QUERIES = [
    # Brand Health
    ("What's Tim Hortons NPS?", "lookup"),
    ("Which brand has the highest loyalty score?", "lookup"),
    ("Compare trust scores across QSR brands", "comparison"),
    # Competitive Benchmark
    ("Which brands have the highest trial rate?", "lookup"),
    ("Compare awareness vs consideration for the top 5 brands", "comparison"),
    ("How does Starbucks rank on value perception vs quality perception?", "comparison"),
    # Market Share
    ("Which brand gained the most market share year over year?", "trend"),
    ("Show me the top 5 brands by market share", "lookup"),
    # Consumer Sentiment
    ("What's the overall consumer sentiment in Canada right now?", "lookup"),
    ("How does sentiment differ by region across Canada?", "comparison"),
    ("Compare sentiment between Gen Z and Boomers", "comparison"),
    # Gen Z Segmentation
    ("What are the key Gen Z lifestyle segments?", "report"),
    ("How do Gen Z purchase behaviors differ from older generations?", "comparison"),
    # Promotion ROI
    ("Which promotion type delivers the best ROI?", "lookup"),
    ("Compare BOGO vs percentage discount ROI across brands", "comparison"),
    # Purchase Drivers
    ("What are the top purchase drivers for Gen Z vs Millennials?", "comparison"),
    ("What drives repeat purchases in the QSR category?", "lookup"),
    # Say-Do Gap
    ("Which brands have the biggest say-do gap?", "lookup"),
    ("Where is the gap between stated preference and actual behavior largest?", "report"),
    # Sponsorship
    ("What are the top sponsorship properties by opportunity score?", "lookup"),
    ("Compare sponsorship ROI across hockey vs soccer properties", "comparison"),
    # Sports Viewership
    ("What sports do Canadians most want to watch?", "lookup"),
    ("How does sports viewership differ by age group?", "comparison"),
    ("Compare sports preferences between 18-34 and 55+ demographics", "comparison"),
    # CSFIMI Insights
    ("What is the nativity divide in Canadian sports fandom?", "report"),
    ("How does FIFA viewership differ between immigrants and Canadian-born?", "comparison"),
    ("What are the regional differences in sports fandom across Canada?", "comparison"),
    ("How does gender affect sports viewership preferences?", "comparison"),
    # Ad Testing
    ("Which ad campaign performed best in pretesting?", "lookup"),
    ("Compare Campaign A vs B vs C on key metrics", "comparison"),
    # Cross-Dataset
    ("Give me a full brand health report for the QSR category", "report"),
    ("How do purchase drivers connect to brand loyalty scores?", "report"),
    ("What's the relationship between promotion ROI and market share?", "report"),
    # Cross-Dataset Synthesis (IMI differentiator)
    ("Which Gen Z segments show the largest say-do gap?", "report"),
    ("Do sponsorship-heavy brands have higher NPS scores?", "report"),
    ("How does consumer sentiment correlate with brand loyalty by region?", "report"),
    ("What purchase drivers predict the smallest say-do gap?", "report"),
    ("Compare promotion ROI effectiveness across different consumer sentiment levels", "report"),
    ("Which brands have strong awareness but weak consideration — and why?", "report"),
    ("How do Gen Z purchase drivers differ from what their say-do gap suggests?", "report"),
    ("What's the connection between sports fandom and brand affinity for Canadian-born vs immigrants?", "report"),
    ("Which growth levers are most effective for brands with low trust scores?", "report"),
    ("Map the brand health funnel drop-offs for the top 5 QSR brands", "report"),
    ("How does regional sentiment in Canada predict sponsorship ROI?", "report"),
    ("What experiential marketing opportunities exist based on Gen Z lifestyle segments?", "report"),
    ("Compare the say-do gap between high-NPS and low-NPS brands", "comparison"),
    ("Which demographic has the most consistent purchase behavior across all categories?", "report"),
    ("How do ad pretest scores correlate with actual market share performance?", "report"),
    ("What's the optimal promotion strategy for brands targeting Gen Z based on their lifestyle segments?", "report"),
]


def generate_training_pair(query: str, query_type: str, rag_chunks: list[dict]) -> dict:
    """Generate a single training pair from a query using the pipeline."""
    from imi_agents import run_pipeline
    response = run_pipeline(query, rag_chunks)

    # Quality gate — only include high-quality responses
    has_bold = "**" in response
    has_structure = any(h in response for h in ["**Executive Summary**", "**Key Data**", "**SO WHAT?**"])

    if len(response) < 100 or not has_bold:
        return None

    return {
        "instruction": query,
        "input": "",
        "output": response,
        "metadata": {
            "type": query_type,
            "quality_score": sum([has_bold, has_structure, len(response) > 500, "SO WHAT" in response.upper()]),
            "response_length": len(response),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    }


def generate_from_seeds(max_pairs: int = 35) -> dict:
    """Generate training pairs from seed queries. Returns stats."""
    from imi_rag import query as rag_query

    os.makedirs(os.path.dirname(TRAINING_OUTPUT), exist_ok=True)

    # Load existing pairs to avoid duplicates
    existing_queries = set()
    if os.path.exists(TRAINING_OUTPUT):
        with open(TRAINING_OUTPUT, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing_queries.add(entry.get("instruction", ""))
                except json.JSONDecodeError:
                    pass

    generated = 0
    skipped = 0
    failed = 0

    for query, qtype in SEED_QUERIES[:max_pairs]:
        if query in existing_queries:
            skipped += 1
            continue

        try:
            chunks = rag_query(query, n=15)
            if not chunks:
                failed += 1
                continue

            pair = generate_training_pair(query, qtype, chunks)
            if pair:
                with open(TRAINING_OUTPUT, "a") as f:
                    f.write(json.dumps(pair) + "\n")
                generated += 1
                logger.info(f"Generated pair {generated}: {query[:50]}... (quality={pair['metadata']['quality_score']})")
            else:
                failed += 1
        except Exception as e:
            logger.warning(f"Failed to generate pair for '{query[:50]}...': {e}")
            failed += 1

    return {
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "total_in_file": len(existing_queries) + generated,
    }


def convert_learning_log_to_training() -> dict:
    """Convert learning log entries into training pairs by re-running high-quality queries."""
    if not os.path.exists(LEARNING_LOG):
        return {"status": "no_learning_log"}

    from imi_rag import query as rag_query

    entries = []
    with open(LEARNING_LOG, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Filter to high-quality interactions only
    good_entries = [e for e in entries if e.get("has_bold_numbers") and e.get("response_length", 0) > 500]

    generated = 0
    for entry in good_entries:
        query = entry["query"]
        chunks = rag_query(query, n=15)
        if chunks:
            pair = generate_training_pair(query, entry.get("classification", {}).get("type", "lookup"), chunks)
            if pair:
                os.makedirs(os.path.dirname(TRAINING_OUTPUT), exist_ok=True)
                with open(TRAINING_OUTPUT, "a") as f:
                    f.write(json.dumps(pair) + "\n")
                generated += 1

    return {"converted": generated, "total_log_entries": len(entries), "high_quality": len(good_entries)}

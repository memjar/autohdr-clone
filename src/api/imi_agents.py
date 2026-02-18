"""
IMI Multi-Agent Pipeline — Optimized for Speed
Query Parser → Data Retriever → Analyzer → Learning Loop

OPTIMIZATION NOTES (Feb 2026):
- Classification uses 7B model (fast) instead of 32B
- Synthesize stage REMOVED — was adding 30-60s for marginal polish
- num_ctx reduced to 4096 (from 16384) for all pipeline stages
- keep_alive=24h prevents model reload between calls
- Simple queries (lookup/meta) skip classification entirely
- num_predict capped at 1024 for analysis, 256 for classification
"""

import json
import os
import time
import httpx
import logging

from config import (
    OLLAMA_BASE, IMI_MODEL as MODEL, LEARNING_LOG_PATH as LEARNING_LOG,
    IMI_FAST_MODEL,
)

logger = logging.getLogger("imi_agents")

# Reuse a single httpx client with reasonable timeout
_pipeline_client = httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0))


def _call_ollama(
    system: str, user: str, temperature: float = 0.2,
    max_tokens: int = 1024, model: str = None, num_ctx: int = 4096,
) -> str:
    """Synchronous Ollama call for pipeline stages."""
    resp = _pipeline_client.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model or MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": "24h",
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
            }
        },
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def classify_query(message: str) -> dict:
    """Stage 1: Classify query intent — uses FAST 7B model."""
    system = """Classify this query for an IMI International marketing research system.
Return JSON only:
{"type": "lookup|comparison|trend|report|sponsorship|sports|saydo|meta", "datasets": ["filename.csv"], "entities": [], "cross_dataset": true/false}

Available datasets: brand_health_tracker_Q4_2025.csv, competitive_benchmark_overall.csv, competitive_benchmark_by_generation.csv, competitive_benchmark_market_share.csv, consumer_sentiment_survey_canada_jan2026.csv, genz_lifestyle_segmentation.csv, promotion_roi_analysis_2025.csv, purchase_drivers_by_generation_Q4_2025.csv, say_do_gap_food_beverage.csv, sponsorship_property_scores.csv, sports_viewership_crosstab_2025.csv, csfimi_sports_fandom_insights_2025.csv, ad_pretest_results_campaign_A_B_C.json"""

    t0 = time.time()
    result = _call_ollama(
        system, message,
        temperature=0.1, max_tokens=256,
        model=IMI_FAST_MODEL, num_ctx=2048,
    )
    logger.info(f"Classification took {time.time() - t0:.1f}s")

    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(result[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    return {"type": "lookup", "datasets": [], "entities": [], "cross_dataset": False}


def retrieve_data(message: str, classification: dict, rag_chunks: list[dict]) -> str:
    """Stage 2: Retrieve and organize relevant data — NO LLM call, pure data formatting."""
    if not rag_chunks:
        return "No relevant data found in IMI datasets."

    by_dataset = {}
    for chunk in rag_chunks:
        ds = chunk.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(chunk["text"])

    sections = []
    for ds, texts in by_dataset.items():
        sections.append(f"=== {ds} ===")
        for t in texts:
            if " | " in t:
                prefix_end = t.find("]")
                if prefix_end > 0:
                    fields = t[prefix_end + 2:].split(" | ")
                    for field in fields:
                        sections.append(f"  {field}")
                    sections.append("")
                else:
                    sections.append(f"  {t}")
            else:
                sections.append(f"  {t}")

    return "\n".join(sections)


def analyze(message: str, data_context: str, classification: dict) -> str:
    """Stage 3: Deep analysis — the ONLY 32B LLM call in the pipeline."""
    query_type = classification.get("type", "lookup")
    cross_dataset = classification.get("cross_dataset", False)

    system = f"""You are Klaus, IMI International's AI insight engine. 54-year-old global firm, 45+ countries, 50,000+ case studies.

QUERY TYPE: {query_type} | CROSS-DATASET: {cross_dataset}

FRAMEWORKS: Say/Do Gap (stated vs actual behavior), 12 Growth Levers, Brand Health Funnel.

VOICE: Lead with the most surprising finding. Bold key numbers (**47%**). Calculate ratios/deltas. Cite dataset names. NEVER fabricate data.

FORMAT:
**Executive Summary** — 2-3 sentence headline
**Key Data** — Markdown table, bold numbers, source cited
**Analysis** — What data means, cross-dataset connections, non-obvious insight
**SO WHAT?** — 3 actionable recommendations with expected impact

DATA:
{data_context}"""

    t0 = time.time()
    result = _call_ollama(system, message, temperature=0.3, max_tokens=1024)
    logger.info(f"Analysis took {time.time() - t0:.1f}s")
    return result


def _log_interaction(message: str, classification: dict, result: str, rag_chunks: list[dict]):
    """Learning loop: log every interaction for future training data generation."""
    os.makedirs(os.path.dirname(LEARNING_LOG), exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": message,
        "classification": classification,
        "datasets_used": list({c.get("dataset", "") for c in rag_chunks}),
        "response_length": len(result),
        "has_table": "|" in result and "---" in result,
        "has_so_what": "SO WHAT" in result.upper(),
        "has_bold_numbers": "**" in result,
    }
    try:
        with open(LEARNING_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Learning log write failed: {e}")


def run_pipeline(message: str, rag_chunks: list[dict]) -> str:
    """
    Run the optimized multi-agent pipeline.

    Pipeline (2 LLM calls instead of 3-4):
      1. Classify (7B, ~2-4s) — determines query type
      2. Retrieve (no LLM) — organizes RAG chunks
      3. Analyze (32B, ~15-30s) — single deep analysis call
      4. Learn (no LLM) — logs interaction

    Eliminated: synthesize stage (was adding 30-60s for marginal improvement)
    """
    t0 = time.time()

    # Stage 1: Classify with fast model
    classification = classify_query(message)
    logger.info(f"Classification: {classification}")

    # Stage 2: Retrieve & organize (no LLM call)
    data_context = retrieve_data(message, classification, rag_chunks)

    # Stage 3: Analyze with 32B
    result = analyze(message, data_context, classification)

    # Stage 4: Learn (no LLM call)
    _log_interaction(message, classification, result, rag_chunks)

    logger.info(f"Full pipeline took {time.time() - t0:.1f}s")
    return result

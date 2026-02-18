"""
IMI Eval Harness — Tests pipeline quality with structured scoring.
Run: python3 imi_eval.py
All local Ollama, zero API cost.
"""

import json
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("imi_eval")

EVAL_OUTPUT = os.path.expanduser("~/.axe/klaus_training/imi_eval_results.json")

# Test cases: (query, expected_datasets, expected_keywords)
TEST_CASES = [
    ("What's Tim Hortons NPS?", ["brand_health"], ["NPS", "Tim Hortons"]),
    ("Compare Gen Z vs Boomers purchase drivers", ["purchase_drivers"], ["Gen Z", "Boomer"]),
    ("Which brand has the highest market share?", ["competitive_benchmark_market_share"], ["market share"]),
    ("What is the say-do gap in food and beverage?", ["say_do_gap"], ["Say", "Do", "gap"]),
    ("How does sports viewership differ by age?", ["sports_viewership", "csfimi"], ["age", "viewership"]),
    ("Which promotion type has the best ROI?", ["promotion_roi"], ["ROI", "promotion"]),
    ("What are the top sponsorship properties?", ["sponsorship"], ["sponsorship", "score"]),
    ("Compare Campaign A vs B vs C", ["ad_pretest"], ["Campaign"]),
    ("What is the nativity divide in Canadian sports fandom?", ["csfimi"], ["nativity", "Canadian-born"]),
    ("Give me a brand health report for QSR", ["brand_health", "competitive"], ["NPS", "awareness"]),
    ("How does consumer sentiment differ by region?", ["consumer_sentiment"], ["region", "sentiment"]),
    ("What are the key Gen Z lifestyle segments?", ["genz_lifestyle"], ["Gen Z", "segment"]),
    ("Which brands have the biggest say-do gap?", ["say_do_gap"], ["gap", "brand"]),
    ("Compare awareness vs consideration for top brands", ["competitive_benchmark"], ["awareness", "consideration"]),
    ("What drives repeat purchases in QSR?", ["purchase_drivers"], ["purchase", "driver"]),
]


def score_response(response: str, expected_datasets: list, expected_keywords: list) -> dict:
    """Score a pipeline response on 4 dimensions (0-3 each, max 12)."""
    scores = {}

    # Structure (0-3): Has Executive Summary, Key Data, Analysis, SO WHAT
    structure_markers = ["**Executive Summary**", "**Key Data**", "**Analysis**", "**SO WHAT?**"]
    scores["structure"] = min(3, sum(1 for m in structure_markers if m in response))

    # Accuracy (0-3): Has bold numbers, tables, dataset citations
    has_bold = "**" in response and any(c.isdigit() for c in response)
    has_table = "|" in response and "---" in response
    has_citations = any(ds in response.lower() for ds in expected_datasets)
    scores["accuracy"] = sum([has_bold, has_table, has_citations])

    # IMI Framework Usage (0-3): Say/Do, Growth Levers, Brand Funnel
    has_saydo = "say" in response.lower() and "do" in response.lower()
    has_levers = any(l in response for l in ["Growth Lever", "Lever", "Promotion", "Brand", "Digital", "Emotion"])
    has_funnel = any(f in response.lower() for f in ["awareness", "consideration", "purchase", "recommendation"])
    scores["imi_frameworks"] = sum([has_saydo, has_levers, has_funnel])

    # Actionability (0-3): Has recommendations, specific actions, expected keywords
    has_recs = response.count("If you do") + response.count("recommend") + response.count("should")
    has_keywords = sum(1 for k in expected_keywords if k.lower() in response.lower())
    keyword_ratio = has_keywords / max(len(expected_keywords), 1)
    scores["actionability"] = min(3, (1 if has_recs > 0 else 0) + (1 if keyword_ratio > 0.5 else 0) + (1 if len(response) > 500 else 0))

    scores["total"] = sum(scores.values())
    scores["max"] = 12
    return scores


def run_eval():
    from imi_agents import run_pipeline
    from imi_rag import query as rag_query

    results = []
    total_score = 0

    for i, (query, expected_ds, expected_kw) in enumerate(TEST_CASES):
        logger.info(f"[{i+1}/{len(TEST_CASES)}] {query}")
        try:
            chunks = rag_query(query, n=15)
            response = run_pipeline(query, chunks)
            scores = score_response(response, expected_ds, expected_kw)
            total_score += scores["total"]

            results.append({
                "query": query,
                "scores": scores,
                "response_length": len(response),
                "pass": scores["total"] >= 7,  # 7/12 = ~58% threshold
            })
            logger.info(f"  Score: {scores['total']}/12 — {'PASS' if scores['total'] >= 7 else 'FAIL'}")
        except Exception as e:
            results.append({"query": query, "error": str(e), "scores": {"total": 0, "max": 12}, "pass": False})
            logger.warning(f"  ERROR: {e}")

        time.sleep(1)

    # Summary
    passed = sum(1 for r in results if r.get("pass"))
    avg_score = total_score / len(TEST_CASES)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": len(TEST_CASES),
        "passed": passed,
        "failed": len(TEST_CASES) - passed,
        "pass_rate": f"{passed/len(TEST_CASES)*100:.0f}%",
        "avg_score": f"{avg_score:.1f}/12",
        "results": results,
    }

    os.makedirs(os.path.dirname(EVAL_OUTPUT), exist_ok=True)
    with open(EVAL_OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nEVAL COMPLETE: {passed}/{len(TEST_CASES)} passed ({passed/len(TEST_CASES)*100:.0f}%), avg {avg_score:.1f}/12")
    logger.info(f"Results saved to {EVAL_OUTPUT}")
    return summary


if __name__ == "__main__":
    run_eval()

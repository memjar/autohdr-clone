"""
Multi-Study Meta-Analysis for Klaus IMI.
Cross-dataset synthesis with common findings, divergences, and confidence levels.
"""

import csv
import json
import logging
import os
import time

from fastapi import APIRouter
from pydantic import BaseModel

from config import IMI_DATA_DIR, OLLAMA_BASE, IMI_MODEL, IMI_NUM_CTX
from imi_rag import query as rag_query, format_context
from imi_agents import _call_ollama, _log_interaction, classify_query

logger = logging.getLogger("imi_meta_analysis")

router = APIRouter(prefix="/klaus/imi", tags=["imi-meta-analysis"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MetaAnalysisRequest(BaseModel):
    question: str
    datasets: list[str] = []


class MetaAnalysisResponse(BaseModel):
    datasets_analyzed: list[str]
    common_findings: list[str]
    divergences: list[str]
    synthesis: str
    confidence: str
    error: str | None = None


class CompareRequest(BaseModel):
    datasets: list[str]
    metric: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_dataset_files() -> list[dict]:
    """Scan IMI_DATA_DIR for CSV/JSON files and return metadata."""
    results = []
    if not os.path.isdir(IMI_DATA_DIR):
        logger.warning("IMI_DATA_DIR not found: %s", IMI_DATA_DIR)
        return results

    for fname in sorted(os.listdir(IMI_DATA_DIR)):
        path = os.path.join(IMI_DATA_DIR, fname)
        lower = fname.lower()

        if lower.endswith(".csv"):
            try:
                with open(path, newline="", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    headers = [h.strip() for h in next(reader)]
                    row_count = sum(1 for _ in reader)
                results.append({
                    "filename": fname,
                    "format": "csv",
                    "columns": headers,
                    "row_count": row_count,
                })
            except Exception as e:
                logger.error("Failed to read %s: %s", fname, e)

        elif lower.endswith(".json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    row_count = len(data)
                    columns = list(data[0].keys()) if data else []
                elif isinstance(data, dict):
                    row_count = 1
                    columns = list(data.keys())
                else:
                    row_count = 0
                    columns = []
                results.append({
                    "filename": fname,
                    "format": "json",
                    "columns": columns,
                    "row_count": row_count,
                })
            except Exception as e:
                logger.error("Failed to read %s: %s", fname, e)

    return results


def _load_csv_column(path: str, metric: str) -> tuple[list[str], list[dict]]:
    """Load a CSV and find a column matching *metric* (case-insensitive).
    Returns (matched_columns, rows_as_dicts)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        matched = [h for h in headers if metric.lower() in h.lower()]
        if not matched:
            return [], []
        rows = list(reader)
    return matched, rows


def _auto_select_datasets(question: str) -> list[str]:
    """Use RAG to find which datasets are most relevant to the question."""
    chunks = rag_query(question, n=20)
    # Collect unique dataset filenames from top results
    seen = []
    for chunk in chunks:
        ds = chunk.get("dataset", "")
        if ds and ds not in seen:
            seen.append(ds)
    return seen


def _retrieve_per_dataset(question: str, datasets: list[str]) -> dict[str, list[dict]]:
    """Retrieve RAG chunks per dataset. Returns {dataset_name: [chunks]}."""
    per_dataset: dict[str, list[dict]] = {}

    for ds in datasets:
        # Try category-filtered query first
        chunks = rag_query(question, n=8, category=ds)

        # If nothing, do unfiltered query and post-filter by dataset name
        if not chunks:
            all_chunks = rag_query(question, n=20)
            ds_lower = ds.lower().replace(".csv", "").replace(".json", "")
            chunks = [
                c for c in all_chunks
                if ds_lower in c.get("dataset", "").lower()
            ][:8]

        if chunks:
            per_dataset[ds] = chunks

    return per_dataset


def _synthesize_meta_analysis(
    question: str,
    per_dataset: dict[str, list[dict]],
) -> dict:
    """Call Ollama to synthesize findings across multiple datasets.
    Returns parsed JSON with common_findings, divergences, synthesis, confidence."""

    # Build per-dataset context summaries
    dataset_sections = []
    for ds, chunks in per_dataset.items():
        texts = [c["text"] for c in chunks]
        section = f"=== {ds} ===\n" + "\n".join(f"  {t}" for t in texts)
        dataset_sections.append(section)

    combined_context = "\n\n".join(dataset_sections)
    datasets_list = ", ".join(per_dataset.keys())

    system = f"""You are Klaus, IMI International's AI insight engine performing a multi-study meta-analysis.

You are analyzing data from these datasets: {datasets_list}

Your task: synthesize findings across ALL datasets to answer the research question.
Identify patterns of agreement and disagreement across studies.

DATA FROM EACH DATASET:
{combined_context}

You MUST return valid JSON only, no other text. Use this exact structure:
{{
  "common_findings": ["finding 1 supported by multiple datasets", "finding 2 ..."],
  "divergences": ["area where datasets disagree or show different patterns"],
  "synthesis": "2-4 paragraph executive synthesis combining all findings, citing specific datasets and numbers. Bold key figures with **. End with actionable implications.",
  "confidence": "high|medium|low — based on how many datasets converge on the same findings"
}}

Rules:
- common_findings: only include findings supported by 2+ datasets. Cite the dataset names.
- divergences: note any contradictions, gaps, or areas where datasets tell different stories.
- confidence: "high" if 3+ datasets agree, "medium" if 2 agree, "low" if findings are mostly from single sources.
- NEVER fabricate data. Only use what appears in the data above.
- If a dataset has no relevant data for the question, note that as a limitation."""

    t0 = time.time()
    result = _call_ollama(
        system, question,
        temperature=0.3, max_tokens=1500,
        num_ctx=IMI_NUM_CTX,
    )
    logger.info(f"Meta-analysis synthesis took {time.time() - t0:.1f}s")

    # Parse JSON from response
    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(result[start:end])
            return {
                "common_findings": parsed.get("common_findings", []),
                "divergences": parsed.get("divergences", []),
                "synthesis": parsed.get("synthesis", ""),
                "confidence": parsed.get("confidence", "low"),
            }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse meta-analysis JSON: %s", e)

    # Fallback: return raw text as synthesis
    return {
        "common_findings": [],
        "divergences": [],
        "synthesis": result,
        "confidence": "low",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/meta-analysis", response_model=MetaAnalysisResponse)
def meta_analysis(req: MetaAnalysisRequest):
    """Multi-study meta-analysis: retrieve per-dataset, synthesize across all."""
    t0 = time.time()

    # Auto-select datasets if none provided
    datasets = req.datasets
    if not datasets:
        datasets = _auto_select_datasets(req.question)
        logger.info(f"Auto-selected datasets: {datasets}")

    if not datasets:
        return MetaAnalysisResponse(
            datasets_analyzed=[],
            common_findings=[],
            divergences=[],
            synthesis="No relevant datasets found for this question.",
            confidence="low",
            error="Could not identify relevant datasets. Try specifying dataset names explicitly.",
        )

    # Retrieve RAG chunks per dataset
    per_dataset = _retrieve_per_dataset(req.question, datasets)

    if not per_dataset:
        return MetaAnalysisResponse(
            datasets_analyzed=[],
            common_findings=[],
            divergences=[],
            synthesis="No relevant data found across the specified datasets.",
            confidence="low",
            error="No matching data found. Ensure datasets are ingested.",
        )

    # Synthesize across datasets via Ollama
    try:
        result = _synthesize_meta_analysis(req.question, per_dataset)
    except Exception as e:
        logger.error("Meta-analysis synthesis error: %s", e)
        return MetaAnalysisResponse(
            datasets_analyzed=list(per_dataset.keys()),
            common_findings=[],
            divergences=[],
            synthesis="",
            confidence="low",
            error=f"Synthesis error: {e}",
        )

    # Log interaction for learning
    all_chunks = [c for chunks in per_dataset.values() for c in chunks]
    _log_interaction(
        req.question,
        {"type": "meta_analysis", "datasets": list(per_dataset.keys())},
        result.get("synthesis", ""),
        all_chunks,
    )

    logger.info(f"Full meta-analysis took {time.time() - t0:.1f}s across {len(per_dataset)} datasets")

    return MetaAnalysisResponse(
        datasets_analyzed=list(per_dataset.keys()),
        common_findings=result["common_findings"],
        divergences=result["divergences"],
        synthesis=result["synthesis"],
        confidence=result["confidence"],
    )


@router.get("/meta-analysis/datasets")
def list_datasets():
    """List available datasets with row counts and column names."""
    datasets = _list_dataset_files()
    return {"datasets": datasets, "count": len(datasets)}


@router.post("/meta-analysis/compare")
def compare_metric(req: CompareRequest):
    """Compare a specific metric across multiple datasets."""

    if not os.path.isdir(IMI_DATA_DIR):
        return {"error": "IMI_DATA_DIR not found", "comparison": []}

    comparison: list[dict] = []

    for ds in req.datasets:
        path = os.path.join(IMI_DATA_DIR, ds)
        if not os.path.exists(path) and not ds.lower().endswith((".csv", ".json")):
            path = os.path.join(IMI_DATA_DIR, ds + ".csv")
        if not os.path.exists(path):
            comparison.append({
                "dataset": ds,
                "error": "File not found",
                "matched_columns": [],
                "rows": [],
            })
            continue

        if path.lower().endswith(".csv"):
            try:
                matched_cols, rows = _load_csv_column(path, req.metric)
                if not matched_cols:
                    comparison.append({
                        "dataset": ds,
                        "error": f"No column matching '{req.metric}'",
                        "matched_columns": [],
                        "rows": [],
                    })
                    continue

                id_candidates = ["brand", "name", "category", "segment", "generation", "region", "property"]
                all_headers = list(rows[0].keys()) if rows else []
                id_cols = [h for h in all_headers if any(ic in h.lower() for ic in id_candidates)]

                keep_cols = list(set(id_cols + matched_cols))
                filtered_rows = [{k: r[k] for k in keep_cols if k in r} for r in rows]

                comparison.append({
                    "dataset": ds,
                    "matched_columns": matched_cols,
                    "id_columns": id_cols,
                    "rows": filtered_rows[:100],
                    "total_rows": len(rows),
                })
            except Exception as e:
                comparison.append({"dataset": ds, "error": str(e), "matched_columns": [], "rows": []})
        else:
            comparison.append({"dataset": ds, "error": "Only CSV comparison supported", "matched_columns": [], "rows": []})

    return {
        "metric": req.metric,
        "datasets_compared": len(req.datasets),
        "comparison": comparison,
    }

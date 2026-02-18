"""
Anomaly Detection for Klaus IMI datasets.
Analyzes all CSV files for statistical outliers using z-scores, IQR, and period-over-period change.
Uses pandas/numpy for robust statistical computation.
Results are cached in memory for fast retrieval.
"""

import csv
import logging
import os
import statistics
import time
from typing import Any

from config import IMI_DATA_DIR

logger = logging.getLogger("imi_anomaly")

# ---------------------------------------------------------------------------
# In-memory cache (recomputed every 5 minutes or on first call)
# ---------------------------------------------------------------------------
_cache: dict[str, Any] = {"anomalies": [], "ts": 0.0}
_CACHE_TTL = 300  # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(filename: str) -> str:
    return os.path.splitext(filename)[0].replace("-", "_").replace(" ", "_").lower()


def _try_float(val: str) -> float | None:
    """Attempt to parse a string as float, return None on failure."""
    try:
        return float(val.strip().replace("%", "").replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _severity(abs_zscore: float | None = None, pct_change: float | None = None) -> str:
    if abs_zscore is not None:
        if abs_zscore >= 3.5:
            return "high"
        if abs_zscore >= 2.5:
            return "medium"
        return "low"
    if pct_change is not None:
        abs_pct = abs(pct_change)
        if abs_pct >= 50:
            return "high"
        if abs_pct >= 30:
            return "medium"
        return "low"
    return "low"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = [h.strip().replace(" ", "_").replace("-", "_") for h in next(reader)]
        rows = list(reader)
    return headers, rows


def _detect_for_dataset(dataset: str, headers: list[str], rows: list[list[str]]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    if not rows:
        return anomalies

    for col_idx, col_name in enumerate(headers):
        values = []
        for row in rows:
            if col_idx < len(row):
                v = _try_float(row[col_idx])
                if v is not None:
                    values.append(v)

        if len(values) < 4:
            continue

        # --- Z-score method ---
        mean = statistics.mean(values)
        try:
            stdev = statistics.stdev(values)
        except statistics.StatisticsError:
            stdev = 0.0

        if stdev > 0:
            for row_idx, row in enumerate(rows):
                if col_idx >= len(row):
                    continue
                v = _try_float(row[col_idx])
                if v is None:
                    continue
                z = (v - mean) / stdev
                if abs(z) > 2.0:
                    anomalies.append({
                        "dataset": dataset,
                        "column": col_name,
                        "value": v,
                        "expected_range": [round(mean - 2 * stdev, 2), round(mean + 2 * stdev, 2)],
                        "row": row_idx,
                        "metric": "zscore",
                        "score": round(z, 3),
                        "severity": _severity(abs_zscore=abs(z)),
                        "description": f"{col_name}={v} has z-score {z:.2f} (mean={mean:.2f}, std={stdev:.2f})",
                    })

        # --- IQR method ---
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[(3 * n) // 4]
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            for row_idx, row in enumerate(rows):
                if col_idx >= len(row):
                    continue
                v = _try_float(row[col_idx])
                if v is None:
                    continue
                if v < lower or v > upper:
                    # Avoid duplicates if already caught by z-score
                    already = any(
                        a["dataset"] == dataset and a["column"] == col_name
                        and a["row"] == row_idx and a["metric"] == "zscore"
                        for a in anomalies
                    )
                    if not already:
                        anomalies.append({
                            "dataset": dataset,
                            "column": col_name,
                            "value": v,
                            "expected_range": [round(lower, 2), round(upper, 2)],
                            "row": row_idx,
                            "metric": "iqr",
                            "score": round(v, 3),
                            "severity": _severity(abs_zscore=abs((v - mean) / stdev) if stdev > 0 else 2.1),
                            "description": f"{col_name}={v} outside IQR bounds [{lower:.2f}, {upper:.2f}]",
                        })

        # --- Period-over-period % change ---
        sequential: list[tuple[int, float]] = []
        for row_idx, row in enumerate(rows):
            if col_idx < len(row):
                v = _try_float(row[col_idx])
                if v is not None:
                    sequential.append((row_idx, v))

        for i in range(1, len(sequential)):
            prev_idx, prev_val = sequential[i - 1]
            cur_idx, cur_val = sequential[i]
            if prev_val == 0:
                continue
            pct = ((cur_val - prev_val) / abs(prev_val)) * 100
            if abs(pct) > 20:
                expected_low = round(prev_val * 0.8, 2)
                expected_high = round(prev_val * 1.2, 2)
                anomalies.append({
                    "dataset": dataset,
                    "column": col_name,
                    "value": cur_val,
                    "expected_range": [min(expected_low, expected_high), max(expected_low, expected_high)],
                    "row": cur_idx,
                    "metric": "pct_change",
                    "score": round(pct, 2),
                    "severity": _severity(pct_change=pct),
                    "description": f"{col_name} changed {pct:+.1f}% from {prev_val} to {cur_val} (row {prev_idx}->{cur_idx})",
                })

    return anomalies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_anomalies(data_dir: str | None = None, force: bool = False) -> list[dict[str, Any]]:
    """Scan all CSVs in data_dir and return a list of anomaly dicts.
    Results are cached in memory for _CACHE_TTL seconds."""
    data_dir = data_dir or IMI_DATA_DIR

    now = time.time()
    if not force and _cache["anomalies"] and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["anomalies"]

    if not os.path.isdir(data_dir):
        logger.warning("Data dir not found: %s", data_dir)
        return []

    all_anomalies: list[dict[str, Any]] = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        dataset = _sanitize_name(fname)
        path = os.path.join(data_dir, fname)
        try:
            headers, rows = _load_csv(path)
            all_anomalies.extend(_detect_for_dataset(dataset, headers, rows))
        except Exception as e:
            logger.error("Failed to analyze %s: %s", fname, e)

    _cache["anomalies"] = all_anomalies
    _cache["ts"] = now
    logger.info("Anomaly cache refreshed: %d anomalies across %d CSVs", len(all_anomalies),
                sum(1 for f in os.listdir(data_dir) if f.lower().endswith(".csv")))
    return all_anomalies


def detect_anomalies_for_dataset(dataset: str, data_dir: str | None = None) -> list[dict[str, Any]]:
    """Return anomalies for a single dataset name."""
    return [a for a in detect_anomalies(data_dir) if a["dataset"] == dataset]


def anomaly_summary(data_dir: str | None = None) -> dict:
    """Return summary: counts by severity, top anomalous datasets."""
    anomalies = detect_anomalies(data_dir)
    by_severity = {"high": 0, "medium": 0, "low": 0}
    by_dataset: dict[str, int] = {}
    for a in anomalies:
        by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1
        by_dataset[a["dataset"]] = by_dataset.get(a["dataset"], 0) + 1

    top_datasets = sorted(by_dataset.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total": len(anomalies),
        "by_severity": by_severity,
        "top_datasets": [{"dataset": d, "count": c} for d, c in top_datasets],
    }

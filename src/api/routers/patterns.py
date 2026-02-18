"""
Pattern Recognition module for Klaus IMI.
Auto-discovers correlations, distributions, and outliers in datasets.
"""

from __future__ import annotations

import os
import glob as _glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

try:
    from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurtosis
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

router = APIRouter(prefix="/klaus/imi", tags=["imi-patterns"])

CSV_DIR = Path.home() / ".axe" / "klaus_data" / "csv"


def _load_dataset(dataset: str) -> pd.DataFrame:
    """Load a CSV dataset by name."""
    path = CSV_DIR / f"{dataset}.csv"
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {e}")


def _compute_skew(series: pd.Series) -> float:
    if _HAS_SCIPY:
        return float(_scipy_skew(series.dropna(), nan_policy="omit"))
    vals = series.dropna()
    n = len(vals)
    if n < 3:
        return 0.0
    m = vals.mean()
    s = vals.std()
    if s == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * ((vals - m) / s).pow(3).sum())


def _compute_kurtosis(series: pd.Series) -> float:
    if _HAS_SCIPY:
        return float(_scipy_kurtosis(series.dropna(), nan_policy="omit"))
    vals = series.dropna()
    n = len(vals)
    if n < 4:
        return 0.0
    m = vals.mean()
    s = vals.std()
    if s == 0:
        return 0.0
    m4 = ((vals - m) / s).pow(4).mean()
    return float(m4 - 3.0)


def _correlations(df: pd.DataFrame, threshold: float = 0.7) -> list[dict[str, Any]]:
    """Find pairwise correlations above threshold among numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr()
    results = []
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            val = corr.loc[c1, c2]
            if not np.isnan(val) and abs(val) > threshold:
                results.append({"col1": c1, "col2": c2, "correlation": round(float(val), 4)})
    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results


def _distributions(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute distribution stats for each numeric column."""
    numeric = df.select_dtypes(include="number")
    results = []
    for col in numeric.columns:
        s = numeric[col]
        results.append({
            "column": col,
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "skew": round(_compute_skew(s), 4),
            "kurtosis": round(_compute_kurtosis(s), 4),
        })
    return results


def _outliers(df: pd.DataFrame, z_threshold: float = 2.0) -> list[dict[str, Any]]:
    """Flag values more than z_threshold std devs from mean."""
    numeric = df.select_dtypes(include="number")
    results = []
    for col in numeric.columns:
        s = numeric[col]
        mean = s.mean()
        std = s.std()
        if std == 0 or np.isnan(std):
            continue
        z_scores = ((s - mean) / std).abs()
        mask = z_scores > z_threshold
        for idx in s[mask].index:
            results.append({
                "column": col,
                "value": round(float(s[idx]), 4),
                "row_index": int(idx),
                "z_score": round(float(z_scores[idx]), 4),
            })
    results.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    return results


def _full_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Run complete pattern detection on a dataframe."""
    corrs = _correlations(df)
    dists = _distributions(df)
    outs = _outliers(df)
    return {
        "correlations": corrs,
        "distributions": dists,
        "outliers": outs,
        "total_patterns": len(corrs) + len(dists) + len(outs),
    }


# ----- Routes (order matters: /all must come before /{dataset}) -----

@router.get("/patterns/all")
async def patterns_all():
    """Run pattern detection across ALL datasets, return summary with top findings."""
    if not CSV_DIR.exists():
        raise HTTPException(status_code=400, detail=f"Data directory not found: {CSV_DIR}")
    files = sorted(CSV_DIR.glob("*.csv"))
    if not files:
        raise HTTPException(status_code=400, detail="No datasets found")

    summary: dict[str, Any] = {"datasets": {}, "top_correlations": [], "top_outliers": [], "total_datasets": 0}
    all_corrs: list[dict] = []
    all_outliers: list[dict] = []

    for f in files:
        name = f.stem
        try:
            df = pd.read_csv(f)
            analysis = _full_analysis(df)
            summary["datasets"][name] = {
                "total_patterns": analysis["total_patterns"],
                "num_correlations": len(analysis["correlations"]),
                "num_outliers": len(analysis["outliers"]),
                "num_columns": len(analysis["distributions"]),
            }
            for c in analysis["correlations"]:
                all_corrs.append({**c, "dataset": name})
            for o in analysis["outliers"]:
                all_outliers.append({**o, "dataset": name})
        except Exception:
            summary["datasets"][name] = {"error": "Failed to analyze"}

    summary["total_datasets"] = len(files)
    all_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    all_outliers.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    summary["top_correlations"] = all_corrs[:20]
    summary["top_outliers"] = all_outliers[:20]
    return summary


@router.get("/patterns/{dataset}")
async def patterns_dataset(dataset: str):
    """Auto-discover patterns in a dataset."""
    df = _load_dataset(dataset)
    return _full_analysis(df)


@router.get("/patterns/{dataset}/correlations")
async def patterns_correlations(dataset: str):
    """Return correlation matrix for a dataset."""
    df = _load_dataset(dataset)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"matrix": {}, "strong_pairs": []}
    corr = numeric.corr()
    return {
        "matrix": {col: {r: round(float(corr.loc[r, col]), 4) for r in corr.index} for col in corr.columns},
        "strong_pairs": _correlations(df),
    }


@router.get("/patterns/{dataset}/distributions")
async def patterns_distributions(dataset: str):
    """Return distribution stats for a dataset."""
    df = _load_dataset(dataset)
    return {"distributions": _distributions(df)}

"""Excel-like spreadsheet interface for Klaus IMI CSV datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/klaus/imi", tags=["imi-spreadsheet"])

CSV_DIR = Path.home() / ".axe" / "klaus_data" / "csv"


def _load_dataset(dataset: str) -> pd.DataFrame:
    path = CSV_DIR / f"{dataset}.csv"
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset}' not found")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {e}")


# --- Models ---

class FilterItem(BaseModel):
    column: str
    operator: str  # eq, ne, gt, lt, gte, lte, contains, startswith
    value: str | int | float

class FilterRequest(BaseModel):
    dataset: str
    filters: List[FilterItem]
    sort_by: Optional[str] = None
    ascending: Optional[bool] = True

class MetricItem(BaseModel):
    column: str
    func: str  # mean, sum, count, min, max, std

class AggregateRequest(BaseModel):
    dataset: str
    group_by: str | List[str]
    metrics: List[MetricItem]

class PivotRequest(BaseModel):
    dataset: str
    index: str | List[str]
    columns: str | List[str]
    values: str | List[str]
    aggfunc: str = "mean"


# --- Endpoints ---

@router.get("/spreadsheet/datasets")
def list_datasets():
    if not CSV_DIR.exists():
        return {"datasets": []}
    datasets = [f.stem for f in sorted(CSV_DIR.glob("*.csv"))]
    return {"datasets": datasets}


@router.get("/spreadsheet/{dataset}")
def get_dataset(dataset: str):
    df = _load_dataset(dataset)
    return {
        "columns": df.columns.tolist(),
        "rows": df.fillna("").values.tolist(),
        "total_rows": len(df),
    }


@router.post("/spreadsheet/pivot")
def pivot_table(req: PivotRequest):
    df = _load_dataset(req.dataset)
    try:
        result = pd.pivot_table(
            df,
            index=req.index,
            columns=req.columns,
            values=req.values,
            aggfunc=req.aggfunc,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pivot failed: {e}")
    result = result.reset_index()
    # Flatten multi-level columns if needed
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = ["_".join(str(c) for c in col).strip("_") for col in result.columns]
    return {
        "columns": result.columns.tolist(),
        "rows": result.fillna("").values.tolist(),
        "total_rows": len(result),
    }


@router.post("/spreadsheet/filter")
def filter_dataset(req: FilterRequest):
    df = _load_dataset(req.dataset)
    for f in req.filters:
        if f.column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{f.column}' not found")
        col = df[f.column]
        ops = {
            "eq": lambda c, v: c == v,
            "ne": lambda c, v: c != v,
            "gt": lambda c, v: c > v,
            "lt": lambda c, v: c < v,
            "gte": lambda c, v: c >= v,
            "lte": lambda c, v: c <= v,
            "contains": lambda c, v: c.astype(str).str.contains(str(v), case=False, na=False),
            "startswith": lambda c, v: c.astype(str).str.startswith(str(v), na=False),
        }
        if f.operator not in ops:
            raise HTTPException(status_code=400, detail=f"Unknown operator '{f.operator}'")
        mask = ops[f.operator](col, f.value)
        df = df[mask]
    if req.sort_by:
        if req.sort_by not in df.columns:
            raise HTTPException(status_code=400, detail=f"Sort column '{req.sort_by}' not found")
        df = df.sort_values(by=req.sort_by, ascending=req.ascending)
    return {
        "columns": df.columns.tolist(),
        "rows": df.fillna("").values.tolist(),
        "total_rows": len(df),
    }


@router.post("/spreadsheet/aggregate")
def aggregate_dataset(req: AggregateRequest):
    df = _load_dataset(req.dataset)
    group_by = req.group_by if isinstance(req.group_by, list) else [req.group_by]
    for col in group_by:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    agg_dict: dict[str, list[str]] = {}
    valid_funcs = {"mean", "sum", "count", "min", "max", "std"}
    for m in req.metrics:
        if m.column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{m.column}' not found")
        if m.func not in valid_funcs:
            raise HTTPException(status_code=400, detail=f"Unknown function '{m.func}'")
        agg_dict.setdefault(m.column, []).append(m.func)
    try:
        result = df.groupby(group_by).agg(agg_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Aggregation failed: {e}")
    # Flatten multi-level columns
    result.columns = ["_".join(col).strip("_") for col in result.columns]
    result = result.reset_index()
    return {
        "columns": result.columns.tolist(),
        "rows": result.fillna("").values.tolist(),
        "total_rows": len(result),
    }

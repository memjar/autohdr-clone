"""
IMI Notebook Engine
====================
Interactive notebook interface for Klaus IMI — supports query, python, and markdown cells.
Notebooks stored as JSON in ~/.axe/klaus_data/notebooks/
"""

import json
import os
import io
import sys
import time
from uuid import uuid4
from contextlib import redirect_stdout
from pathlib import Path

from imi_agents import run_pipeline

NOTEBOOKS_DIR = os.path.expanduser("~/.axe/klaus_data/notebooks")
CSV_DIR = os.path.expanduser("~/.axe/klaus_data/csv")

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _nb_path(notebook_id: str) -> str:
    return os.path.join(NOTEBOOKS_DIR, f"{notebook_id}.json")


def _load(notebook_id: str) -> dict | None:
    path = _nb_path(notebook_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(notebook: dict):
    notebook["updated_at"] = _now()
    with open(_nb_path(notebook["id"]), "w") as f:
        json.dump(notebook, f, indent=2)


# ── Public API ──────────────────────────────────────────


def create_notebook(title: str) -> dict:
    nb = {
        "id": str(uuid4()),
        "title": title,
        "cells": [],
        "created_at": _now(),
        "updated_at": _now(),
    }
    _save(nb)
    return nb


def get_notebook(notebook_id: str) -> dict | None:
    return _load(notebook_id)


def list_notebooks() -> list[dict]:
    results = []
    for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(NOTEBOOKS_DIR, fname)) as f:
                nb = json.load(f)
            results.append({
                "id": nb["id"],
                "title": nb["title"],
                "cell_count": len(nb["cells"]),
                "created_at": nb["created_at"],
                "updated_at": nb["updated_at"],
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def add_cell(notebook_id: str, cell_type: str, source: str) -> dict | None:
    nb = _load(notebook_id)
    if nb is None:
        return None
    if cell_type not in ("query", "python", "markdown"):
        raise ValueError(f"Invalid cell type: {cell_type}")
    cell = {
        "id": str(uuid4()),
        "type": cell_type,
        "source": source,
        "output": None,
        "created_at": _now(),
    }
    nb["cells"].append(cell)
    _save(nb)
    return cell


def execute_cell(notebook_id: str, cell_id: str) -> dict | None:
    nb = _load(notebook_id)
    if nb is None:
        return None
    cell = next((c for c in nb["cells"] if c["id"] == cell_id), None)
    if cell is None:
        return None

    if cell["type"] == "markdown":
        cell["output"] = cell["source"]

    elif cell["type"] == "query":
        # Run through IMI pipeline with empty RAG chunks (notebook context)
        try:
            cell["output"] = run_pipeline(cell["source"], rag_chunks=[])
        except Exception as e:
            cell["output"] = f"Pipeline error: {e}"

    elif cell["type"] == "python":
        cell["output"] = _exec_python(cell["source"])

    _save(nb)
    return cell


def delete_notebook(notebook_id: str) -> bool:
    path = _nb_path(notebook_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ── Python cell execution ───────────────────────────────


def _exec_python(source: str) -> str:
    """Execute python code in a restricted namespace with pandas/numpy/json."""
    import pandas as pd
    import numpy as np

    namespace = {
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "json": json,
        "CSV_DIR": CSV_DIR,
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "round": round,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "isinstance": isinstance,
            "type": type,
            "hasattr": hasattr,
            "getattr": getattr,
        },
    }

    # Pre-load CSVs from the csv directory
    if os.path.isdir(CSV_DIR):
        for fname in os.listdir(CSV_DIR):
            if fname.endswith(".csv"):
                var_name = fname.replace(".csv", "").replace("-", "_").replace(" ", "_")
                try:
                    namespace[var_name] = pd.read_csv(os.path.join(CSV_DIR, fname))
                except Exception:
                    pass

    stdout_capture = io.StringIO()
    try:
        with redirect_stdout(stdout_capture):
            exec(source, namespace)
        output = stdout_capture.getvalue()
        return output if output else "(no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

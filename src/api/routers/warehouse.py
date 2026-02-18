"""
DuckDB Data Warehouse for Klaus IMI.
Handles billion-row datasets on local Apple Silicon hardware.

Architecture:
  Upload (zip/csv/xlsx) → DuckDB on-disk database → NL→SQL → results + charts

DuckDB runs columnar analytics in-process — no server needed.
On M1 Max 64GB it can scan 1B+ rows in seconds via memory-mapped IO.
"""

import io
import os
import re
import json
import base64
import hashlib
import logging
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import OLLAMA_BASE, IMI_MODEL, AUDIT_LOG_PATH

logger = logging.getLogger("imi_warehouse")

router = APIRouter(prefix="/klaus/imi/warehouse", tags=["imi-warehouse"])

# ---------------------------------------------------------------------------
# DuckDB persistent database
# ---------------------------------------------------------------------------

WAREHOUSE_DIR = Path.home() / ".axe" / "klaus_data" / "warehouse"
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = WAREHOUSE_DIR / "imi.duckdb"

_conn = duckdb.connect(str(DB_PATH))
_conn.execute("SET memory_limit='16GB'")
_conn.execute("SET threads=8")

# Track table metadata
_table_meta: dict[str, dict] = {}  # table_name -> {columns, row_count, size_mb, created}


def _refresh_meta():
    """Sync _table_meta with actual DuckDB tables."""
    _table_meta.clear()
    try:
        tables = _conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        for (tbl,) in tables:
            cols = _conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{tbl}'"
            ).fetchall()
            count = _conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            _table_meta[tbl] = {
                "columns": [{"name": c, "type": t} for c, t in cols],
                "row_count": count,
            }
    except Exception as e:
        logger.error("Failed to refresh metadata: %s", e)


_refresh_meta()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Make a safe table name from a filename."""
    base = Path(name).stem
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", base).lower().strip("_")
    return clean or "uploaded_data"


def _audit_log(action: str, detail: str, success: bool, error: str | None = None):
    try:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "type": "warehouse",
            "action": action,
            "detail": detail,
            "success": success,
            "error": error,
        }
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _call_ollama(system: str, user: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": IMI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": max_tokens},
        },
        timeout=600.0,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Schema prompt for SQL generation
# ---------------------------------------------------------------------------

def _build_schema_prompt() -> str:
    lines = ["Available DuckDB tables:\n"]
    for tbl, meta in _table_meta.items():
        cols = ", ".join(f'{c["name"]} ({c["type"]})' for c in meta["columns"])
        lines.append(f'  "{tbl}" ({meta["row_count"]:,} rows): {cols}')
    return "\n".join(lines) if len(lines) > 1 else "No tables loaded yet."


def _sql_system_prompt() -> str:
    return f"""You are a SQL generator for marketing research data in DuckDB.

{_build_schema_prompt()}

Rules:
- Output ONLY a single valid DuckDB SQL SELECT statement. No markdown, no explanation, no backticks.
- All column and table names must be double-quoted.
- Use DuckDB functions: TRY_CAST() for type conversion, LIST_AGG, QUANTILE, etc.
- LIMIT 200 unless the user asks for a specific count or says "all".
- For aggregations, always include a GROUP BY.
- Never use DELETE, UPDATE, INSERT, DROP, ALTER, or CREATE.
- Use approximate counts (APPROX_COUNT_DISTINCT) for large tables over 1M rows.
- For percentage calculations, use ROUND(... * 100.0, 1) for readability.
"""


# ---------------------------------------------------------------------------
# File ingestion
# ---------------------------------------------------------------------------

def _ingest_csv(filepath: str, table_name: str) -> dict:
    """Load a CSV file into DuckDB."""
    _conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    _conn.execute(f"""
        CREATE TABLE "{table_name}" AS
        SELECT * FROM read_csv_auto('{filepath}',
            header=true,
            sample_size=10000,
            ignore_errors=true,
            all_varchar=false
        )
    """)
    count = _conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    cols = _conn.execute(
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name}'"
    ).fetchall()
    return {"table": table_name, "rows": count, "columns": [{"name": c, "type": t} for c, t in cols]}


def _ingest_parquet(filepath: str, table_name: str) -> dict:
    """Load a Parquet file into DuckDB."""
    _conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    _conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM read_parquet(\'{filepath}\')')
    count = _conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    cols = _conn.execute(
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name}'"
    ).fetchall()
    return {"table": table_name, "rows": count, "columns": [{"name": c, "type": t} for c, t in cols]}


def _ingest_excel(filepath: str, table_name: str) -> dict:
    """Load an Excel file into DuckDB via pandas (fallback)."""
    import pandas as pd
    df = pd.read_excel(filepath, engine="openpyxl")
    # Clean column names
    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", str(c)).strip("_") for c in df.columns]
    _conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    _conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')
    count = _conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    cols = _conn.execute(
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name}'"
    ).fetchall()
    return {"table": table_name, "rows": count, "columns": [{"name": c, "type": t} for c, t in cols]}


INGESTORS = {
    ".csv": _ingest_csv,
    ".tsv": _ingest_csv,
    ".parquet": _ingest_parquet,
    ".xlsx": _ingest_excel,
    ".xls": _ingest_excel,
}


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _generate_chart(data: list[dict], chart_type: str = "bar", title: str = "") -> str | None:
    """Generate a chart as base64 PNG. Returns None if matplotlib unavailable or data empty."""
    if not data:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = list(data[0].keys())
        if len(keys) < 2:
            return None

        label_col = keys[0]
        value_cols = keys[1:]

        labels = [str(row[label_col]) for row in data]
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            for vc in value_cols:
                values = []
                for row in data:
                    try:
                        values.append(float(row[vc]))
                    except (ValueError, TypeError):
                        values.append(0)
                ax.bar(range(len(labels)), values, label=vc, alpha=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        elif chart_type == "line":
            for vc in value_cols:
                values = []
                for row in data:
                    try:
                        values.append(float(row[vc]))
                    except (ValueError, TypeError):
                        values.append(0)
                ax.plot(labels, values, marker="o", label=vc)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        elif chart_type == "pie":
            values = []
            for row in data:
                try:
                    values.append(float(row[value_cols[0]]))
                except (ValueError, TypeError):
                    values.append(0)
            ax.pie(values, labels=labels, autopct="%1.1f%%")

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        if chart_type != "pie" and len(value_cols) > 1:
            ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_to_warehouse(file: UploadFile = File(...), table_name: Optional[str] = None):
    """
    Upload a data file (CSV, Excel, Parquet, or ZIP containing them) into DuckDB.
    For 1B+ row datasets, use CSV or Parquet for best performance.
    """
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    tbl = _sanitize_name(table_name or filename)

    content = await file.read()
    results = []

    if ext == ".zip":
        # Extract ZIP and ingest each data file
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                member_ext = os.path.splitext(member)[1].lower()
                if member_ext not in INGESTORS:
                    continue
                # Skip macOS resource forks and hidden files
                if member.startswith("__MACOSX") or os.path.basename(member).startswith("."):
                    continue
                with tempfile.NamedTemporaryFile(suffix=member_ext, delete=False) as tmp:
                    tmp.write(zf.read(member))
                    tmp_path = tmp.name
                try:
                    member_tbl = _sanitize_name(member)
                    info = INGESTORS[member_ext](tmp_path, member_tbl)
                    results.append(info)
                    logger.info("Ingested %s → %s (%d rows)", member, member_tbl, info["rows"])
                except Exception as e:
                    logger.error("Failed to ingest %s: %s", member, e)
                    results.append({"table": _sanitize_name(member), "error": str(e)})
                finally:
                    os.unlink(tmp_path)
    elif ext in INGESTORS:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            info = INGESTORS[ext](tmp_path, tbl)
            results.append(info)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to ingest: {e}")
        finally:
            os.unlink(tmp_path)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Use CSV, TSV, Parquet, Excel, or ZIP."
        )

    _refresh_meta()
    total_rows = sum(r.get("rows", 0) for r in results if "error" not in r)
    _audit_log("upload", f"{filename} → {len(results)} tables, {total_rows:,} rows", True)

    return {
        "status": "success",
        "filename": filename,
        "tables": results,
        "total_rows": total_rows,
        "message": f"Loaded {total_rows:,} rows into {len(results)} table(s). Ready for analysis.",
    }


@router.get("/tables")
def list_warehouse_tables():
    """List all tables in the warehouse with schema info."""
    _refresh_meta()
    tables = []
    for tbl, meta in _table_meta.items():
        tables.append({
            "name": tbl,
            "row_count": meta["row_count"],
            "columns": meta["columns"],
        })
    total_rows = sum(t["row_count"] for t in tables)
    return {"tables": tables, "total_rows": total_rows}


@router.get("/tables/{table_name}/sample")
def sample_table(table_name: str, limit: int = 10):
    """Return sample rows from a table."""
    if table_name not in _table_meta:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")
    rows = _conn.execute(f'SELECT * FROM "{table_name}" LIMIT {min(limit, 100)}').fetchdf().to_dict("records")
    return {"table": table_name, "sample": rows, "row_count": _table_meta[table_name]["row_count"]}


@router.get("/tables/{table_name}/profile")
def profile_table(table_name: str):
    """Generate a statistical profile of a table — distributions, nulls, uniques."""
    if table_name not in _table_meta:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")

    meta = _table_meta[table_name]
    profile = []
    for col_info in meta["columns"]:
        col = col_info["name"]
        dtype = col_info["type"]
        try:
            stats = _conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT("{col}") as non_null,
                    COUNT(DISTINCT "{col}") as unique_vals
                FROM "{table_name}"
            """).fetchone()

            col_profile = {
                "column": col,
                "type": dtype,
                "total": stats[0],
                "non_null": stats[1],
                "null_pct": round((stats[0] - stats[1]) / max(stats[0], 1) * 100, 1),
                "unique": stats[2],
            }

            # Numeric stats
            if any(t in dtype.upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "BIGINT", "NUMERIC"]):
                num_stats = _conn.execute(f"""
                    SELECT
                        MIN("{col}") as min_val,
                        MAX("{col}") as max_val,
                        AVG("{col}") as mean,
                        MEDIAN("{col}") as median,
                        STDDEV("{col}") as stddev
                    FROM "{table_name}"
                """).fetchone()
                col_profile.update({
                    "min": num_stats[0],
                    "max": num_stats[1],
                    "mean": round(float(num_stats[2]), 2) if num_stats[2] is not None else None,
                    "median": num_stats[3],
                    "stddev": round(float(num_stats[4]), 2) if num_stats[4] is not None else None,
                })

            # Top values for categorical columns (low cardinality)
            if col_profile["unique"] <= 50:
                top = _conn.execute(f"""
                    SELECT "{col}" as val, COUNT(*) as cnt
                    FROM "{table_name}"
                    WHERE "{col}" IS NOT NULL
                    GROUP BY "{col}"
                    ORDER BY cnt DESC
                    LIMIT 10
                """).fetchall()
                col_profile["top_values"] = [{"value": v, "count": c} for v, c in top]

            profile.append(col_profile)
        except Exception as e:
            profile.append({"column": col, "type": dtype, "error": str(e)})

    return {
        "table": table_name,
        "row_count": meta["row_count"],
        "column_count": len(meta["columns"]),
        "profile": profile,
    }


class QueryRequest(BaseModel):
    query: str
    chart: Optional[str] = None  # "bar", "line", "pie", or None


class SQLDirectRequest(BaseModel):
    sql: str
    chart: Optional[str] = None


@router.post("/query")
def warehouse_query(req: QueryRequest):
    """
    Natural language query against the warehouse.
    Generates SQL via LLM, executes on DuckDB, returns results + optional chart.
    """
    if not _table_meta:
        return JSONResponse(
            {"error": "No data loaded. Upload data first via POST /klaus/imi/warehouse/upload"},
            status_code=400,
        )

    # 1. Generate SQL
    try:
        raw_sql = _call_ollama(_sql_system_prompt(), req.query).strip()
    except Exception as e:
        _audit_log("query", req.query, False, str(e))
        return JSONResponse({"error": f"LLM error: {e}"}, status_code=500)

    # Strip markdown fences
    sql = raw_sql.strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()

    # Safety: only SELECT/WITH
    first_word = sql.split()[0].upper() if sql.split() else ""
    if first_word not in ("SELECT", "WITH"):
        _audit_log("query", req.query, False, "Non-SELECT blocked")
        return JSONResponse({"error": "Only SELECT queries are allowed.", "sql": sql}, status_code=400)

    # 2. Execute
    try:
        result = _conn.execute(sql)
        cols = [desc[0] for desc in result.description]
        rows = result.fetchmany(500)
        data = [dict(zip(cols, row)) for row in rows]
        total = _conn.execute(f"SELECT COUNT(*) FROM ({sql}) _sub").fetchone()[0] if len(rows) == 500 else len(rows)
    except duckdb.Error as e:
        _audit_log("query", req.query, False, str(e))
        return JSONResponse({"error": f"SQL error: {e}", "sql": sql}, status_code=400)

    # 3. Generate chart if requested or auto-detect
    chart_b64 = None
    chart_type = req.chart
    if not chart_type and len(data) >= 2:
        # Auto-detect: if first column is text and second is numeric, chart it
        if data and len(data[0]) >= 2:
            vals = list(data[0].values())
            if isinstance(vals[0], str) and isinstance(vals[1], (int, float)):
                chart_type = "bar"

    if chart_type and data:
        chart_b64 = _generate_chart(data[:30], chart_type, title=req.query[:80])

    # 4. Generate explanation
    try:
        explanation = _call_ollama(
            "You are a concise market research analyst at IMI International. "
            "Explain what this data shows and highlight key insights in 2-3 sentences. "
            "Focus on patterns, outliers, and actionable findings.",
            f"Question: {req.query}\nSQL: {sql}\nTotal rows: {total}\nFirst 5 rows: {json.dumps(data[:5], default=str)}",
            max_tokens=300,
        )
    except Exception:
        explanation = f"Query returned {total:,} rows."

    _audit_log("query", req.query, True)

    response = {
        "sql": sql,
        "results": data,
        "total_rows": total,
        "returned_rows": len(data),
        "explanation": explanation,
        "columns": cols,
    }
    if chart_b64:
        response["chart"] = chart_b64
        response["chart_type"] = chart_type

    return response


@router.post("/sql")
def warehouse_direct_sql(req: SQLDirectRequest):
    """Execute raw SQL directly (for power users or follow-up queries)."""
    sql = req.sql.strip()
    first_word = sql.split()[0].upper() if sql.split() else ""
    if first_word not in ("SELECT", "WITH", "DESCRIBE", "SHOW", "EXPLAIN"):
        return JSONResponse({"error": "Only read-only queries allowed."}, status_code=400)

    try:
        result = _conn.execute(sql)
        cols = [desc[0] for desc in result.description]
        rows = result.fetchmany(500)
        data = [dict(zip(cols, row)) for row in rows]
    except duckdb.Error as e:
        return JSONResponse({"error": f"SQL error: {e}"}, status_code=400)

    chart_b64 = None
    if req.chart and data:
        chart_b64 = _generate_chart(data[:30], req.chart)

    response = {"sql": sql, "results": data, "row_count": len(data), "columns": cols}
    if chart_b64:
        response["chart"] = chart_b64
    return response


@router.delete("/tables/{table_name}")
def drop_table(table_name: str):
    """Drop a table from the warehouse."""
    if table_name not in _table_meta:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")
    _conn.execute(f'DROP TABLE "{table_name}"')
    _refresh_meta()
    _audit_log("drop_table", table_name, True)
    return {"status": "dropped", "table": table_name}


@router.get("/stats")
def warehouse_stats():
    """Overall warehouse statistics."""
    _refresh_meta()
    total_rows = sum(m["row_count"] for m in _table_meta.values())
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024) if DB_PATH.exists() else 0
    return {
        "tables": len(_table_meta),
        "total_rows": total_rows,
        "db_size_mb": round(db_size_mb, 1),
        "db_path": str(DB_PATH),
    }

"""
SQL Generation + Transparency Layer for Klaus IMI.
Natural language -> SQL -> results against in-memory SQLite loaded from CSVs.
"""

import csv
import json
import logging
import os
import sqlite3

from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel

from config import IMI_DATA_DIR, OLLAMA_BASE, AUDIT_LOG_PATH, IMI_MODEL

import httpx

logger = logging.getLogger("imi_sql")

router = APIRouter(prefix="/klaus/imi", tags=["imi-sql"])

# ---------------------------------------------------------------------------
# In-memory SQLite — loaded once at import time
# ---------------------------------------------------------------------------

_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.row_factory = sqlite3.Row
_table_meta: dict[str, list[str]] = {}  # table_name -> [col1, col2, ...]


def _sanitize_table_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    return name.replace("-", "_").replace(" ", "_").lower()


def _load_csvs():
    """Read every CSV in IMI_DATA_DIR into SQLite."""
    if not os.path.isdir(IMI_DATA_DIR):
        logger.warning("IMI_DATA_DIR not found: %s", IMI_DATA_DIR)
        return
    cur = _conn.cursor()
    for fname in sorted(os.listdir(IMI_DATA_DIR)):
        if not fname.lower().endswith(".csv"):
            continue
        table = _sanitize_table_name(fname)
        path = os.path.join(IMI_DATA_DIR, fname)
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                headers = [h.strip().replace(" ", "_").replace("-", "_") for h in next(reader)]
                cols_ddl = ", ".join(f'"{h}" TEXT' for h in headers)
                cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_ddl})')
                placeholders = ", ".join("?" for _ in headers)
                rows = list(reader)
                if rows:
                    cur.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', rows)
                _table_meta[table] = headers
                logger.info("Loaded %s: %d rows, %d cols", table, len(rows), len(headers))
        except Exception as e:
            logger.error("Failed to load %s: %s", fname, e)
    _conn.commit()


_load_csvs()

# ---------------------------------------------------------------------------
# Ollama helper (sync, matching imi_agents pattern)
# ---------------------------------------------------------------------------

_SQL_MODEL = IMI_MODEL


def _call_ollama(system: str, user: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": _SQL_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": max_tokens},
        },
        timeout=600.0,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Schema description for the LLM
# ---------------------------------------------------------------------------

def _build_schema_prompt() -> str:
    lines = ["Available SQLite tables and their columns:\n"]
    for table, cols in _table_meta.items():
        lines.append(f"  {table}: {', '.join(cols)}")
    return "\n".join(lines)


_SYSTEM_PROMPT = f"""You are a SQL generator for marketing research data stored in SQLite.

{_build_schema_prompt()}

Rules:
- Output ONLY a single valid SQLite SELECT statement. No markdown, no explanation, no backticks.
- All column and table names must be double-quoted.
- Use CAST() when doing numeric comparisons on TEXT columns (e.g. CAST("NPS" AS REAL) > 40).
- LIMIT 100 unless the user asks for a specific count.
- Never use DELETE, UPDATE, INSERT, DROP, ALTER, or CREATE.
"""

# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def _audit_log(query: str, sql: str, success: bool, error: str | None = None):
    try:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "type": "sql",
            "query": query,
            "sql": sql,
            "success": success,
            "error": error,
        }
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class SQLRequest(BaseModel):
    query: str


class SQLResponse(BaseModel):
    sql: str
    results: list[dict]
    explanation: str
    row_count: int
    error: str | None = None


@router.post("/sql", response_model=SQLResponse)
def sql_query(req: SQLRequest):
    """Natural language -> SQL -> execute -> results."""

    # 1. Generate SQL via LLM
    try:
        raw_sql = _call_ollama(_SYSTEM_PROMPT, req.query).strip()
    except Exception as e:
        _audit_log(req.query, "", False, str(e))
        return SQLResponse(sql="", results=[], explanation="", row_count=0, error=f"LLM error: {e}")

    # Strip markdown fences if model wraps them
    sql = raw_sql.strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()

    # Safety: reject non-SELECT
    first_word = sql.split()[0].upper() if sql.split() else ""
    if first_word != "SELECT" and first_word != "WITH":
        _audit_log(req.query, sql, False, "Non-SELECT statement blocked")
        return SQLResponse(sql=sql, results=[], explanation="", row_count=0, error="Only SELECT queries are allowed.")

    # 2. Execute
    try:
        cur = _conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(100)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        results = [dict(zip(cols, row)) for row in rows]
        row_count = len(results)
    except sqlite3.Error as e:
        _audit_log(req.query, sql, False, str(e))
        return SQLResponse(sql=sql, results=[], explanation="", row_count=0, error=f"SQL error: {e}")

    # 3. Generate explanation
    try:
        explanation = _call_ollama(
            "You are a concise analyst. Explain what this SQL query does and summarize the results in 1-2 sentences.",
            f"Query: {req.query}\nSQL: {sql}\nRow count: {row_count}\nFirst 3 rows: {json.dumps(results[:3])}",
            max_tokens=256,
        )
    except Exception:
        explanation = f"Returned {row_count} rows."

    _audit_log(req.query, sql, True)
    return SQLResponse(sql=sql, results=results, explanation=explanation, row_count=row_count)


@router.get("/sql/tables")
def list_tables():
    """Return available tables and their columns."""
    return {
        "tables": [
            {"name": table, "columns": cols}
            for table, cols in _table_meta.items()
        ]
    }

"""
Live Warehouse Connections module for Klaus IMI.
Register, test, query external data sources (SQLite, CSV URL, local CSV).
"""

import json
import os
import sqlite3
import urllib.request
import io
from datetime import datetime, timezone
from uuid import uuid4
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import pandas as pd

router = APIRouter(prefix="/klaus/imi", tags=["imi-connections"])

CONNECTIONS_FILE = os.path.expanduser("~/.axe/klaus_data/connections.json")


# ── Models ──────────────────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    name: str
    type: Literal["sqlite", "csv_url", "local_csv"]
    connection_string: str


class QueryRequest(BaseModel):
    query: str


class ConnectionOut(BaseModel):
    id: str
    name: str
    type: str
    connection_string: str
    status: str
    created_at: str


# ── Persistence helpers ─────────────────────────────────────────────────

def _ensure_dir():
    os.makedirs(os.path.dirname(CONNECTIONS_FILE), exist_ok=True)


def _load_connections() -> list[dict]:
    if not os.path.exists(CONNECTIONS_FILE):
        return []
    with open(CONNECTIONS_FILE, "r") as f:
        return json.load(f)


def _save_connections(conns: list[dict]):
    _ensure_dir()
    with open(CONNECTIONS_FILE, "w") as f:
        json.dump(conns, f, indent=2)


def _find_connection(source_id: str) -> tuple[list[dict], dict]:
    conns = _load_connections()
    for c in conns:
        if c["id"] == source_id:
            return conns, c
    raise HTTPException(status_code=404, detail=f"Source {source_id} not found")


# ── Connection testers ──────────────────────────────────────────────────

def _test_sqlite(cs: str) -> bool:
    conn = sqlite3.connect(cs)
    conn.execute("SELECT 1")
    conn.close()
    return True


def _test_csv_url(cs: str) -> bool:
    req = urllib.request.Request(cs, method="HEAD")
    resp = urllib.request.urlopen(req, timeout=10)
    return resp.status == 200


def _test_local_csv(cs: str) -> bool:
    if not os.path.isfile(cs):
        raise FileNotFoundError(f"File not found: {cs}")
    return True


def _test_connection(typ: str, cs: str) -> bool:
    try:
        if typ == "sqlite":
            return _test_sqlite(cs)
        elif typ == "csv_url":
            return _test_csv_url(cs)
        elif typ == "local_csv":
            return _test_local_csv(cs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection test failed: {e}")
    return False


# ── Query executors ─────────────────────────────────────────────────────

def _query_sqlite(cs: str, query: str) -> list[dict]:
    conn = sqlite3.connect(cs)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _load_csv_df(typ: str, cs: str) -> pd.DataFrame:
    if typ == "csv_url":
        with urllib.request.urlopen(cs, timeout=30) as resp:
            data = resp.read().decode("utf-8")
        return pd.read_csv(io.StringIO(data))
    else:
        return pd.read_csv(cs)


def _query_csv(typ: str, cs: str, query: str) -> list[dict]:
    df = _load_csv_df(typ, cs)
    result = df.query(query)
    return result.to_dict(orient="records")


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/connect", response_model=ConnectionOut)
async def connect_source(req: ConnectRequest):
    """Register a new data source and test connectivity."""
    _test_connection(req.type, req.connection_string)
    conn = {
        "id": str(uuid4()),
        "name": req.name,
        "type": req.type,
        "connection_string": req.connection_string,
        "status": "connected",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    conns = _load_connections()
    conns.append(conn)
    _save_connections(conns)
    return conn


@router.get("/sources", response_model=list[ConnectionOut])
async def list_sources():
    """List all registered data sources."""
    return _load_connections()


@router.get("/sources/{source_id}", response_model=ConnectionOut)
async def get_source(source_id: str):
    """Get details for a single source."""
    _, conn = _find_connection(source_id)
    return conn


@router.delete("/sources/{source_id}")
async def delete_source(source_id: str):
    """Remove a registered source."""
    conns, conn = _find_connection(source_id)
    conns.remove(conn)
    _save_connections(conns)
    return {"detail": "deleted", "id": source_id}


@router.post("/sources/{source_id}/query")
async def query_source(source_id: str, req: QueryRequest):
    """Execute a query against a connected source."""
    _, conn = _find_connection(source_id)
    typ, cs = conn["type"], conn["connection_string"]
    try:
        if typ == "sqlite":
            rows = _query_sqlite(cs, req.query)
        else:
            rows = _query_csv(typ, cs, req.query)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query failed: {e}")
    return {"source_id": source_id, "row_count": len(rows), "rows": rows}


@router.post("/sources/{source_id}/test")
async def test_source(source_id: str):
    """Test that a connection is still alive."""
    conns, conn = _find_connection(source_id)
    try:
        _test_connection(conn["type"], conn["connection_string"])
        conn["status"] = "connected"
    except HTTPException:
        conn["status"] = "disconnected"
        _save_connections(conns)
        raise
    _save_connections(conns)
    return {"source_id": source_id, "status": "connected"}

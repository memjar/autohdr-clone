"""
Tool Registry for Klaus (Ollama native tool-calling API).

Defines all tool schemas, executor functions, and the dispatcher.
Each tool arms local Qwen3 32B with capabilities similar to Claude AI.
"""

import json
import os
import re
import signal
import subprocess
import sys
from io import StringIO, BytesIO
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path security helpers
# ---------------------------------------------------------------------------

HOME = str(Path.home())

_READ_ALLOWED = [
    os.path.join(HOME, ".axe"),
    os.path.join(HOME, "klausimi-backend"),
    os.path.join(HOME, "Desktop/M1transfer"),
]

_WRITE_ALLOWED = [
    os.path.join(HOME, ".axe"),
    os.path.join(HOME, "Desktop/M1transfer"),
]

_DANGEROUS_COMMANDS = re.compile(
    r"(rm\s+-rf\s+/|sudo\s|mkfs|dd\s+if=|shutdown|reboot|halt|"
    r"chmod\s+-R\s+777\s+/|chown\s+-R\s|>\s*/dev/sd|format\s+c:)",
    re.IGNORECASE,
)


def _check_path(path: str, allowed: list[str]) -> str:
    """Resolve and validate a path against allowed prefixes. Returns resolved path or raises."""
    resolved = os.path.realpath(os.path.expanduser(path))
    for prefix in allowed:
        if resolved.startswith(os.path.realpath(prefix)):
            return resolved
    raise PermissionError(f"Path not allowed: {path}")


# ---------------------------------------------------------------------------
# Tool schemas (Ollama-compatible)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    # ── Tier 1: Core Intelligence ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Execute a read-only SQL query against loaded IMI survey datasets. Use this when the user asks quantitative questions about survey data, demographics, or statistics. Only SELECT and WITH (CTE) statements are allowed.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "The SQL query to execute (SELECT/WITH only)"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Use this when the user asks about current events, external facts, or anything not in the local datasets.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "num_results": {"type": "integer", "description": "Number of results to return (default 5)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and extract text content from a URL. Use this to read a web page after finding it via web_search, or when the user provides a URL to read.",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem. Allowed paths: ~/.axe/, ~/klausimi-backend/, ~/Desktop/M1transfer/. Use this to inspect code, config, memory files, or data.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Absolute or ~-relative path to read"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating directories if needed. Allowed paths: ~/.axe/, ~/Desktop/M1transfer/. Use this to save results, update memory, or create files.",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string", "description": "Absolute or ~-relative path to write"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code in a restricted sandbox. Allowed modules: math, statistics, json, csv, datetime, collections, itertools, re. Use this for calculations, data transformations, or quick scripts.",
            "parameters": {
                "type": "object",
                "required": ["code"],
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command with a 30-second timeout. Dangerous commands (rm -rf /, sudo, etc.) are blocked. Use this for git operations, file listings, or system checks.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
            },
        },
    },
    # ── Tier 2: IMI Power Tools ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_data",
            "description": "Semantic (RAG) search across all IMI datasets. Use this when the user asks a natural-language question about IMI data and you need to find relevant rows or passages.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "num_results": {"type": "integer", "description": "Number of results to return (default 10)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Run anomaly detection on IMI datasets. Use this when the user asks about outliers, unusual patterns, or data quality issues.",
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {
                    "dataset": {"type": "string", "description": "Specific dataset name to check (empty = all datasets)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_case_studies",
            "description": "Search IMI case studies by keyword or topic. Use this when the user asks about specific case studies, examples, or real-world applications from the IMI data.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query for case studies"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Create a chart (bar, line, pie, scatter) and return it as a base64-encoded PNG. Use this when the user asks for a visualization or graph of data.",
            "parameters": {
                "type": "object",
                "required": ["chart_type", "title", "data"],
                "properties": {
                    "chart_type": {"type": "string", "description": "Chart type: bar, line, pie, or scatter"},
                    "title": {"type": "string", "description": "Chart title"},
                    "data": {"type": "string", "description": "JSON array of {label, value} for bar/pie/line or {x, y} for scatter"},
                    "x_label": {"type": "string", "description": "X-axis label (optional)"},
                    "y_label": {"type": "string", "description": "Y-axis label (optional)"},
                },
            },
        },
    },
    # ── Tier 3: System & Memory ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read from persistent cloud memory (~/.axe/cloud-memory/). Use this to recall saved notes, context, or shared team data.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Relative path within cloud-memory (e.g. 'notes/topic.md')"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Write to persistent cloud memory with auto git commit+push. Use this to save important information for later sessions.",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string", "description": "Relative path within cloud-memory"},
                    "content": {"type": "string", "description": "Content to write"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a macOS desktop notification or an iMessage. Use this to alert the user or send a message to a contact.",
            "parameters": {
                "type": "object",
                "required": ["message", "type"],
                "properties": {
                    "message": {"type": "string", "description": "The notification or message text"},
                    "type": {"type": "string", "description": "'notification' for desktop alert, 'imessage' for iMessage"},
                    "recipient": {"type": "string", "description": "Phone number for iMessage (required if type=imessage)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory, optionally filtered by glob pattern. Allowed paths: ~/.axe/, ~/klausimi-backend/, ~/Desktop/M1transfer/.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "pattern": {"type": "string", "description": "Optional glob pattern (e.g. '*.py')"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by finding a string and replacing its first occurrence. Allowed paths: ~/.axe/, ~/Desktop/M1transfer/. Use this for targeted edits to existing files.",
            "parameters": {
                "type": "object",
                "required": ["path", "find", "replace"],
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "find": {"type": "string", "description": "String to find (first occurrence)"},
                    "replace": {"type": "string", "description": "Replacement string"},
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Executor functions
# ---------------------------------------------------------------------------

def _exec_sql_query(**kwargs) -> dict:
    try:
        query = kwargs["query"].strip()
        upper = query.upper().lstrip()
        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            return {"result": None, "error": "Only SELECT and WITH statements are allowed"}
        from routers.sql import _conn, _table_meta  # noqa: lazy import
        if _conn is None:
            return {"result": None, "error": "SQL database not loaded"}
        cur = _conn.execute(query)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        result = [dict(zip(cols, row)) for row in rows[:500]]
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_web_search(**kwargs) -> dict:
    try:
        from duckduckgo_search import DDGS
        query = kwargs["query"]
        num = kwargs.get("num_results", 5)
        raw = DDGS().text(query, max_results=num)
        results = [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in raw]
        return {"result": results, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_web_fetch(**kwargs) -> dict:
    try:
        import httpx
        url = kwargs["url"]
        resp = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Klaus/1.0"})
        resp.raise_for_status()
        text = re.sub(r"<script[^>]*>.*?</script>", "", resp.text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return {"result": text[:20000], "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_read_file(**kwargs) -> dict:
    try:
        resolved = _check_path(kwargs["path"], _READ_ALLOWED)
        content = Path(resolved).read_text(errors="replace")
        return {"result": content[:50000], "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_write_file(**kwargs) -> dict:
    try:
        resolved = _check_path(kwargs["path"], _WRITE_ALLOWED)
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        Path(resolved).write_text(kwargs["content"])
        return {"result": f"Written {len(kwargs['content'])} chars to {resolved}", "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_run_python(**kwargs) -> dict:
    try:
        code = kwargs["code"]
        # Block dangerous patterns
        for forbidden in ("os.system", "subprocess", "shutil.rmtree", "__import__", "eval(", "exec(", "open("):
            if forbidden in code:
                return {"result": None, "error": f"Forbidden pattern: {forbidden}"}

        import math, statistics, datetime, collections, itertools, csv
        safe_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range, "int": int, "float": float,
                "str": str, "list": list, "dict": dict, "tuple": tuple, "set": set,
                "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map,
                "filter": filter, "sum": sum, "min": min, "max": max, "abs": abs,
                "round": round, "isinstance": isinstance, "type": type, "bool": bool,
                "True": True, "False": False, "None": None,
            },
            "math": math,
            "statistics": statistics,
            "json": json,
            "csv": csv,
            "datetime": datetime,
            "collections": collections,
            "itertools": itertools,
            "re": re,
        }

        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        try:
            signal.alarm(30)
            exec(code, safe_globals)
            signal.alarm(0)
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()
        return {"result": output if output else "(no output)", "error": None}
    except Exception as e:
        sys.stdout = sys.__stdout__
        signal.alarm(0)
        return {"result": None, "error": str(e)}


def _exec_run_shell(**kwargs) -> dict:
    try:
        command = kwargs["command"]
        if _DANGEROUS_COMMANDS.search(command):
            return {"result": None, "error": "Command blocked for safety"}
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr] {proc.stderr}"
        return {"result": output[:50000], "error": None}
    except subprocess.TimeoutExpired:
        return {"result": None, "error": "Command timed out after 30 seconds"}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_search_data(**kwargs) -> dict:
    try:
        from imi_rag import query as rag_query
        q = kwargs["query"]
        n = kwargs.get("num_results", 10)
        results = rag_query(q, n=n)
        return {"result": results, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_detect_anomalies(**kwargs) -> dict:
    try:
        dataset = kwargs.get("dataset", "")
        if dataset:
            from imi_anomaly import detect_anomalies_for_dataset
            results = detect_anomalies_for_dataset(dataset)
        else:
            from imi_anomaly import detect_anomalies as _detect
            results = _detect()
        return {"result": results, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_search_case_studies(**kwargs) -> dict:
    try:
        from imi_case_studies import search_studies
        q = kwargs["query"]
        n = kwargs.get("num_results", 5)
        results = search_studies(q, n=n)
        return {"result": results, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_generate_chart(**kwargs) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64

        chart_type = kwargs["chart_type"]
        title = kwargs["title"]
        data = json.loads(kwargs["data"])
        x_label = kwargs.get("x_label", "")
        y_label = kwargs.get("y_label", "")

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333333")

        if chart_type == "bar":
            labels = [d["label"] for d in data]
            values = [d["value"] for d in data]
            ax.bar(labels, values, color="#00ff00")
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "line":
            labels = [d["label"] for d in data]
            values = [d["value"] for d in data]
            ax.plot(labels, values, color="#00ff00", marker="o")
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "pie":
            labels = [d["label"] for d in data]
            values = [d["value"] for d in data]
            ax.pie(values, labels=labels, autopct="%1.1f%%", textprops={"color": "white"})
        elif chart_type == "scatter":
            xs = [d["x"] for d in data]
            ys = [d["y"] for d in data]
            ax.scatter(xs, ys, color="#00ff00", alpha=0.7)

        ax.set_title(title, fontsize=14, fontweight="bold")
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, facecolor="#111111")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return {"result": b64, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_memory_read(**kwargs) -> dict:
    try:
        rel = kwargs["path"].lstrip("/")
        full = os.path.join(HOME, ".axe", "cloud-memory", rel)
        full = os.path.realpath(full)
        if not full.startswith(os.path.realpath(os.path.join(HOME, ".axe", "cloud-memory"))):
            return {"result": None, "error": "Path traversal blocked"}
        content = Path(full).read_text(errors="replace")
        return {"result": content, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_memory_write(**kwargs) -> dict:
    try:
        rel = kwargs["path"].lstrip("/")
        base = os.path.join(HOME, ".axe", "cloud-memory")
        full = os.path.realpath(os.path.join(base, rel))
        if not full.startswith(os.path.realpath(base)):
            return {"result": None, "error": "Path traversal blocked"}
        Path(full).parent.mkdir(parents=True, exist_ok=True)
        Path(full).write_text(kwargs["content"])
        # Attempt git commit+push
        try:
            subprocess.run(
                f"cd {base} && git add -A && git commit -m 'memory: update {rel}' && git push",
                shell=True, capture_output=True, timeout=15,
            )
        except Exception:
            pass  # Non-critical if git push fails
        return {"result": f"Written to cloud-memory/{rel}", "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_send_notification(**kwargs) -> dict:
    try:
        message = kwargs["message"]
        ntype = kwargs["type"]
        if ntype == "notification":
            escaped = message.replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e", f'display notification "{escaped}" with title "Klaus"'],
                capture_output=True, timeout=10,
            )
            return {"result": "Notification sent", "error": None}
        elif ntype == "imessage":
            recipient = kwargs.get("recipient", "")
            if not recipient:
                return {"result": None, "error": "recipient required for imessage"}
            script = os.path.join(HOME, ".axe", "applescript", "imessage.sh")
            subprocess.run([script, recipient, message], capture_output=True, timeout=15)
            return {"result": f"iMessage sent to {recipient}", "error": None}
        else:
            return {"result": None, "error": f"Unknown type: {ntype}"}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_list_files(**kwargs) -> dict:
    try:
        import glob as globmod
        resolved = _check_path(kwargs["path"], _READ_ALLOWED)
        pattern = kwargs.get("pattern", "")
        if pattern:
            entries = globmod.glob(os.path.join(resolved, pattern))
            entries = [os.path.basename(e) for e in entries]
        else:
            entries = os.listdir(resolved)
        entries.sort()
        return {"result": entries, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _exec_edit_file(**kwargs) -> dict:
    try:
        resolved = _check_path(kwargs["path"], _WRITE_ALLOWED)
        content = Path(resolved).read_text()
        find = kwargs["find"]
        if find not in content:
            return {"result": None, "error": "String not found in file"}
        content = content.replace(find, kwargs["replace"], 1)
        Path(resolved).write_text(content)
        return {"result": f"Edited {resolved}", "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool map and dispatcher
# ---------------------------------------------------------------------------

TOOL_MAP: dict[str, Any] = {
    "sql_query": _exec_sql_query,
    "web_search": _exec_web_search,
    "web_fetch": _exec_web_fetch,
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "run_python": _exec_run_python,
    "run_shell": _exec_run_shell,
    "search_data": _exec_search_data,
    "detect_anomalies": _exec_detect_anomalies,
    "search_case_studies": _exec_search_case_studies,
    "generate_chart": _exec_generate_chart,
    "memory_read": _exec_memory_read,
    "memory_write": _exec_memory_write,
    "send_notification": _exec_send_notification,
    "list_files": _exec_list_files,
    "edit_file": _exec_edit_file,
}


def get_tools() -> list[dict]:
    """Return the full TOOLS list for passing to Ollama's tools parameter."""
    return TOOLS


def execute_tool(name: str, arguments: dict) -> dict:
    """Dispatch a tool call by name. Returns {"result": ..., "error": ...}."""
    executor = TOOL_MAP.get(name)
    if executor is None:
        return {"result": None, "error": f"Unknown tool: {name}"}
    try:
        return executor(**arguments)
    except Exception as e:
        return {"result": None, "error": f"Tool execution failed: {e}"}

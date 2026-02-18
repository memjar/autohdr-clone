"""
Shared Ollama client — one httpx client, one interface.
Eliminates duplicated httpx.post() calls across the codebase.
"""

import httpx
import logging
from config import (
    OLLAMA_BASE, OLLAMA_CHAT_TIMEOUT, OLLAMA_EMBED_TIMEOUT,
    OLLAMA_CONNECT_TIMEOUT, OLLAMA_STREAM_TIMEOUT, EMBED_MODEL,
)

logger = logging.getLogger("ollama_client")

# Async client for streaming endpoints
async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(OLLAMA_STREAM_TIMEOUT, connect=OLLAMA_CONNECT_TIMEOUT)
)

# Sync client for pipeline stages (batch training, agents)
sync_client = httpx.Client(
    timeout=httpx.Timeout(OLLAMA_CHAT_TIMEOUT, connect=OLLAMA_CONNECT_TIMEOUT)
)


def chat_sync(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    num_ctx: int = 16384,
    num_predict: int = 1024,
    stream: bool = False,
) -> str | dict:
    """Synchronous chat completion. Returns content string (non-stream) or raw response dict."""
    resp = sync_client.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "24h",
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        },
    )
    resp.raise_for_status()
    if stream:
        return resp.json()
    return resp.json().get("message", {}).get("content", "")


def embed_sync(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Get embedding vector from Ollama."""
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=OLLAMA_EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

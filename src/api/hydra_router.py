"""
Hydra Cascade Router — smart model selection.
Routes queries to the optimal model based on complexity.
Zero-latency classification using pre-compiled regex heuristics.

Architecture:
  simple  → qwen2.5:7b via Ollama  (~0.3s)
  complex → qwen3:32b via Hydra/Ollama
  code    → deepseek-coder:6.7b via Ollama
  data    → qwen3:32b via IMI RAG pipeline
"""
import re
import httpx

try:
    from config import OLLAMA_BASE, IMI_FAST_MODEL, IMI_MODEL, HYDRA_BASE
except ImportError:
    OLLAMA_BASE = "http://localhost:11434"
    IMI_FAST_MODEL = "qwen2.5:7b"
    IMI_MODEL = "qwen3:32b"
    HYDRA_BASE = "http://localhost:8080/v1"

# Classification heuristics (zero-latency, no LLM call)
SIMPLE_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|ok|yes|no|sure)\b",
    r"^what (is|are) \d",  # "what is 2+2"
    r"^(define|meaning of|what does .+ mean)",
    r"^(who|when|where) (is|was|are|were)\b",
]

CODE_PATTERNS = [
    r"\b(code|function|class|def |import |syntax|debug|error|traceback|bug|fix)\b",
    r"\b(python|javascript|typescript|rust|go|java|sql|html|css)\b",
    r"```",
]

DATA_PATTERNS = [
    r"\b(data|dataset|csv|analysis|analyze|chart|graph|trend|metric|kpi)\b",
    r"\b(say.?do|growth lever|brand health|funnel|nps|awareness|pulse)\b",
    r"\b(imi|segment|benchmark|roi|sponsorship|promotion)\b",
]

COMPLEX_PATTERNS = [
    r"\b(explain|compare|contrast|evaluate|recommend|strategy|plan|design)\b",
    r"\b(why|how does|what if|implications|consequences)\b",
]


def classify_query(query: str) -> str:
    """Classify query complexity. Returns: simple|complex|code|data"""
    q = query.lower().strip()

    # Short queries are almost always simple
    if len(q.split()) <= 5 and not any(re.search(p, q) for p in DATA_PATTERNS):
        for p in SIMPLE_PATTERNS:
            if re.search(p, q):
                return "simple"

    # Check data patterns first (highest priority for IMI)
    for p in DATA_PATTERNS:
        if re.search(p, q):
            return "data"

    # Code patterns
    for p in CODE_PATTERNS:
        if re.search(p, q):
            return "code"

    # Complex patterns
    for p in COMPLEX_PATTERNS:
        if re.search(p, q):
            return "complex"

    # Length heuristic: long queries are usually complex
    if len(q.split()) > 30:
        return "complex"

    # Default: simple for short, complex for medium+
    return "simple" if len(q.split()) <= 15 else "complex"


def get_model_for_query(query: str) -> dict:
    """Returns routing config for a query."""
    category = classify_query(query)

    routes = {
        "simple": {
            "model": IMI_FAST_MODEL,  # qwen2.5:7b
            "backend": "ollama",
            "base_url": OLLAMA_BASE,
            "category": "simple",
            "options": {"num_ctx": 2048, "num_predict": 256},
        },
        "complex": {
            "model": IMI_MODEL,  # qwen3:32b
            "backend": "ollama",  # or "hydra" if vllm-mlx running
            "base_url": OLLAMA_BASE,
            "category": "complex",
            "options": {"num_ctx": 4096, "num_predict": 1024},
        },
        "code": {
            "model": "deepseek-coder:6.7b",
            "backend": "ollama",
            "base_url": OLLAMA_BASE,
            "category": "code",
            "options": {"num_ctx": 4096, "num_predict": 1024},
        },
        "data": {
            "model": IMI_MODEL,
            "backend": "ollama",
            "base_url": OLLAMA_BASE,
            "category": "data",
            "options": {"num_ctx": 8192, "num_predict": 2048},
        },
    }

    route = routes[category]

    # Check if Hydra (vllm-mlx) is available for complex queries
    if category in ("complex", "data"):
        try:
            resp = httpx.get(f"{HYDRA_BASE}/models", timeout=1.0)
            if resp.status_code == 200:
                route["backend"] = "hydra"
                route["base_url"] = HYDRA_BASE
        except Exception:
            pass  # Fall back to Ollama

    return route


async def hydra_chat(messages: list, query: str, system_prompt: str = None, stream: bool = True):
    """Route a chat request through the optimal model."""
    route = get_model_for_query(query)

    if route["backend"] == "hydra":
        # Use OpenAI-compatible API
        payload = {
            "model": route["model"],
            "messages": messages,
            "max_tokens": route["options"].get("num_predict", 1024),
            "stream": stream,
        }
        async with httpx.AsyncClient(timeout=300) as client:
            if stream:
                async with client.stream("POST", f"{route['base_url']}/chat/completions", json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            yield line[6:]
            else:
                resp = await client.post(f"{route['base_url']}/chat/completions", json=payload)
                yield resp.json()
    else:
        # Use Ollama API
        ollama_payload = {
            "model": route["model"],
            "messages": messages,
            "stream": stream,
            "options": route["options"],
        }
        # Inject think:false for qwen3 models
        if "qwen3" in route["model"]:
            ollama_payload["think"] = False

        if system_prompt:
            ollama_payload["messages"] = [{"role": "system", "content": system_prompt}] + messages

        async with httpx.AsyncClient(timeout=300) as client:
            if stream:
                async with client.stream("POST", f"{route['base_url']}/api/chat", json=ollama_payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield line
            else:
                resp = await client.post(f"{route['base_url']}/api/chat", json=ollama_payload)
                yield resp.json()

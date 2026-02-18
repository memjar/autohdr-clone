"""
Centralized configuration — single source of truth.
Every constant lives here. Nothing is duplicated.
"""

import os

# Ollama
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
IMI_MODEL = os.environ.get("IMI_MODEL", "qwen3:32b")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "qwen3:32b")

# Timeouts (seconds)
OLLAMA_CHAT_TIMEOUT = 600.0
OLLAMA_EMBED_TIMEOUT = 30.0
OLLAMA_CONNECT_TIMEOUT = 10.0
OLLAMA_STREAM_TIMEOUT = 300.0

# Context windows
IMI_NUM_CTX = 4096
IMI_NUM_PREDICT = 1024

# Fast model for lightweight tasks (classification, etc.)
IMI_FAST_MODEL = os.environ.get("IMI_FAST_MODEL", "qwen2.5:7b")

# Hydra Engine (vllm-mlx)
HYDRA_BASE = os.environ.get("HYDRA_BASE", "http://localhost:8080/v1")
HYDRA_ENABLED = os.environ.get("HYDRA_ENABLED", "false").lower() == "true"

# Paths
IMI_DATA_DIR = os.path.expanduser("~/Desktop/M1transfer/klaus-chat/public/data/imi")
VECTOR_STORE_PATH = os.path.expanduser("~/.axe/klaus_data/imi_vectors.json")
LEARNING_LOG_PATH = os.path.expanduser("~/.axe/klaus_data/imi_learning_log.jsonl")
TRAINING_FILE_PATH = os.path.expanduser("~/.axe/klaus_training/imi_finetune.jsonl")
IMI_MEMORY_DIR = os.path.expanduser("~/.axe/memory/imi_conversations")
TEAM_CHANNEL_PATH = os.path.expanduser("~/Desktop/M1transfer/axe-memory/team/channel.jsonl")
ACTIVE_CONTEXT_PATH = os.path.expanduser("~/.axe/memory/active_context.json")

# Audit
AUDIT_LOG_PATH = os.path.expanduser("~/.axe/klaus_audit/audit.jsonl")

# Tool Calling (Qwen3 native)
TOOL_CALLING_ENABLED = True
MAX_TOOL_ITERATIONS = 5
TOOL_TIMEOUT_SECONDS = 30
ALLOWED_READ_PATHS = [
    os.path.expanduser("~/.axe/"),
    os.path.expanduser("~/klausimi-backend/"),
    os.path.expanduser("~/Desktop/M1transfer/"),
]
ALLOWED_WRITE_PATHS = [
    os.path.expanduser("~/.axe/"),
    os.path.expanduser("~/Desktop/M1transfer/"),
]
CLOUD_MEMORY_DIR = os.path.expanduser("~/.axe/cloud-memory/")
PYTHON_SANDBOX_ALLOWED_MODULES = [
    "math", "statistics", "json", "csv", "datetime", "collections",
    "itertools", "re", "functools", "operator", "decimal", "fractions",
    "random", "string", "textwrap", "hashlib", "base64", "urllib.parse",
]

# Ensure directories exist
os.makedirs(IMI_MEMORY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)

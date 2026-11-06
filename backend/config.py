"""Model access configuration powered by NVIDIA NIM (NVIDIA API).

All roles (chat, vision, embedding) connect to NVIDIA's OpenAI-compatible /v1 endpoints.
Provider settings are controlled via environment variables (.env).
"""
import json
import logging
import logging.handlers
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from backend/ or project root
_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent
load_dotenv(_BACKEND_DIR / ".env")
load_dotenv(_ROOT_DIR / ".env")
load_dotenv()

log = logging.getLogger("ada.config")

# NVIDIA NIM API Defaults
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
REQUEST_MAX_RETRIES = int(os.getenv("REQUEST_MAX_RETRIES", "2"))

# Agent Loop Limits (Adaptive Execution)
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "25"))
AGENT_WALL_CLOCK_SECONDS = int(os.getenv("AGENT_WALL_CLOCK_SECONDS", "600"))
AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "35"))
AGENT_MAX_REPAIRS = int(os.getenv("AGENT_MAX_REPAIRS", "3"))

# PDF & Document Ingestion & Chunking
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "150000000"))  # 150MB default
PDF_PAGE_BATCH_SIZE = int(os.getenv("PDF_PAGE_BATCH_SIZE", "25"))   # Incremental batch
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))                   # Target chunk character count
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))              # Chunk character overlap
ENABLE_PDF_OCR = os.getenv("ENABLE_PDF_OCR", "0").lower() in ("1", "true", "yes")
INDEXING_TIMEOUT_SECONDS = int(os.getenv("INDEXING_TIMEOUT_SECONDS", "600"))

DATA_DIR = Path(os.getenv("ADA_DATA_DIR") or (_BACKEND_DIR / "ada_data"))
OUTPUT_DIR = DATA_DIR / "output"
RUNS_DIR = DATA_DIR / "runs"


def setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        DATA_DIR / "ada.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("ada")
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
        root.addHandler(logging.StreamHandler())


def _client(base_url_env: str, key_env: str) -> OpenAI:
    base_url = os.getenv(base_url_env) or os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL)
    api_key = os.getenv(key_env) or os.getenv("NVIDIA_API_KEY", '')
    return OpenAI(
        base_url=base_url,
        api_key=api_key or "no-key",
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_MAX_RETRIES,
    )


def chat_client() -> OpenAI:
    return _client("CHAT_BASE_URL", "CHAT_API_KEY")


def vision_client() -> OpenAI:
    return _client("VISION_BASE_URL", "VISION_API_KEY")


def embed_client() -> OpenAI:
    return _client("EMBED_BASE_URL", "EMBED_API_KEY")


# NVIDIA NIM Model Defaults:
# - Chat: nvidia/nemotron-3.5-lightning-30b-a3b (or meta/llama-3.1-70b-instruct)
# - Vision: meta/llama-3.2-11b-vision-instruct (multimodal OCR and diagram inspection)
# - Embedding: nvidia/nemotron-3-embed-1b (high-precision retrieval embeddings)
CHAT_MODEL = os.getenv("CHAT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
VISION_MODEL = os.getenv("VISION_MODEL", "meta/llama-3.2-11b-vision-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")


def tuned_chat_model() -> str:
    """Returns the active chat model configured for Ada."""
    return CHAT_MODEL


def keep_warm(model: str, minutes: int = 60) -> None:
    """No-op for cloud-hosted NVIDIA NIM models."""
    pass


# Batch size for embedding requests
EMBED_BATCH = int(os.getenv("EMBED_BATCH_SIZE") or os.getenv("EMBED_BATCH", "32"))
EMBED_BATCH_SIZE = EMBED_BATCH


def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Embeds texts in batches using the NVIDIA NIM embeddings endpoint."""
    if not texts:
        return []
    client = embed_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
                extra_body={"input_type": input_type},
            )
        except Exception:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
        out.extend(d.embedding for d in resp.data)
    return out

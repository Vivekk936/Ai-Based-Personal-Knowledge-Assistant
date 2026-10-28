"""Model access config: three independently-swappable roles (chat, vision, embedding).

Every role talks OpenAI-compatible /v1 — Ollama, OpenRouter, NVIDIA NIM, and later
a self-hosted vLLM/TGI server all speak this. Switching providers is env vars, not code.
"""
import json
import logging
import logging.handlers
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

log = logging.getLogger("ada.config")

# 127.0.0.1, never "localhost". On Windows, "localhost" resolves to ::1 first, Ollama
# binds IPv4 only, and the refused IPv6 connect costs ~2s of SYN retry before the
# fallback — measured 2030ms vs 0.8ms per call on this machine. That tax was paid by
# every embed, chat and vision call in the app, which is most of why indexing and
# chat felt broken rather than merely slow.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
_OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}/v1"
_OLLAMA_NATIVE = f"http://{OLLAMA_HOST}"

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
REQUEST_MAX_RETRIES = int(os.getenv("REQUEST_MAX_RETRIES", "2"))

# Anchored to this file, not the working directory. Electron spawns the backend with
# cwd=backend/ while a bare `uvicorn --app-dir backend` from the project root does not,
# and a cwd-relative path quietly gives those two runs separate databases — the index
# looks empty depending on how the app was started.
DATA_DIR = Path(os.getenv("ADA_DATA_DIR") or (Path(__file__).resolve().parent / "ada_data"))
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
    return OpenAI(
        base_url=os.getenv(base_url_env, _OLLAMA_BASE_URL),
        api_key=os.getenv(key_env, "ollama"),  # Ollama ignores the key; hosted providers require it
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_MAX_RETRIES,
    )


def chat_client() -> OpenAI:
    return _client("CHAT_BASE_URL", "CHAT_API_KEY")


def vision_client() -> OpenAI:
    return _client("VISION_BASE_URL", "VISION_API_KEY")


def embed_client() -> OpenAI:
    return _client("EMBED_BASE_URL", "EMBED_API_KEY")


# hermes3 (8B) over qwen3.5:9b: measured 3/3 vs 3/3 on tool-call correctness but 2x the
# tokens/sec, because the 9B's weights don't fit this 6GB card. A model that can't call
# tools is useless as an agent however clever it is, so tool-calling is the first filter
# and speed is the tiebreak — phi4-mini is 3.5x faster still and scored 1/3, so it's out.
CHAT_MODEL = os.getenv("CHAT_MODEL", "hermes3:latest")
VISION_MODEL = os.getenv("VISION_MODEL", "moondream")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Ollama sizes the KV cache from the model's advertised max context (262144 for qwen3.5,
# 131072 for hermes3), which on a 6GB card pushes most layers onto the CPU. Capping it
# moved hermes3 from 29% to 71% GPU-resident. The OpenAI-compatible endpoint silently
# drops an `options` block, so the only way to set it without abandoning the
# provider-agnostic client is to bake it into a derived model once (see tuned_chat_model).
CHAT_NUM_CTX = int(os.getenv("CHAT_NUM_CTX", "8192"))
TUNED_MODEL_SUFFIX = "-ada"


def _ollama_post(path: str, payload: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        f"{_OLLAMA_NATIVE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    return json.loads(body.splitlines()[-1]) if body.strip() else {}


def _is_ollama() -> bool:
    """Only the local Ollama daemon gets the tuning treatment — a hosted provider has
    neither the concept nor the need."""
    if os.getenv("CHAT_BASE_URL", _OLLAMA_BASE_URL) != _OLLAMA_BASE_URL:
        return False
    try:
        urllib.request.urlopen(f"{_OLLAMA_NATIVE}/api/version", timeout=5).read()
        return True
    except Exception:
        return False


def tuned_chat_model() -> str:
    """Returns a model name whose context is capped to CHAT_NUM_CTX, creating it if
    needed. Idempotent and cheap: Ollama's create is a metadata-only layer over the
    existing weights, so this costs no extra disk and no re-download. Falls back to the
    untuned model on any failure — a slower agent beats a dead one."""
    if not _is_ollama():
        return CHAT_MODEL
    tuned = f"{CHAT_MODEL.split(':')[0]}{TUNED_MODEL_SUFFIX}"
    try:
        _ollama_post(
            "/api/create",
            {
                "model": tuned,
                "from": CHAT_MODEL,
                "parameters": {"num_ctx": CHAT_NUM_CTX, "temperature": 0.1},
                "stream": False,
            },
        )
        return tuned
    except Exception as e:
        log.warning("could not create tuned model %s (%s); using %s", tuned, e, CHAT_MODEL)
        return CHAT_MODEL


def keep_warm(model: str, minutes: int = 60) -> None:
    """Pin a model in VRAM. Ollama unloads after 5 minutes by default, and a cold load
    is ~15s here — long enough that an assistant used intermittently feels broken every
    single time. keep_alive isn't expressible on the OpenAI endpoint, hence the native call."""
    if not _is_ollama():
        return
    try:
        _ollama_post("/api/generate", {"model": model, "keep_alive": f"{minutes}m"}, timeout=300)
    except Exception as e:
        log.warning("warmup failed for %s: %s", model, e)


# One embeddings request costs ~14ms of compute plus a fixed per-request overhead, so the
# batch size is almost pure win up to the point where a single request gets slow enough to
# stall the progress UI. 64 keeps a batch under ~1s on this hardware.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds in batches rather than one request per caller. The old one-call-per-file
    pattern meant a 3-chunk file and a 300-chunk file cost nearly the same, which is the
    wrong shape entirely — cost should track content, not file count."""
    if not texts:
        return []
    client = embed_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        out.extend(d.embedding for d in resp.data)
    return out

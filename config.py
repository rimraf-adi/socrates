import os
import itertools
from langchain_openai import ChatOpenAI

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
MAX_TOKENS = 8192
MAX_INPUT_CHARS = 28000

# ─── Groq Configuration ──────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",   # 30K TPM, 1K RPD
    "llama-3.1-8b-instant",                          # 6K TPM, 14.4K RPD (fastest)
    "qwen/qwen3-32b",                                # 6K TPM, 60 RPM
    "llama-3.3-70b-versatile",                        # 12K TPM, 1K RPD (strongest)
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # 6K TPM, 1K RPD
    "moonshotai/kimi-k2-instruct",                    # 10K TPM, 60 RPM
]

# Round-robin iterator over Groq models to avoid rate limits
_groq_model_cycle = itertools.cycle(GROQ_MODELS)

# ─── Backend Switch ──────────────────────────────────────────────────────────
# "groq" or "lmstudio"
_current_backend = "groq"


def set_backend(backend: str):
    """Switch between 'groq' and 'lmstudio' backends at runtime."""
    global _current_backend
    backend = backend.lower().strip()
    if backend not in ("groq", "lmstudio"):
        raise ValueError(f"Unknown backend '{backend}'. Use 'groq' or 'lmstudio'.")
    _current_backend = backend


def get_backend() -> str:
    return _current_backend


def truncate(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated to fit token budget]"


def get_llm(temperature: float = 0.4, model: str | None = None):
    """
    Return a chat LLM for the active backend.
    Groq: rotates through GROQ_MODELS to spread rate limits.
    LMStudio: uses local server (model auto-detected by server if not specified).
    """
    if _current_backend == "groq":
        from langchain_groq import ChatGroq

        chosen_model = model or next(_groq_model_cycle)
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=chosen_model,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
    else:
        # LMStudio path — unchanged from original
        kwargs = {
            "base_url": LMSTUDIO_BASE_URL,
            "api_key": "lm-studio",
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
        }
        if model:
            kwargs["model"] = model
        return ChatOpenAI(**kwargs)

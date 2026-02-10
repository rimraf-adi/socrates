import os
from langchain_openai import ChatOpenAI

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")


def get_llm(temperature: float = 0.4, model: str | None = None) -> ChatOpenAI:
    kwargs = {
        "base_url": LMSTUDIO_BASE_URL,
        "api_key": "lm-studio",
        "temperature": temperature,
    }
    if model:
        kwargs["model"] = model
    return ChatOpenAI(**kwargs)

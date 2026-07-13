"""Environment configuration helpers for LLM clients."""

import os

from banso.llm.openai_sdk_client import OpenAISDKLLMClient

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_API_KEY = "unused"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0


def build_external_llm_client_from_env() -> OpenAISDKLLMClient:
    """Build an external LLM client from EXTERNAL_LLM_* env vars."""

    model = os.getenv("EXTERNAL_LLM_MODEL")
    if not model:
        raise RuntimeError("EXTERNAL_LLM_MODEL is required in .env")

    return OpenAISDKLLMClient(
        base_url=os.getenv("EXTERNAL_LLM_BASE_URL"),
        api_key=os.getenv("EXTERNAL_LLM_API_KEY") or "dummy",
        model=model,
        timeout=_float_env("EXTERNAL_LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS),
    )


def build_vllm_llm_client_from_env() -> OpenAISDKLLMClient:
    """Build a local vLLM OpenAI-compatible client from VLLM_* env vars."""

    model = os.getenv("VLLM_MODEL")
    if not model:
        raise RuntimeError("VLLM_MODEL is required in .env")

    return OpenAISDKLLMClient(
        base_url=os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL),
        api_key=os.getenv("VLLM_API_KEY", DEFAULT_VLLM_API_KEY),
        model=model,
        timeout=_float_env("VLLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS),
    )


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))

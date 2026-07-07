"""Tests for LLM environment configuration helpers."""

import pytest

from banso.llm import config


class CapturingClient:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)


@pytest.fixture(autouse=True)
def reset_client(monkeypatch):
    CapturingClient.calls = []
    monkeypatch.setattr(config, "OpenAISDKLLMClient", CapturingClient)


def test_vllm_llm_config_uses_vllm_defaults(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("VLLM_MODEL", "local-model")

    client = config.build_vllm_llm_client_from_env()

    assert isinstance(client, CapturingClient)
    assert client.kwargs == {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "unused",
        "model": "local-model",
        "timeout": 60.0,
    }


def test_external_llm_config_reads_external_vars(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_LLM_BASE_URL", "https://external.example.test/v1")
    monkeypatch.setenv("EXTERNAL_LLM_API_KEY", "external-key")
    monkeypatch.setenv("EXTERNAL_LLM_MODEL", "external-model")
    monkeypatch.setenv("EXTERNAL_LLM_TIMEOUT_SECONDS", "45")

    client = config.build_external_llm_client_from_env()

    assert client.kwargs == {
        "base_url": "https://external.example.test/v1",
        "api_key": "external-key",
        "model": "external-model",
        "timeout": 45.0,
    }


def test_vllm_llm_config_reads_vllm_vars(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key")
    monkeypatch.setenv("VLLM_MODEL", "local-model")
    monkeypatch.setenv("VLLM_TIMEOUT_SECONDS", "30")

    client = config.build_vllm_llm_client_from_env()

    assert client.kwargs == {
        "base_url": "http://127.0.0.1:8001/v1",
        "api_key": "vllm-key",
        "model": "local-model",
        "timeout": 30.0,
    }


@pytest.mark.parametrize(
    ("env_var", "builder"),
    [
        ("EXTERNAL_LLM_MODEL", config.build_external_llm_client_from_env),
        ("VLLM_MODEL", config.build_vllm_llm_client_from_env),
    ],
)
def test_llm_config_requires_model(monkeypatch, env_var, builder) -> None:
    monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(RuntimeError, match=env_var):
        builder()

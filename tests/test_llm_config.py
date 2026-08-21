"""Tests for LLM environment configuration helpers."""

import pytest

import banso.llm.config as config


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


@pytest.mark.parametrize(
    ("prefix", "builder", "values"),
    [
        (
            "EXTERNAL_LLM",
            config.build_external_llm_client_from_env,
            (
                "https://external.example.test/v1",
                "external-key",
                "external-model",
                "45",
            ),
        ),
        (
            "VLLM",
            config.build_vllm_llm_client_from_env,
            ("http://127.0.0.1:8001/v1", "vllm-key", "local-model", "30"),
        ),
    ],
    ids=["external", "vllm"],
)
def test_llm_config_reads_explicit_environment(
    monkeypatch,
    prefix,
    builder,
    values,
) -> None:
    base_url, api_key, model, timeout = values
    monkeypatch.setenv(f"{prefix}_BASE_URL", base_url)
    monkeypatch.setenv(f"{prefix}_API_KEY", api_key)
    monkeypatch.setenv(f"{prefix}_MODEL", model)
    monkeypatch.setenv(f"{prefix}_TIMEOUT_SECONDS", timeout)

    client = builder()

    assert client.kwargs == {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "timeout": float(timeout),
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

"""Tests for local corpus environment configuration."""

import pytest

import banso.corpus.config as config


class _FakeProvider:
    def __init__(self, **options: object) -> None:
        self.options = options


def test_embedding_provider_defaults_to_openai(monkeypatch) -> None:
    monkeypatch.setattr(config, "OpenAIEmbeddingProvider", _FakeProvider)
    monkeypatch.delenv("BANSO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("BANSO_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("BANSO_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("BANSO_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("BANSO_EMBEDDING_DIMENSIONS", "512")

    provider = config.build_embedding_provider_from_env()

    assert isinstance(provider, _FakeProvider)
    assert provider.options == {
        "model": "embedding-model",
        "dimensions": 512,
        "base_url": None,
        "api_key": None,
    }


def test_embedding_provider_builds_jina(monkeypatch) -> None:
    monkeypatch.setattr(config, "JinaEmbeddingProvider", _FakeProvider)
    monkeypatch.setenv("BANSO_EMBEDDING_PROVIDER", "jina")
    monkeypatch.setenv("BANSO_EMBEDDING_MODEL", "jina-model")
    monkeypatch.setenv("BANSO_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("BANSO_EMBEDDING_BASE_URL", "https://api.jina.ai/v1")
    monkeypatch.setenv("BANSO_EMBEDDING_API_KEY", "secret")

    provider = config.build_embedding_provider_from_env()

    assert isinstance(provider, _FakeProvider)
    assert provider.options == {
        "model": "jina-model",
        "dimensions": 1024,
        "base_url": "https://api.jina.ai/v1",
        "api_key": "secret",
    }


def test_embedding_provider_rejects_unknown_provider(monkeypatch) -> None:
    monkeypatch.setenv("BANSO_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("BANSO_EMBEDDING_DIMENSIONS", "512")
    monkeypatch.setenv("BANSO_EMBEDDING_PROVIDER", "unsupported")

    with pytest.raises(RuntimeError, match="must be 'openai' or 'jina'"):
        config.build_embedding_provider_from_env()

"""Tests for corpus embedding providers."""

from types import SimpleNamespace

import pytest

from banso.corpus import OpenAIEmbeddingProvider


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        texts = kwargs["input"]
        assert isinstance(texts, list)
        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1), 1.0])
            for index, _text in enumerate(texts)
        ]
        return SimpleNamespace(data=list(reversed(data)))


def test_openai_embedding_provider_batches_and_restores_response_order() -> None:
    embeddings = _FakeEmbeddings()
    provider = OpenAIEmbeddingProvider(
        model="embedding-model",
        dimensions=2,
        batch_size=2,
        client=SimpleNamespace(embeddings=embeddings),
    )

    assert provider.embed_documents(("one", "two", "three")) == (
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 1.0),
    )
    assert provider.embed_query("query") == (1.0, 1.0)
    assert embeddings.calls == [
        {"model": "embedding-model", "input": ["one", "two"]},
        {"model": "embedding-model", "input": ["three"]},
        {"model": "embedding-model", "input": ["query"]},
    ]


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        ([1.0], "1 dimensions"),
        ([float("nan"), 1.0], "non-finite"),
        ([0.0, 0.0], "all zeros"),
    ],
)
def test_openai_embedding_provider_rejects_invalid_vectors(
    embedding: list[float],
    message: str,
) -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=embedding)]
    )
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_kwargs: response)
    )
    provider = OpenAIEmbeddingProvider(
        model="embedding-model",
        dimensions=2,
        client=client,
    )

    with pytest.raises(ValueError, match=message):
        provider.embed_query("query")

"""Embedding providers used by the local corpus index."""

from math import isfinite
from typing import Any, Literal, Protocol, Sequence

from openai import OpenAI


class EmbeddingProvider(Protocol):
    """Generate document and query vectors for one embedding model."""

    model: str
    dimensions: int

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Embed document chunks in input order."""
        ...

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed one search query."""
        ...


class OpenAIEmbeddingProvider:
    """OpenAI-compatible synchronous embedding provider."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        model = model.strip()
        if not model:
            raise ValueError("embedding model must not be blank")
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be greater than zero")
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be greater than zero")

        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._client = client or OpenAI(
            base_url=base_url,
            api_key=api_key or "dummy",
            timeout=timeout,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Embed document texts in bounded API batches."""

        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                self._embed_batch(
                    texts[start : start + self.batch_size],
                    role="document",
                )
            )
        return tuple(vectors)

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed one non-blank query."""

        if not text.strip():
            raise ValueError("embedding query must not be blank")
        return self._embed_batch((text,), role="query")[0]

    def _embed_batch(
        self,
        texts: Sequence[str],
        *,
        role: Literal["document", "query"],
    ) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        if any(not text.strip() for text in texts):
            raise ValueError("embedding input must not be blank")

        response = self._client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dimensions,
            **self._request_options(role=role),
        )
        items = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(texts))):
            raise ValueError("embedding response indices do not match the input")
        return tuple(
            _validate_vector(item.embedding, dimensions=self.dimensions)
            for item in items
        )

    def _request_options(
        self,
        *,
        role: Literal["document", "query"],
    ) -> dict[str, object]:
        """Return provider-specific embedding request options."""

        return {}


class JinaEmbeddingProvider(OpenAIEmbeddingProvider):
    """Jina embedding provider using its OpenAI-compatible transport."""

    def _request_options(
        self,
        *,
        role: Literal["document", "query"],
    ) -> dict[str, object]:
        task = "retrieval.query" if role == "query" else "retrieval.passage"
        return {"extra_body": {"task": task}}


def _validate_vector(
    vector: Sequence[float],
    *,
    dimensions: int,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if len(values) != dimensions:
        raise ValueError(
            f"embedding vector has {len(values)} dimensions, expected {dimensions}"
        )
    if not all(isfinite(value) for value in values):
        raise ValueError("embedding vector contains a non-finite value")
    if not any(value != 0 for value in values):
        raise ValueError("embedding vector must not be all zeros")
    return values

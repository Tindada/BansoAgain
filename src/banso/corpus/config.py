"""Environment configuration for the local corpus."""

import os

from banso.corpus.indexing.embeddings import OpenAIEmbeddingProvider


def build_embedding_provider_from_env() -> OpenAIEmbeddingProvider:
    """Build the configured OpenAI-compatible embedding provider."""

    model = os.getenv("BANSO_EMBEDDING_MODEL")
    dimensions = os.getenv("BANSO_EMBEDDING_DIMENSIONS")
    if model is None or dimensions is None:
        raise RuntimeError(
            "BANSO_EMBEDDING_MODEL and BANSO_EMBEDDING_DIMENSIONS are required"
        )
    return OpenAIEmbeddingProvider(
        model=model,
        dimensions=int(dimensions),
        base_url=os.getenv("BANSO_EMBEDDING_BASE_URL"),
        api_key=os.getenv("BANSO_EMBEDDING_API_KEY"),
    )

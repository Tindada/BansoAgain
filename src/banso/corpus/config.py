"""Environment configuration for the local corpus."""

import os

from banso.corpus.indexing.embeddings import (
    EmbeddingProvider,
    JinaEmbeddingProvider,
    OpenAIEmbeddingProvider,
)

def build_embedding_provider_from_env() -> EmbeddingProvider:
    """Build the configured embedding provider."""

    model = os.getenv("BANSO_EMBEDDING_MODEL")
    dimensions = os.getenv("BANSO_EMBEDDING_DIMENSIONS")
    if model is None or dimensions is None:
        raise RuntimeError(
            "BANSO_EMBEDDING_MODEL and BANSO_EMBEDDING_DIMENSIONS are required"
        )
    provider_name = os.getenv("BANSO_EMBEDDING_PROVIDER", "openai").strip().casefold()
    provider_types = {
        "openai": OpenAIEmbeddingProvider,
        "jina": JinaEmbeddingProvider,
    }
    if provider_name not in provider_types:
        raise RuntimeError(
            "BANSO_EMBEDDING_PROVIDER must be 'openai' or 'jina', "
            f"got {provider_name!r}"
        )

    return provider_types[provider_name](
        model=model,
        dimensions=int(dimensions),
        base_url=os.getenv("BANSO_EMBEDDING_BASE_URL"),
        api_key=os.getenv("BANSO_EMBEDDING_API_KEY"),
    )

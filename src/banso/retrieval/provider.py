"""Retrieval provider interface."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.retrieval.models import SearchResult


class SearchRequest(BaseModel):
    """Structured search request passed to retrieval providers."""

    query: str
    max_results: int = 10
    language: str | None = None
    region: str | None = None
    time_range: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalProvider(Protocol):
    """Search provider abstraction."""

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Return search results for a request."""
        ...

"""Fake retrieval provider for local smoke tests."""

from banso.retrieval.models import SearchResult
from banso.retrieval.provider import SearchRequest
from banso.source import Source, SourceType


class FakeRetrievalProvider:
    """Returns deterministic search results without external services."""

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        source = Source(
            name="Fake News Source",
            url="https://example.com",
            type=SourceType.NEWS,
        )
        return [
            SearchResult(
                title=f"Fake result for {request.query}",
                url="https://example.com/news/fake-result",
                snippet=f"A fake search result about {request.query}.",
                source=source,
                rank=1,
            )
        ][: request.max_results]

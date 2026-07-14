"""Tavily-backed retrieval provider."""

from typing import Any

import httpx

from banso.retrieval.models import SearchResult, Source, SourceType
from banso.retrieval.url_utils import publisher_domain, publisher_home_url
from banso.retrieval.provider import RetrievalProvider, SearchRequest


class TavilyRetrievalProvider(RetrievalProvider):
    """Retrieval provider backed by Tavily Search API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.tavily.com",
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        topic: str = "general",
        search_depth: str = "basic",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout
        self.topic = topic
        self.search_depth = search_depth

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Return search results for a request."""

        payload = self._build_payload(request)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self._client is not None:
            response = await self._client.post(
                f"{self.base_url}/search",
                json=payload,
                headers=headers,
            )
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers=headers,
                )

        response.raise_for_status()
        data = response.json()
        return self._parse_results(data)

    def _build_payload(self, request: SearchRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": request.query,
            "max_results": request.max_results,
            "topic": self.topic,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_usage": True,
        }

        if request.time_range is not None:
            payload["time_range"] = request.time_range

        if request.region is not None and self.topic == "general":
            payload["country"] = request.region

        return payload

    def _parse_results(self, data: dict[str, Any]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for index, item in enumerate(data.get("results", []), start=1):
            if not isinstance(item, dict):
                continue

            title = item.get("title")
            url = item.get("url")
            if not isinstance(title, str) or not isinstance(url, str):
                continue

            content = item.get("content")
            score = item.get("score")
            favicon = item.get("favicon")

            metadata = {
                "provider": "tavily",
                "score": score,
                "favicon": favicon,
            }

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=content if isinstance(content, str) else None,
                    source=Source(
                        name=publisher_domain(url) or "Unknown publisher",
                        url=publisher_home_url(url),
                        type=SourceType.UNKNOWN,
                    ),
                    rank=index,
                    metadata=metadata,
                )
            )

        return results

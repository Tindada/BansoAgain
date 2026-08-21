"""Tavily-backed retrieval provider."""

from typing import Any

import httpx

from banso.http_errors import classify_httpx_error
from banso.retrieval.models import SearchResult
from banso.retrieval.url_utils import publisher_domain, publisher_home_url
from banso.retrieval.provider import (
    RetrievalError,
    RetrievalProvider,
    SearchRequest,
)
from banso.source import Source, SourceType


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

        try:
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
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            failure = classify_httpx_error(error)
            if isinstance(error, httpx.HTTPStatusError):
                reason = "http_status"
                message = f"Tavily returned HTTP {error.response.status_code}"
            else:
                reason = "transport"
                message = str(error) or "Tavily search transport failed"
            raise RetrievalError(
                provider="tavily",
                reason=reason,
                status_code=failure.status_code,
                message=message,
                source_error_type=failure.source_error_type,
            ) from error

        try:
            data = response.json()
        except ValueError as error:
            raise RetrievalError(
                provider="tavily",
                reason="invalid_response",
                message="Tavily returned a non-JSON response",
                source_error_type=type(error).__name__,
            ) from error
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RetrievalError(
                provider="tavily",
                reason="invalid_response",
                message="Tavily response is missing a results array",
                source_error_type="ResponseValidationError",
            )
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

        if request.source_domains is not None:
            payload["include_domains"] = request.source_domains

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

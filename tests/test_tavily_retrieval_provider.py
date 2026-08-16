"""Tests for the Tavily retrieval provider."""

import asyncio
import json

import httpx
import pytest

from banso.retrieval import RetrievalError, SearchRequest, TavilyRetrievalProvider


async def _capture_search_error(handler, *, api_key: str = "key") -> RetrievalError:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = TavilyRetrievalProvider(api_key=api_key, client=client)
        with pytest.raises(RetrievalError) as caught:
            await provider.search(SearchRequest(query="query"))
        return caught.value
    finally:
        await client.aclose()


async def _run_tavily_provider_maps_request_and_response() -> None:
    calls: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            json={
                "query": "latest AI news",
                "results": [
                    {
                        "title": "AI News One",
                        "url": "https://example.com/ai-news-one",
                        "content": "First AI news snippet.",
                        "score": 0.91,
                        "favicon": "https://example.com/favicon.ico",
                    },
                    {
                        "title": "AI News Two",
                        "url": "https://example.com/ai-news-two",
                        "content": "Second AI news snippet.",
                        "score": 0.82,
                    },
                ],
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TavilyRetrievalProvider(
        api_key="tvly-test-key",
        base_url="https://api.tavily.test",
        client=client,
    )

    try:
        results = await provider.search(
            SearchRequest(
                query="latest AI news",
                max_results=2,
                time_range="day",
            )
        )
    finally:
        await client.aclose()

    assert len(calls) == 1
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "https://api.tavily.test/search"
    assert calls[0]["headers"]["authorization"] == "Bearer tvly-test-key"
    assert calls[0]["headers"]["content-type"] == "application/json"
    assert calls[0]["json"] == {
        "query": "latest AI news",
        "max_results": 2,
        "topic": "general",
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "include_usage": True,
        "time_range": "day",
    }
    assert len(results) == 2
    assert results[0].title == "AI News One"
    assert results[0].url == "https://example.com/ai-news-one"
    assert results[0].snippet == "First AI news snippet."
    assert results[0].source is not None
    assert results[0].source.name == "example.com"
    assert results[0].source.url == "https://example.com"
    assert results[0].source.type.value == "unknown"
    assert results[0].rank == 1
    assert results[0].metadata["provider"] == "tavily"
    assert results[0].metadata["score"] == 0.91
    assert results[0].metadata["favicon"] == "https://example.com/favicon.ico"
    assert results[1].rank == 2


async def _run_tavily_provider_skips_invalid_result_items() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Missing URL"},
                    {"url": "https://example.com/missing-title"},
                    "not an object",
                    {
                        "title": "Valid",
                        "url": "https://example.com/valid",
                    },
                ]
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TavilyRetrievalProvider(api_key="tvly-test-key", client=client)

    try:
        results = await provider.search(SearchRequest(query="latest AI news"))
    finally:
        await client.aclose()

    assert len(results) == 1
    assert results[0].title == "Valid"
    assert results[0].url == "https://example.com/valid"
    assert results[0].rank == 4


def test_tavily_provider_maps_request_and_response() -> None:
    asyncio.run(_run_tavily_provider_maps_request_and_response())


def test_tavily_provider_skips_invalid_result_items() -> None:
    asyncio.run(_run_tavily_provider_skips_invalid_result_items())


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [
        (400, False),
        (429, True),
        (500, True),
    ],
)
def test_tavily_provider_maps_http_failures(
    status_code: int,
    retryable: bool,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            text='{"detail":"failed tvly-test-key"}',
            request=request,
        )

    error = asyncio.run(_capture_search_error(handler, api_key="tvly-test-key"))

    assert error.reason == "http_status"
    assert (error.status_code, error.retryable) == (status_code, retryable)
    assert error.message == f"Tavily returned HTTP {status_code}"


def test_tavily_provider_maps_transport_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    error = asyncio.run(_capture_search_error(handler))

    assert error.reason == "transport"
    assert error.retryable is True


def test_tavily_provider_rejects_invalid_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json", request=request)

    error = asyncio.run(_capture_search_error(handler))

    assert error.reason == "invalid_response"
    assert error.retryable is False

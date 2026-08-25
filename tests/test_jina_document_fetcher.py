"""Tests for the Jina Reader-backed document fetcher."""

import asyncio

import httpx
import pytest

from banso.documents.fetcher import DocumentFetchError, DocumentFetchRequest
from banso.documents.jina_fetcher import JinaDocumentFetcher
from banso.source import Source, SourceType


def _response_data(**overrides):
    data = {
        "title": "Jina title",
        "url": "https://example.com/final",
        "content": "# Heading\n\nArticle body.",
        "publishedTime": "2026-08-24T12:30:00Z",
        "usage": {"tokens": 42},
    }
    data.update(overrides)
    return {"code": 200, "status": 20000, "data": data}


async def _fetch_with_handler(handler, **fetcher_options):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = JinaDocumentFetcher(client=client, **fetcher_options)
    try:
        return await fetcher.fetch(
            DocumentFetchRequest(
                url="https://example.com/article?edition=global",
                title="Search result title",
                source=Source(
                    name="Example",
                    url="https://example.com",
                    type=SourceType.NEWS,
                ),
                metadata={"search_result_id": "result-1"},
            )
        )
    finally:
        await client.aclose()


def test_jina_fetcher_maps_json_document_without_api_key() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "https://r.jina.ai/https://example.com/article?edition=global"
        )
        assert request.headers["accept"] == "application/json"
        assert request.headers["x-no-cache"] == "true"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=_response_data(),
            request=request,
        )

    document = asyncio.run(_fetch_with_handler(handler))

    assert document.url == "https://example.com/final"
    assert document.title == "Search result title"
    assert document.text == "# Heading\n\nArticle body."
    assert document.source is not None
    assert document.source.name == "Example"
    assert document.published_at is not None
    assert document.published_at.isoformat() == "2026-08-24T12:30:00+00:00"
    assert document.metadata == {
        "search_result_id": "result-1",
        "fetcher": "jina_reader",
        "status_code": 200,
        "content_type": "application/json",
        "final_url": "https://example.com/final",
        "extraction_strategy": "jina_reader",
        "jina_code": 200,
        "jina_status": 20000,
        "jina_usage_tokens": 42,
        "extracted_text_chars": len(document.text),
    }


def test_jina_fetcher_sends_optional_api_key_and_uses_jina_title() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json=_response_data(), request=request)

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        fetcher = JinaDocumentFetcher(client=client, api_key=" secret ")
        try:
            return await fetcher.fetch(
                DocumentFetchRequest(url="https://example.com/article")
            )
        finally:
            await client.aclose()

    document = asyncio.run(run())
    assert document.title == "Jina title"


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(403, False), (429, True), (500, True)],
)
def test_jina_fetcher_maps_http_status(status_code, retryable) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request)

    with pytest.raises(DocumentFetchError) as captured:
        asyncio.run(_fetch_with_handler(handler))

    error = captured.value
    assert error.url == "https://example.com/article?edition=global"
    assert error.reason == "http_status"
    assert error.status_code == status_code
    assert error.retryable is retryable
    assert error.source_error_type == "HTTPStatusError"


def test_jina_fetcher_maps_transport_failure() -> None:
    source_error = httpx.ConnectError("connection failed")

    async def handler(request: httpx.Request) -> httpx.Response:
        raise source_error

    with pytest.raises(DocumentFetchError) as captured:
        asyncio.run(_fetch_with_handler(handler))

    error = captured.value
    assert error.reason == "transport"
    assert error.status_code is None
    assert error.source_error_type == "ConnectError"
    assert error.__cause__ is source_error


def test_jina_fetcher_maps_embedded_target_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = _response_data(
            warning="Target URL returned error 403: Forbidden",
            content="Blocked page",
        )
        return httpx.Response(200, json=payload, request=request)

    with pytest.raises(DocumentFetchError) as captured:
        asyncio.run(_fetch_with_handler(handler))

    error = captured.value
    assert error.url == "https://example.com/final"
    assert error.reason == "http_status"
    assert error.status_code == 403
    assert error.source_error_type == "JinaTargetHTTPError"


@pytest.mark.parametrize(
    ("response_options", "expected_source_type"),
    [
        ({"content": b"{"}, "JSONDecodeError"),
        (
            {"json": {"code": 200, "status": 20000, "data": {}}},
            "JinaResponseValidationError",
        ),
    ],
    ids=["invalid-json", "invalid-schema"],
)
def test_jina_fetcher_rejects_invalid_response(
    response_options,
    expected_source_type,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, **response_options)

    with pytest.raises(DocumentFetchError) as captured:
        asyncio.run(_fetch_with_handler(handler))

    assert captured.value.reason == "parse_error"
    assert captured.value.source_error_type == expected_source_type


def test_jina_fetcher_rejects_empty_content() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response_data(content="  "),
            request=request,
        )

    with pytest.raises(DocumentFetchError) as captured:
        asyncio.run(_fetch_with_handler(handler))

    assert captured.value.reason == "no_extractable_text"


def test_jina_fetcher_ignores_invalid_published_time() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response_data(publishedTime="not-a-date"),
            request=request,
        )

    document = asyncio.run(_fetch_with_handler(handler))
    assert document.published_at is None


@pytest.mark.parametrize(
    "options",
    [
        {"base_url": " "},
        {"timeout": 0},
    ],
)
def test_jina_fetcher_validates_options(options) -> None:
    with pytest.raises(ValueError):
        JinaDocumentFetcher(**options)

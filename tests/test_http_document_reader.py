"""Tests for the HTTP-backed document reader."""

import asyncio

import httpx

from banso.documents import DocumentReadRequest, HTTPDocumentReader
from banso.retrieval import Source, SourceType


async def _run_http_document_reader_extracts_html_document() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/news"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="""
            <html>
              <head>
                <title>Page title</title>
                <style>.hidden { display: none; }</style>
                <script>console.log("ignore");</script>
              </head>
              <body>
                <article>
                  <h1>Article heading</h1>
                  <p>First paragraph.</p>
                  <p>Second paragraph.</p>
                </article>
              </body>
            </html>
            """,
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reader = HTTPDocumentReader(client=client)

    try:
        document = await reader.read(
            DocumentReadRequest(
                url="https://example.com/news",
                source=Source(
                    name="Example News",
                    url="https://example.com",
                    type=SourceType.NEWS,
                ),
            )
        )
    finally:
        await client.aclose()

    assert document.url == "https://example.com/news"
    assert document.title == "Page title"
    assert document.source is not None
    assert document.source.name == "Example News"
    assert document.text == "Article heading\nFirst paragraph.\nSecond paragraph."
    assert "console.log" not in document.text
    assert document.metadata["reader"] == "http"
    assert document.metadata["status_code"] == 200
    assert document.metadata["content_type"] == "text/html; charset=utf-8"
    assert document.metadata["final_url"] == "https://example.com/news"


async def _run_http_document_reader_prefers_request_title() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><head><title>HTML title</title></head><body>Body text</body></html>",
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reader = HTTPDocumentReader(client=client)

    try:
        document = await reader.read(
            DocumentReadRequest(
                url="https://example.com/news",
                title="Search result title",
                metadata={"search_result_id": "result-1"},
            )
        )
    finally:
        await client.aclose()

    assert document.title == "Search result title"
    assert document.text == "Body text"
    assert document.metadata["search_result_id"] == "result-1"


def test_http_document_reader_extracts_html_document() -> None:
    asyncio.run(_run_http_document_reader_extracts_html_document())


def test_http_document_reader_prefers_request_title() -> None:
    asyncio.run(_run_http_document_reader_prefers_request_title())

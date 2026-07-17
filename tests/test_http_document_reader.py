"""Tests for the HTTP-backed document reader."""

import asyncio

import httpx

from banso.documents import (
    DocumentReadError,
    DocumentReadRequest,
    HTTPDocumentReader,
)
from banso.documents.http_reader import _extract_html_content
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
    assert document.metadata["extraction_strategy"] == "article"
    assert document.metadata["raw_html_chars"] > len(document.text)
    assert document.metadata["extracted_text_chars"] == len(document.text)


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


async def _run_http_document_reader_exposes_http_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reader = HTTPDocumentReader(client=client)

    try:
        try:
            await reader.read(DocumentReadRequest(url="https://example.com/blocked"))
        except DocumentReadError as error:
            assert error.url == "https://example.com/blocked"
            assert error.reason == "http_status"
            assert error.status_code == 403
            assert error.source_error_type == "HTTPStatusError"
            assert isinstance(error.__cause__, httpx.HTTPStatusError)
        else:
            raise AssertionError("expected DocumentReadError")
    finally:
        await client.aclose()


def test_http_document_reader_exposes_http_status() -> None:
    asyncio.run(_run_http_document_reader_exposes_http_status())


async def _run_http_document_reader_exposes_transport_failure(
    source_error: httpx.TransportError,
    expected_reason: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise source_error

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reader = HTTPDocumentReader(client=client)

    try:
        try:
            await reader.read(DocumentReadRequest(url="https://example.com/news"))
        except DocumentReadError as error:
            assert error.url == "https://example.com/news"
            assert error.reason == expected_reason
            assert error.status_code is None
            assert error.source_error_type == type(source_error).__name__
            assert error.__cause__ is source_error
        else:
            raise AssertionError("expected DocumentReadError")
    finally:
        await client.aclose()


def test_http_document_reader_exposes_timeout() -> None:
    asyncio.run(
        _run_http_document_reader_exposes_transport_failure(
            httpx.ReadTimeout("read timed out"),
            "timeout",
        )
    )


def test_http_document_reader_exposes_other_transport_error() -> None:
    asyncio.run(
        _run_http_document_reader_exposes_transport_failure(
            httpx.ConnectError("connection failed"),
            "transport",
        )
    )


def test_html_extraction_prefers_longest_article_and_removes_noise() -> None:
    title, text, strategy = _extract_html_content(
        """
        <html>
          <head><title>Page title</title></head>
          <body>
            <nav>Site navigation</nav>
            <article><p>Short recommendation.</p></article>
            <article>
              <header><h1>Article headline</h1></header>
              <p>OpenAI released <a href="/model">a new model</a> today.</p>
              <p>Second article paragraph with more detail.</p>
              <aside>Related stories</aside>
            </article>
          </body>
        </html>
        """
    )

    assert title == "Page title"
    assert strategy == "article"
    assert text == (
        "Article headline\n"
        "OpenAI released a new model today.\n"
        "Second article paragraph with more detail."
    )
    assert "Site navigation" not in text
    assert "Related stories" not in text
    assert "Short recommendation" not in text


def test_html_extraction_uses_main_then_role_main() -> None:
    _, main_text, main_strategy = _extract_html_content(
        "<body><main><h1>Main heading</h1><p>Main text.</p></main></body>"
    )
    _, role_text, role_strategy = _extract_html_content(
        '<body><section role="main"><p>Role main text.</p></section></body>'
    )

    assert main_strategy == "main"
    assert main_text == "Main heading\nMain text."
    assert role_strategy == "role_main"
    assert role_text == "Role main text."


def test_html_extraction_prefers_main_over_article_cards() -> None:
    _, text, strategy = _extract_html_content(
        """
        <body>
          <main>
            <h1>Year in review</h1>
            <p>The main article contains the complete retrospective.</p>
            <article><p>A comparatively long recommended story card.</p></article>
            <article><p>Another recommendation.</p></article>
          </main>
        </body>
        """
    )

    assert strategy == "main"
    assert text.startswith(
        "Year in review\nThe main article contains the complete retrospective."
    )


def test_html_extraction_falls_back_to_body_and_removes_page_chrome() -> None:
    _, text, strategy = _extract_html_content(
        """
        <body>
          <header>Site header</header>
          <div><h1>Body heading</h1><p>Body text.</p></div>
          <footer>Site footer</footer>
        </body>
        """
    )

    assert strategy == "body"
    assert text == "Body heading\nBody text."

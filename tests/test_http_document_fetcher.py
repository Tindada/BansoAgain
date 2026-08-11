"""Tests for the HTTP-backed document fetcher."""

import asyncio
from io import BytesIO

import httpx
import pytest
from pypdf import PdfReader, PdfWriter

from banso.documents import (
    DocumentFetchError,
    DocumentFetchRequest,
    DocumentParseError,
    DocumentParser,
    HTTPDocumentFetcher,
)
from banso.retrieval import Source, SourceType
from tests.pdf_fixtures import make_text_pdf


def _encrypt_pdf(content: bytes) -> bytes:
    reader = PdfReader(BytesIO(content))
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


async def _run_http_document_fetcher_extracts_html_document() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/news"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="""
            <html>
              <head>
                <title>Page title</title>
                <meta property="article:published_time" content="2026-08-10T08:00:00Z">
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
    fetcher = HTTPDocumentFetcher(client=client)

    try:
        document = await fetcher.fetch(
            DocumentFetchRequest(
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
    assert document.published_at is not None
    assert document.published_at.isoformat() == "2026-08-10T08:00:00+00:00"
    assert document.text == "Article heading\nFirst paragraph.\nSecond paragraph."
    assert "console.log" not in document.text
    assert document.metadata["fetcher"] == "http"
    assert document.metadata["status_code"] == 200
    assert document.metadata["content_type"] == "text/html; charset=utf-8"
    assert document.metadata["final_url"] == "https://example.com/news"
    assert document.metadata["extraction_strategy"] == "article"
    assert document.metadata["raw_html_chars"] > len(document.text)
    assert document.metadata["extracted_text_chars"] == len(document.text)


async def _run_http_document_fetcher_prefers_request_title() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><head><title>HTML title</title></head><body>Body text</body></html>",
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = HTTPDocumentFetcher(client=client)

    try:
        document = await fetcher.fetch(
            DocumentFetchRequest(
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


def test_http_document_fetcher_extracts_html_document() -> None:
    asyncio.run(_run_http_document_fetcher_extracts_html_document())


def test_http_document_fetcher_prefers_request_title() -> None:
    asyncio.run(_run_http_document_fetcher_prefers_request_title())


async def _run_http_document_fetcher_exposes_http_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = HTTPDocumentFetcher(client=client)

    try:
        try:
            await fetcher.fetch(DocumentFetchRequest(url="https://example.com/blocked"))
        except DocumentFetchError as error:
            assert error.url == "https://example.com/blocked"
            assert error.reason == "http_status"
            assert error.status_code == 403
            assert error.source_error_type == "HTTPStatusError"
            assert isinstance(error.__cause__, httpx.HTTPStatusError)
        else:
            raise AssertionError("expected DocumentFetchError")
    finally:
        await client.aclose()


def test_http_document_fetcher_exposes_http_status() -> None:
    asyncio.run(_run_http_document_fetcher_exposes_http_status())


async def _run_http_document_fetcher_exposes_transport_failure(
    source_error: httpx.TransportError,
    expected_reason: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise source_error

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = HTTPDocumentFetcher(client=client)

    try:
        try:
            await fetcher.fetch(DocumentFetchRequest(url="https://example.com/news"))
        except DocumentFetchError as error:
            assert error.url == "https://example.com/news"
            assert error.reason == expected_reason
            assert error.status_code is None
            assert error.source_error_type == type(source_error).__name__
            assert error.__cause__ is source_error
        else:
            raise AssertionError("expected DocumentFetchError")
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("source_error", "expected_reason"),
    (
        (httpx.ReadTimeout("read timed out"), "timeout"),
        (httpx.ConnectError("connection failed"), "transport"),
    ),
    ids=["timeout", "transport"],
)
def test_http_document_fetcher_exposes_transport_failure(
    source_error: httpx.TransportError,
    expected_reason: str,
) -> None:
    case = _run_http_document_fetcher_exposes_transport_failure(
        source_error,
        expected_reason,
    )
    asyncio.run(case)


async def _run_http_document_fetcher_rejects_unsupported_content_type(
    content_type: str | None,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": content_type} if content_type is not None else {}
        return httpx.Response(
            200,
            headers=headers,
            content=b"%PDF-binary-content",
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = HTTPDocumentFetcher(client=client)

    try:
        try:
            await fetcher.fetch(DocumentFetchRequest(url="https://example.com/report"))
        except DocumentFetchError as error:
            assert error.url == "https://example.com/report"
            assert error.reason == "unsupported_content_type"
            assert error.status_code == 200
            assert error.source_error_type == "UnsupportedContentType"
            assert (content_type or "<missing>") in error.message
        else:
            raise AssertionError("expected DocumentFetchError")
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "content_type",
    ["image/png", None],
    ids=["unsupported", "missing"],
)
def test_http_document_fetcher_rejects_unsupported_content_type(
    content_type: str | None,
) -> None:
    asyncio.run(
        _run_http_document_fetcher_rejects_unsupported_content_type(content_type)
    )


async def _fetch_pdf(
    content: bytes,
    *,
    content_type: str = "Application/PDF; charset=binary",
    title: str | None = None,
    max_pdf_bytes: int | None = None,
    max_pdf_pages: int | None = None,
):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            content=content,
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parser_options: dict[str, int] = {}
    if max_pdf_bytes is not None:
        parser_options["max_pdf_bytes"] = max_pdf_bytes
    if max_pdf_pages is not None:
        parser_options["max_pdf_pages"] = max_pdf_pages
    fetcher = HTTPDocumentFetcher(
        client=client,
        parser=DocumentParser(**parser_options),
    )
    try:
        return await fetcher.fetch(
            DocumentFetchRequest(url="https://example.com/report", title=title)
        )
    finally:
        await client.aclose()


def test_http_document_fetcher_extracts_pdf_and_prefers_request_title() -> None:
    content = make_text_pdf("Report body.", title="PDF title")

    document = asyncio.run(_fetch_pdf(content, title="Search result title"))

    assert document.title == "Search result title"
    assert document.text == "Report body."
    assert document.metadata["content_type"] == "Application/PDF; charset=binary"
    assert document.metadata["extraction_strategy"] == "pypdf"
    assert document.metadata["raw_bytes"] == len(content)
    assert document.metadata["pdf_page_count"] == 1
    assert document.metadata["pdf_pages_with_text"] == 1
    assert document.metadata["extracted_text_chars"] == len(document.text)


def test_http_document_fetcher_uses_pdf_metadata_title() -> None:
    document = asyncio.run(
        _fetch_pdf(make_text_pdf("Report body.", title="PDF title"))
    )

    assert document.title == "PDF title"


def test_http_document_fetcher_reports_pdf_without_extractable_text() -> None:
    try:
        asyncio.run(_fetch_pdf(make_text_pdf("")))
    except DocumentFetchError as error:
        assert error.reason == "no_extractable_text"
        assert error.status_code == 200
        assert error.source_error_type == "NoExtractableText"
        assert isinstance(error.__cause__, DocumentParseError)
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_reports_malformed_pdf() -> None:
    try:
        asyncio.run(_fetch_pdf(b"%PDF-broken"))
    except DocumentFetchError as error:
        assert error.reason == "parse_error"
        assert error.status_code == 200
        assert isinstance(error.__cause__, DocumentParseError)
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_reports_password_protected_pdf() -> None:
    encrypted = _encrypt_pdf(make_text_pdf("Secret report."))

    try:
        asyncio.run(_fetch_pdf(encrypted))
    except DocumentFetchError as error:
        assert error.reason == "parse_error"
        assert error.status_code == 200
        assert isinstance(error.__cause__, DocumentParseError)
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_enforces_pdf_byte_limit() -> None:
    content = make_text_pdf("Report body.")

    try:
        asyncio.run(_fetch_pdf(content, max_pdf_bytes=len(content) - 1))
    except DocumentFetchError as error:
        assert error.reason == "document_too_large"
        assert error.source_error_type == "PDFByteLimitExceeded"
        assert isinstance(error.__cause__, DocumentParseError)
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_enforces_pdf_page_limit() -> None:
    content = make_text_pdf("First page.", "Second page.")

    try:
        asyncio.run(_fetch_pdf(content, max_pdf_pages=1))
    except DocumentFetchError as error:
        assert error.reason == "document_too_large"
        assert error.source_error_type == "PDFPageLimitExceeded"
        assert isinstance(error.__cause__, DocumentParseError)
    else:
        raise AssertionError("expected DocumentFetchError")

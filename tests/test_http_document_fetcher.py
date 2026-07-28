"""Tests for the HTTP-backed document fetcher."""

import asyncio
from io import BytesIO

import httpx
import pytest
from pypdf import PdfReader, PdfWriter

from banso.documents import (
    DocumentFetchError,
    DocumentFetchRequest,
    HTTPDocumentFetcher,
)
from banso.documents.http_fetcher import _extract_html_content, _extract_pdf_content
from banso.retrieval import Source, SourceType


def _pdf_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _make_text_pdf(*page_texts: str, title: str | None = None) -> bytes:
    page_count = len(page_texts)
    font_id = 3 + page_count
    content_start_id = font_id + 1
    info_id = content_start_id + page_count if title is not None else None
    page_ids = list(range(3, 3 + page_count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
            f"/Count {page_count} >>"
        ).encode(),
    ]

    for index, page_id in enumerate(page_ids):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_start_id + index} 0 R >>"
            ).encode()
        )

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page_text in page_texts:
        stream = (
            f"BT /F1 12 Tf 72 720 Td ({_pdf_string(page_text)}) Tj ET"
        ).encode()
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )

    if title is not None:
        objects.append(f"<< /Title ({_pdf_string(title)}) >>".encode())

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())

    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R"
    if info_id is not None:
        trailer += f" /Info {info_id} 0 R"
    trailer += f" >>\nstartxref\n{xref_offset}\n%%EOF\n"
    output.extend(trailer.encode())
    return bytes(output)


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


def test_html_extraction_prefers_longest_article_and_removes_noise() -> None:
    extraction = _extract_html_content(
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

    assert extraction.title == "Page title"
    assert extraction.strategy == "article"
    assert extraction.text == (
        "Article headline\n"
        "OpenAI released a new model today.\n"
        "Second article paragraph with more detail."
    )
    assert "Site navigation" not in extraction.text
    assert "Related stories" not in extraction.text
    assert "Short recommendation" not in extraction.text


def test_html_extraction_uses_main_then_role_main() -> None:
    main_extraction = _extract_html_content(
        "<body><main><h1>Main heading</h1><p>Main text.</p></main></body>"
    )
    role_extraction = _extract_html_content(
        '<body><section role="main"><p>Role main text.</p></section></body>'
    )

    assert main_extraction.strategy == "main"
    assert main_extraction.text == "Main heading\nMain text."
    assert role_extraction.strategy == "role_main"
    assert role_extraction.text == "Role main text."


def test_html_extraction_prefers_main_over_article_cards() -> None:
    extraction = _extract_html_content(
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

    assert extraction.strategy == "main"
    assert extraction.text.startswith(
        "Year in review\nThe main article contains the complete retrospective."
    )


def test_html_extraction_falls_back_to_body_and_removes_page_chrome() -> None:
    extraction = _extract_html_content(
        """
        <body>
          <header>Site header</header>
          <div><h1>Body heading</h1><p>Body text.</p></div>
          <footer>Site footer</footer>
        </body>
        """
    )

    assert extraction.strategy == "body"
    assert extraction.text == "Body heading\nBody text."


def test_pdf_extraction_combines_pages_and_extracts_title() -> None:
    content = _make_text_pdf("First page.", "Second page.", title="PDF title")

    extraction = _extract_pdf_content(content, max_pages=10)

    assert extraction.title == "PDF title"
    assert extraction.text == "First page.\n\nSecond page."
    assert extraction.strategy == "pypdf"
    assert extraction.metadata == {
        "raw_bytes": len(content),
        "pdf_page_count": 2,
        "pdf_pages_with_text": 2,
    }


async def _fetch_pdf(
    content: bytes,
    *,
    content_type: str = "Application/PDF; charset=binary",
    title: str | None = None,
    max_pdf_bytes: int = 20 * 1024 * 1024,
    max_pdf_pages: int = 200,
):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            content=content,
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = HTTPDocumentFetcher(
        client=client,
        max_pdf_bytes=max_pdf_bytes,
        max_pdf_pages=max_pdf_pages,
    )
    try:
        return await fetcher.fetch(
            DocumentFetchRequest(url="https://example.com/report", title=title)
        )
    finally:
        await client.aclose()


def test_http_document_fetcher_extracts_pdf_and_prefers_request_title() -> None:
    content = _make_text_pdf("Report body.", title="PDF title")

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
        _fetch_pdf(_make_text_pdf("Report body.", title="PDF title"))
    )

    assert document.title == "PDF title"


def test_http_document_fetcher_reports_pdf_without_extractable_text() -> None:
    try:
        asyncio.run(_fetch_pdf(_make_text_pdf("")))
    except DocumentFetchError as error:
        assert error.reason == "no_extractable_text"
        assert error.status_code == 200
        assert error.source_error_type == "NoExtractableText"
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_reports_malformed_pdf() -> None:
    try:
        asyncio.run(_fetch_pdf(b"%PDF-broken"))
    except DocumentFetchError as error:
        assert error.reason == "parse_error"
        assert error.status_code == 200
        assert error.__cause__ is not None
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_reports_password_protected_pdf() -> None:
    encrypted = _encrypt_pdf(_make_text_pdf("Secret report."))

    try:
        asyncio.run(_fetch_pdf(encrypted))
    except DocumentFetchError as error:
        assert error.reason == "parse_error"
        assert error.status_code == 200
        assert error.__cause__ is not None
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_enforces_pdf_byte_limit() -> None:
    content = _make_text_pdf("Report body.")

    try:
        asyncio.run(_fetch_pdf(content, max_pdf_bytes=len(content) - 1))
    except DocumentFetchError as error:
        assert error.reason == "document_too_large"
        assert error.source_error_type == "PDFByteLimitExceeded"
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_enforces_pdf_page_limit() -> None:
    content = _make_text_pdf("First page.", "Second page.")

    try:
        asyncio.run(_fetch_pdf(content, max_pdf_pages=1))
    except DocumentFetchError as error:
        assert error.reason == "document_too_large"
        assert error.source_error_type == "PDFPageLimitExceeded"
        assert error.__cause__ is not None
    else:
        raise AssertionError("expected DocumentFetchError")


def test_http_document_fetcher_validates_pdf_limits() -> None:
    for kwargs in ({"max_pdf_bytes": 0}, {"max_pdf_pages": 0}):
        try:
            HTTPDocumentFetcher(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

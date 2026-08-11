"""Tests for reusable HTML and PDF document parsing."""

import asyncio
from datetime import datetime, timezone

from banso.documents import DocumentParser
from tests.pdf_fixtures import make_text_pdf


def _parse_html(html: str):
    return asyncio.run(
        DocumentParser().parse(
            content=html,
            media_type="text/html",
        )
    )


def test_html_extraction_prefers_longest_article_and_removes_noise() -> None:
    extraction = _parse_html(
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
    main_extraction = _parse_html(
        "<body><main><h1>Main heading</h1><p>Main text.</p></main></body>"
    )
    role_extraction = _parse_html(
        '<body><section role="main"><p>Role main text.</p></section></body>'
    )

    assert main_extraction.strategy == "main"
    assert main_extraction.text == "Main heading\nMain text."
    assert role_extraction.strategy == "role_main"
    assert role_extraction.text == "Role main text."


def test_html_extraction_prefers_main_over_article_cards() -> None:
    extraction = _parse_html(
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
    extraction = _parse_html(
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


def test_html_extraction_reads_json_ld_publication_date_before_open_graph() -> None:
    extraction = _parse_html(
        """
        <html>
          <head>
            <meta property="article:published_time" content="2026-08-01T00:00:00Z">
            <script type="application/ld+json">
              [{"@graph": [{"@type": "NewsArticle", "datePublished": "2026-08-10"}]}]
            </script>
          </head>
          <body><main><p>Article body.</p></main></body>
        </html>
        """
    )

    assert extraction.published_at == datetime(2026, 8, 10, tzinfo=timezone.utc)


def test_html_extraction_uses_valid_open_graph_publication_date() -> None:
    extraction = _parse_html(
        """
        <html>
          <head>
            <script type="application/ld+json">{"datePublished": "not-a-date"}</script>
            <meta property="article:published_time" content="2026-08-09T18:00:00-04:00">
          </head>
          <body><main><p>Article body.</p></main></body>
        </html>
        """
    )

    assert extraction.published_at == datetime(
        2026, 8, 9, 22, tzinfo=timezone.utc
    )


def test_pdf_extraction_combines_pages_and_extracts_title() -> None:
    content = make_text_pdf("First page.", "Second page.", title="PDF title")

    extraction = asyncio.run(
        DocumentParser(max_pdf_pages=10).parse(
            content=content,
            media_type="application/pdf",
        )
    )

    assert extraction.title == "PDF title"
    assert extraction.text == "First page.\n\nSecond page."
    assert extraction.strategy == "pypdf"
    assert extraction.metadata == {
        "raw_bytes": len(content),
        "pdf_page_count": 2,
        "pdf_pages_with_text": 2,
    }


def test_document_parser_validates_pdf_limits() -> None:
    for kwargs in ({"max_pdf_bytes": 0}, {"max_pdf_pages": 0}):
        try:
            DocumentParser(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

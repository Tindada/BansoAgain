"""Reusable HTML and PDF document parsing."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any, Literal

from bs4 import BeautifulSoup
from bs4.element import Tag
from pypdf import PdfReader

from banso.datetime_utils import parse_external_datetime


HTML_CONTENT_TYPES = frozenset({"application/xhtml+xml", "text/html"})
PDF_CONTENT_TYPE = "application/pdf"
SUPPORTED_CONTENT_TYPES = frozenset({*HTML_CONTENT_TYPES, PDF_CONTENT_TYPE})
_DEFAULT_MAX_PDF_BYTES = 20 * 1024 * 1024
_DEFAULT_MAX_PDF_PAGES = 200

DocumentParseFailureReason = Literal[
    "document_too_large",
    "no_extractable_text",
    "parse_error",
]


@dataclass(frozen=True)
class ParsedDocument:
    """Plain-text content and metadata extracted from a document body."""

    title: str | None
    text: str
    strategy: str
    published_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParseError(Exception):
    """A known failure while parsing a supported document body."""

    def __init__(
        self,
        *,
        reason: DocumentParseFailureReason,
        message: str,
        source_error_type: str,
    ) -> None:
        self.reason = reason
        self.message = message
        self.source_error_type = source_error_type
        super().__init__(message)


class _PDFPageLimitError(Exception):
    """Raised when a PDF exceeds the configured page limit."""


class DocumentParser:
    """Extracts plain text from supported HTML and PDF document bodies."""

    def __init__(
        self,
        *,
        max_pdf_bytes: int = _DEFAULT_MAX_PDF_BYTES,
        max_pdf_pages: int = _DEFAULT_MAX_PDF_PAGES,
    ) -> None:
        if max_pdf_bytes <= 0:
            raise ValueError("max_pdf_bytes must be greater than zero")
        if max_pdf_pages <= 0:
            raise ValueError("max_pdf_pages must be greater than zero")

        self._max_pdf_bytes = max_pdf_bytes
        self._max_pdf_pages = max_pdf_pages

    async def parse(
        self,
        *,
        content: str | bytes,
        media_type: str,
    ) -> ParsedDocument:
        """Parse a document body whose normalized media type is known."""

        if media_type in HTML_CONTENT_TYPES:
            html = (
                content
                if isinstance(content, str)
                else content.decode("utf-8", errors="replace")
            )
            extraction = _extract_html_content(html)
            if not extraction.text.strip():
                raise DocumentParseError(
                    reason="no_extractable_text",
                    message="HTML contains no extractable text",
                    source_error_type="NoExtractableText",
                )
            return extraction

        if media_type != PDF_CONTENT_TYPE:
            raise ValueError(f"unsupported document media type: {media_type}")
        if isinstance(content, str):
            raise TypeError("PDF document content must be bytes")

        if len(content) > self._max_pdf_bytes:
            raise DocumentParseError(
                reason="document_too_large",
                message=(
                    "PDF exceeds the configured byte limit; "
                    f"max_pdf_bytes={self._max_pdf_bytes}; "
                    f"pdf_bytes={len(content)}"
                ),
                source_error_type="PDFByteLimitExceeded",
            )

        try:
            extraction = await asyncio.to_thread(
                _extract_pdf_content,
                content,
                max_pages=self._max_pdf_pages,
            )
        except _PDFPageLimitError as error:
            raise DocumentParseError(
                reason="document_too_large",
                message=str(error),
                source_error_type="PDFPageLimitExceeded",
            ) from error
        except Exception as error:
            raise DocumentParseError(
                reason="parse_error",
                message=str(error) or "PDF parser failed",
                source_error_type=type(error).__name__,
            ) from error

        if not extraction.text.strip():
            raise DocumentParseError(
                reason="no_extractable_text",
                message="PDF contains no extractable text",
                source_error_type="NoExtractableText",
            )
        return extraction


_REMOVABLE_TAGS = ("script", "style", "noscript", "template", "svg")
_CONTENT_NOISE_TAGS = ("nav", "aside", "form", "dialog")
_CONTENT_BLOCK_TAGS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "blockquote",
    "pre",
    "table",
)


def _extract_html_content(html: str) -> ParsedDocument:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    published_at = _extract_html_published_at(soup)

    for tag in soup(_REMOVABLE_TAGS):
        tag.decompose()

    content_root, strategy = _select_content_root(soup)
    _remove_content_noise(content_root, strategy=strategy)
    text = _extract_block_text(content_root)

    return ParsedDocument(
        title=title,
        text=text,
        strategy=strategy,
        published_at=published_at,
        metadata={"raw_html_chars": len(html)},
    )


def _extract_html_published_at(soup: BeautifulSoup) -> datetime | None:
    for script in soup.find_all("script"):
        script_type = script.get("type")
        if (
            not isinstance(script_type, str)
            or script_type.strip().casefold() != "application/ld+json"
        ):
            continue
        try:
            payload = json.loads(script.string or script.get_text())
        except (TypeError, ValueError):
            continue
        published_at = _find_json_ld_published_at(payload)
        if published_at is not None:
            return published_at

    for meta in soup.find_all("meta"):
        property_name = meta.get("property")
        if not isinstance(property_name, str):
            continue
        if property_name.strip().casefold() != "article:published_time":
            continue
        published_at = parse_external_datetime(meta.get("content"))
        if published_at is not None:
            return published_at
    return None


def _find_json_ld_published_at(value: object) -> datetime | None:
    if isinstance(value, dict):
        published_at = parse_external_datetime(value.get("datePublished"))
        if published_at is not None:
            return published_at
        for nested in value.values():
            published_at = _find_json_ld_published_at(nested)
            if published_at is not None:
                return published_at
    elif isinstance(value, list):
        for nested in value:
            published_at = _find_json_ld_published_at(nested)
            if published_at is not None:
                return published_at
    return None


def _extract_pdf_content(content: bytes, *, max_pages: int) -> ParsedDocument:
    reader = PdfReader(BytesIO(content))
    page_count = len(reader.pages)
    if page_count > max_pages:
        raise _PDFPageLimitError(
            "PDF exceeds the configured page limit; "
            f"max_pdf_pages={max_pages}; pdf_page_count={page_count}"
        )

    page_texts: list[str] = []
    pages_with_text = 0
    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            page_texts.append(page_text)
            pages_with_text += 1

    metadata_title = reader.metadata.title if reader.metadata is not None else None
    title = str(metadata_title).strip() if metadata_title else None
    return ParsedDocument(
        title=title or None,
        text="\n\n".join(page_texts),
        strategy="pypdf",
        metadata={
            "raw_bytes": len(content),
            "pdf_page_count": page_count,
            "pdf_pages_with_text": pages_with_text,
        },
    )


def _select_content_root(soup: BeautifulSoup) -> tuple[Tag, str]:
    main = soup.find("main")
    if isinstance(main, Tag):
        return main, "main"

    role_main = soup.find(attrs={"role": "main"})
    if isinstance(role_main, Tag):
        return role_main, "role_main"

    articles = soup.find_all("article")
    if articles:
        return max(articles, key=_text_length), "article"

    return soup.body or soup, "body"


def _remove_content_noise(content_root: Tag, *, strategy: str) -> None:
    for tag in content_root.find_all(_CONTENT_NOISE_TAGS):
        tag.decompose()

    if strategy == "body":
        for tag in content_root.find_all(("header", "footer"), recursive=False):
            tag.decompose()


def _extract_block_text(content_root: Tag) -> str:
    lines: list[str] = []

    for block in content_root.find_all(_CONTENT_BLOCK_TAGS):
        if _has_block_ancestor(block, content_root):
            continue
        text = (
            _extract_table_text(block)
            if block.name == "table"
            else block.get_text(separator=" ", strip=True)
        )
        if text:
            lines.append(text)

    if lines:
        return "\n".join(lines)

    return content_root.get_text(separator=" ", strip=True)


def _has_block_ancestor(block: Tag, content_root: Tag) -> bool:
    for parent in block.parents:
        if parent is content_root:
            return False
        if isinstance(parent, Tag) and parent.name in _CONTENT_BLOCK_TAGS:
            return True
    return False


def _extract_table_text(table: Tag) -> str:
    content_lines: list[str] = []
    caption = table.find("caption", recursive=False)
    if isinstance(caption, Tag):
        caption_text = _compact_tag_text(caption)
        if caption_text:
            content_lines.append(f"Caption: {caption_text}")

    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = row.find_all(("th", "td"), recursive=False)
        if not cells:
            continue
        content_lines.append("\t".join(_extract_table_cell(cell) for cell in cells))

    if not content_lines:
        return ""
    return "\n".join(("[Table]", *content_lines, "[/Table]"))


def _extract_table_cell(cell: Tag) -> str:
    text = _compact_tag_text(cell)
    spans = [
        annotation
        for attribute in ("rowspan", "colspan")
        if (annotation := _table_span_annotation(cell, attribute)) is not None
    ]
    if not spans:
        return text
    annotation = f"[{' '.join(spans)}]"
    return f"{text} {annotation}" if text else annotation


def _table_span_annotation(cell: Tag, attribute: str) -> str | None:
    value = cell.get(attribute)
    if not isinstance(value, str):
        return None
    try:
        span = int(value)
    except ValueError:
        return None
    if span <= 1:
        return None
    return f"{attribute}={span}"


def _compact_tag_text(tag: Tag) -> str:
    return " ".join(tag.stripped_strings)


def _text_length(tag: Tag) -> int:
    return len(tag.get_text(separator=" ", strip=True))

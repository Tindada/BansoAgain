"""HTTP-backed document reader."""

from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag

from banso.documents.models import Document
from banso.documents.reader import DocumentReadError, DocumentReadRequest


_SUPPORTED_CONTENT_TYPES = {"application/xhtml+xml", "text/html"}


class HTTPDocumentReader:
    """Reads HTML documents over HTTP and extracts plain text."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._headers = headers or {
            "User-Agent": "banso-news-agent/0.1",
            "Accept": "text/html,application/xhtml+xml",
        }

    async def read(self, request: DocumentReadRequest) -> Document:
        """Fetch and parse a document from the requested URL."""

        try:
            if self._client is not None:
                response = await self._client.get(request.url)
            else:
                async with httpx.AsyncClient(
                    timeout=self._timeout,
                    headers=self._headers,
                    follow_redirects=True,
                ) as client:
                    response = await client.get(request.url)

            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            message = (
                f"HTTP {error.response.status_code} while reading document: "
                f"{error.response.url}"
            )
            raise DocumentReadError(
                url=str(response.url),
                status_code=response.status_code,
                reason="http_status",
                message=message,
                source_error_type=type(error).__name__,
            ) from error
        except httpx.TimeoutException as error:
            message = str(error) or f"Timed out while reading document: {request.url}"
            raise DocumentReadError(
                url=request.url,
                reason="timeout",
                message=message,
                source_error_type=type(error).__name__,
            ) from error
        except httpx.TransportError as error:
            message = str(error) or (f"Transport error while reading document: {request.url}")
            raise DocumentReadError(
                url=request.url,
                reason="transport",
                message=message,
                source_error_type=type(error).__name__,
            ) from error

        content_type = response.headers.get("content-type")
        media_type = _media_type(content_type)
        if media_type not in _SUPPORTED_CONTENT_TYPES:
            displayed_content_type = content_type or "<missing>"
            message = (
                f"Unsupported content type {displayed_content_type!r} while reading "
                f"document: {response.url}"
            )
            raise DocumentReadError(
                url=str(response.url),
                status_code=response.status_code,
                reason="unsupported_content_type",
                message=message,
                source_error_type="UnsupportedContentType",
            )

        title, text, extraction_strategy = _extract_html_content(response.text)
        resolved_title = request.title or title or request.url

        metadata: dict[str, Any] = {
            **request.metadata,
            "reader": "http",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "final_url": str(response.url),
            "extraction_strategy": extraction_strategy,
            "raw_html_chars": len(response.text),
            "extracted_text_chars": len(text),
        }

        return Document(
            url=str(response.url),
            title=resolved_title,
            text=text,
            source=request.source,
            retrieved_at=datetime.now(timezone.utc),
            metadata=metadata,
        )


def _media_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    media_type = content_type.partition(";")[0].strip().casefold()
    return media_type or None


_REMOVABLE_TAGS = ("script", "style", "noscript", "template", "svg")
_CONTENT_NOISE_TAGS = ("nav", "aside", "form", "dialog")
_TEXT_BLOCK_TAGS = (
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
)


def _extract_html_content(html: str) -> tuple[str | None, str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    for tag in soup(_REMOVABLE_TAGS):
        tag.decompose()

    content_root, strategy = _select_content_root(soup)
    _remove_content_noise(content_root, strategy=strategy)
    text = _extract_block_text(content_root)

    return title, text, strategy


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

    for block in content_root.find_all(_TEXT_BLOCK_TAGS):
        if _has_block_ancestor(block, content_root):
            continue
        text = block.get_text(separator=" ", strip=True)
        if text:
            lines.append(text)

    if lines:
        return "\n".join(lines)

    return content_root.get_text(separator=" ", strip=True)


def _has_block_ancestor(block: Tag, content_root: Tag) -> bool:
    for parent in block.parents:
        if parent is content_root:
            return False
        if isinstance(parent, Tag) and parent.name in _TEXT_BLOCK_TAGS:
            return True
    return False


def _text_length(tag: Tag) -> int:
    return len(tag.get_text(separator=" ", strip=True))

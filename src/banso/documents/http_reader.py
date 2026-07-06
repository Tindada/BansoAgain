"""HTTP-backed document reader."""

from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag

from banso.documents.models import Document
from banso.documents.reader import DocumentHTTPStatusError, DocumentReadRequest


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

        if self._client is not None:
            response = await self._client.get(request.url)
        else:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers=self._headers,
                follow_redirects=True,
            ) as client:
                response = await client.get(request.url)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise DocumentHTTPStatusError(
                url=str(response.url),
                status_code=response.status_code,
            ) from error

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
    articles = soup.find_all("article")
    if articles:
        return max(articles, key=_text_length), "article"

    main = soup.find("main")
    if isinstance(main, Tag):
        return main, "main"

    role_main = soup.find(attrs={"role": "main"})
    if isinstance(role_main, Tag):
        return role_main, "role_main"

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

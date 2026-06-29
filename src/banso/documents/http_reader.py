"""HTTP-backed document reader."""

from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

from banso.documents.models import Document
from banso.documents.reader import DocumentReadRequest


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

        response.raise_for_status()

        title, text = _extract_html_content(response.text)
        resolved_title = request.title or title or request.url

        metadata: dict[str, Any] = {
            **request.metadata,
            "reader": "http",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "final_url": str(response.url),
        }

        return Document(
            url=str(response.url),
            title=resolved_title,
            text=text,
            source=request.source,
            retrieved_at=datetime.now(timezone.utc),
            metadata=metadata,
        )


def _extract_html_content(html: str) -> tuple[str | None, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    body = soup.body or soup
    text = body.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)

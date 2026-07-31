"""Conditional fetching and parsing for corpus content pages."""

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from banso.documents.parser import (
    HTML_CONTENT_TYPES,
    SUPPORTED_CONTENT_TYPES,
    DocumentParser,
    ParsedDocument,
)


@dataclass(frozen=True)
class PageFetchResult:
    """A parsed page and the HTTP metadata needed for corpus storage."""

    final_url: str
    document: ParsedDocument
    media_type: str
    fetched_at: datetime
    etag: str | None
    last_modified: str | None


class CorpusPageFetcher:
    """Fetch and parse HTML or PDF pages with saved HTTP validators."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        parser: DocumentParser | None = None,
        timeout: float = 20.0,
        user_agent: str = "banso-corpus/0.1",
    ) -> None:
        self._client = client
        self._parser = parser or DocumentParser()
        self._timeout = timeout
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/pdf",
        }

    async def fetch(
        self,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> PageFetchResult | None:
        """Return a parsed page, or ``None`` when it was not modified."""

        headers = dict(self._headers)
        if etag is not None:
            headers["If-None-Match"] = etag
        if last_modified is not None:
            headers["If-Modified-Since"] = last_modified

        if self._client is not None:
            response = await self._client.get(url, headers=headers)
        else:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url, headers=headers)

        if response.status_code == 304:
            return None
        response.raise_for_status()

        content_type = response.headers.get("content-type")
        media_type = (
            content_type.partition(";")[0].strip().casefold()
            if content_type is not None
            else ""
        )
        if media_type not in SUPPORTED_CONTENT_TYPES:
            raise ValueError(f"unsupported document media type: {media_type or None}")

        document = await self._parser.parse(
            content=(
                response.text
                if media_type in HTML_CONTENT_TYPES
                else response.content
            ),
            media_type=media_type,
        )
        return PageFetchResult(
            final_url=str(response.url),
            document=document,
            media_type=media_type,
            fetched_at=datetime.now(timezone.utc),
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
        )

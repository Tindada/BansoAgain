"""HTTP-backed document fetcher."""

from datetime import datetime, timezone
from typing import Any

import httpx

from banso.documents.fetcher import DocumentFetchError, DocumentFetchRequest
from banso.documents.models import Document
from banso.documents.parser import (
    HTML_CONTENT_TYPES,
    SUPPORTED_CONTENT_TYPES,
    DocumentParseError,
    DocumentParser,
    ParsedDocument,
)


class HTTPDocumentFetcher:
    """Fetches supported documents over HTTP and extracts plain text."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
        headers: dict[str, str] | None = None,
        parser: DocumentParser | None = None,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._parser = parser or DocumentParser()
        self._headers = headers or {
            "User-Agent": "banso-news-agent/0.1",
            "Accept": "text/html,application/xhtml+xml,application/pdf",
        }

    async def fetch(self, request: DocumentFetchRequest) -> Document:
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
                f"HTTP {error.response.status_code} while fetching document: "
                f"{error.response.url}"
            )
            raise DocumentFetchError(
                url=str(response.url),
                status_code=response.status_code,
                reason="http_status",
                message=message,
                source_error_type=type(error).__name__,
            ) from error
        except httpx.TimeoutException as error:
            message = str(error) or f"Timed out while fetching document: {request.url}"
            raise DocumentFetchError(
                url=request.url,
                reason="timeout",
                message=message,
                source_error_type=type(error).__name__,
            ) from error
        except httpx.TransportError as error:
            message = str(error) or (f"Transport error while fetching document: {request.url}")
            raise DocumentFetchError(
                url=request.url,
                reason="transport",
                message=message,
                source_error_type=type(error).__name__,
            ) from error

        content_type = response.headers.get("content-type")
        media_type = _media_type(content_type)
        if media_type not in SUPPORTED_CONTENT_TYPES:
            displayed_content_type = content_type or "<missing>"
            message = (
                f"Unsupported content type {displayed_content_type!r} while fetching "
                f"document: {response.url}"
            )
            raise DocumentFetchError(
                url=str(response.url),
                status_code=response.status_code,
                reason="unsupported_content_type",
                message=message,
                source_error_type="UnsupportedContentType",
            )

        extraction = await self._extract_response(response, media_type)
        resolved_title = request.title or extraction.title or request.url

        metadata: dict[str, Any] = {
            **request.metadata,
            "fetcher": "http",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "final_url": str(response.url),
            "extraction_strategy": extraction.strategy,
            **extraction.metadata,
            "extracted_text_chars": len(extraction.text),
        }

        return Document(
            url=str(response.url),
            title=resolved_title,
            text=extraction.text,
            source=request.source,
            retrieved_at=datetime.now(timezone.utc),
            metadata=metadata,
        )

    async def _extract_response(
        self,
        response: httpx.Response,
        media_type: str,
    ) -> ParsedDocument:
        try:
            return await self._parser.parse(
                content=(
                    response.text
                    if media_type in HTML_CONTENT_TYPES
                    else response.content
                ),
                media_type=media_type,
            )
        except DocumentParseError as error:
            raise DocumentFetchError(
                url=str(response.url),
                status_code=response.status_code,
                reason=error.reason,
                message=f"{error.message}; url={response.url}",
                source_error_type=error.source_error_type,
            ) from error


def _media_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    media_type = content_type.partition(";")[0].strip().casefold()
    return media_type or None

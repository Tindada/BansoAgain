"""Jina Reader-backed document fetcher."""

import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from banso.datetime_utils import parse_external_datetime
from banso.documents.fetcher import DocumentFetchError, DocumentFetchRequest
from banso.documents.models import Document
from banso.http_errors import classify_httpx_error


_TARGET_HTTP_ERROR = re.compile(r"Target URL returned error\s+(\d{3})\b", re.I)


class _JinaModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _JinaUsage(_JinaModel):
    tokens: int | None = Field(default=None, ge=0)


class _JinaData(_JinaModel):
    title: str | None = None
    url: str | None = None
    content: str
    published_time: str | None = Field(default=None, alias="publishedTime")
    warning: str | None = None
    usage: _JinaUsage | None = None


class _JinaResponse(_JinaModel):
    code: Literal[200]
    status: Literal[20000]
    data: _JinaData


class JinaDocumentFetcher:
    """Fetch already-extracted Markdown through the Jina Reader API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://r.jina.ai",
        timeout: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        base_url = base_url.strip().rstrip("/")
        if not base_url:
            raise ValueError("base_url must not be blank")

        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._base_url = base_url
        self._timeout = timeout
        self._client = client

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        """Fetch one URL from Jina Reader and return its extracted Markdown."""

        reader_url = f"{self._base_url}/{request.url}"
        headers = {
            "Accept": "application/json",
            "X-No-Cache": "true",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            if self._client is not None:
                response = await self._client.get(
                    reader_url,
                    headers=headers,
                    timeout=self._timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(reader_url, headers=headers)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            failure = classify_httpx_error(error)
            raise DocumentFetchError(
                url=request.url,
                status_code=failure.status_code,
                reason=failure.kind,
                message=str(error) or f"Jina Reader request failed: {request.url}",
                source_error_type=failure.source_error_type,
            ) from error

        payload = self._parse_payload(response, request.url)
        data = payload.data

        final_url = (data.url or "").strip() or request.url
        warning = (data.warning or "").strip() or None
        if warning is not None:
            match = _TARGET_HTTP_ERROR.search(warning)
            if match is not None:
                status_code = int(match.group(1))
                raise DocumentFetchError(
                    url=final_url,
                    status_code=status_code,
                    reason="http_status",
                    message=f"Jina Reader could not fetch target URL: {warning}",
                    source_error_type="JinaTargetHTTPError",
                )

        text = data.content.strip()
        if not text:
            raise DocumentFetchError(
                url=final_url,
                status_code=response.status_code,
                reason="no_extractable_text",
                message=f"Jina Reader returned no extractable text: {final_url}",
                source_error_type="NoExtractableText",
            )

        jina_title = (data.title or "").strip() or None
        usage_tokens = data.usage.tokens if data.usage is not None else None
        metadata: dict[str, Any] = {
            **request.metadata,
            "fetcher": "jina_reader",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "final_url": final_url,
            "extraction_strategy": "jina_reader",
            "jina_code": payload.code,
            "jina_status": payload.status,
            "jina_usage_tokens": usage_tokens,
            "extracted_text_chars": len(text),
        }
        if warning is not None:
            metadata["jina_warning"] = warning

        return Document(
            url=final_url,
            title=request.title or jina_title or request.url,
            text=text,
            source=request.source,
            published_at=parse_external_datetime(data.published_time),
            metadata=metadata,
        )

    @staticmethod
    def _parse_payload(
        response: httpx.Response,
        request_url: str,
    ) -> _JinaResponse:
        try:
            raw_payload = response.json()
        except ValueError as error:
            raise DocumentFetchError(
                url=request_url,
                status_code=response.status_code,
                reason="parse_error",
                message=f"Jina Reader returned invalid JSON: {request_url}",
                source_error_type=type(error).__name__,
            ) from error

        try:
            return _JinaResponse.model_validate(raw_payload)
        except ValidationError as error:
            raise DocumentFetchError(
                url=request_url,
                status_code=response.status_code,
                reason="parse_error",
                message=f"Jina Reader response envelope is invalid: {request_url}",
                source_error_type="JinaResponseValidationError",
            ) from error

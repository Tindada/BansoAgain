"""Document fetcher interface."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from banso.documents.models import Document
from banso.http_errors import is_retryable_http_status
from banso.retrieval.models import Source


DocumentFetchFailureReason = Literal[
    "document_too_large",
    "http_status",
    "no_extractable_text",
    "parse_error",
    "timeout",
    "transport",
    "unsupported_content_type",
]


class DocumentFetchError(Exception):
    """A known external failure while fetching a document."""

    def __init__(
        self,
        *,
        url: str,
        reason: DocumentFetchFailureReason,
        message: str,
        source_error_type: str,
        status_code: int | None = None,
    ) -> None:
        self.url = url
        self.reason = reason
        self.message = message
        self.source_error_type = source_error_type
        self.status_code = status_code
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        """Return whether repeating the same fetch may recover."""
        if self.reason in {"timeout", "transport"}:
            return True
        if self.reason != "http_status" or self.status_code is None:
            return False
        return is_retryable_http_status(self.status_code)


class DocumentFetchRequest(BaseModel):
    """Structured request for fetching a document."""

    url: str
    title: str | None = None
    source: Source | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentFetcher(Protocol):
    """Fetches and parses documents from a source location."""

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        """Return a parsed document."""
        ...

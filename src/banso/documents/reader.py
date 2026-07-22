"""Document reader interface."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from banso.documents.models import Document
from banso.retrieval.models import Source


DocumentReadFailureReason = Literal[
    "document_too_large",
    "http_status",
    "no_extractable_text",
    "parse_error",
    "timeout",
    "transport",
    "unsupported_content_type",
]


class DocumentReadError(Exception):
    """A known external failure while reading a document."""

    def __init__(
        self,
        *,
        url: str,
        reason: DocumentReadFailureReason,
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


class DocumentReadRequest(BaseModel):
    """Structured request for reading a document."""

    url: str
    title: str | None = None
    source: Source | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentReader(Protocol):
    """Reads and parses documents from a source location."""

    async def read(self, request: DocumentReadRequest) -> Document:
        """Return a parsed document."""
        ...

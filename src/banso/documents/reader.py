"""Document reader interface."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.documents.models import Document
from banso.retrieval.models import Source


class DocumentHTTPStatusError(Exception):
    """Raised when a document server returns an unsuccessful HTTP status."""

    def __init__(self, *, url: str, status_code: int) -> None:
        self.url = url
        self.status_code = status_code
        super().__init__(f"HTTP {status_code} while reading document: {url}")


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

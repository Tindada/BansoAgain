"""Document reader interface."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.documents.models import Document
from banso.retrieval.models import Source


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

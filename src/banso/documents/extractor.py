"""Evidence extraction interface."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.documents.models import Document, EvidenceItem
from banso.core.state import UserQuery


class EvidenceExtractionRequest(BaseModel):
    """Structured request for extracting evidence from a document."""

    query: UserQuery
    document: Document
    max_items_per_chunk: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceExtractionError(Exception):
    """Raised when an extractor cannot interpret its response as evidence."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class EvidenceExtractor(Protocol):
    """Extracts evidence items from documents."""

    async def extract(self, request: EvidenceExtractionRequest) -> list[EvidenceItem]:
        """Return evidence items extracted from a document."""
        ...

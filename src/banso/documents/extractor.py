"""Evidence extraction interface."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.documents.models import Document, EvidenceItem


class EvidenceExtractionRequest(BaseModel):
    """Structured request for extracting evidence from a document."""

    query: str
    document: Document
    max_items_per_chunk: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceExtractionError(Exception):
    """Raised when an extractor cannot interpret its response as evidence."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason

    @property
    def retryable(self) -> bool:
        """Return whether repeating the same extraction may recover."""
        return self.reason == "llm_error"


class EvidenceExtractor(Protocol):
    """Extracts evidence items from documents."""

    async def extract(self, request: EvidenceExtractionRequest) -> list[EvidenceItem]:
        """Return evidence items extracted from a document."""
        ...

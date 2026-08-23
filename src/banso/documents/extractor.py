"""Evidence extraction interface."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.documents.models import Document


class EvidenceExtractionRequest(BaseModel):
    """Structured request for extracting evidence from a document."""

    query: str
    document: Document
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
    """Distills query-relevant evidence text from documents."""

    async def extract(self, request: EvidenceExtractionRequest) -> str | None:
        """Return query-relevant text distilled from a document, if any."""
        ...

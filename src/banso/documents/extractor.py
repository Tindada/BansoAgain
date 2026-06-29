"""Evidence extraction interface."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.documents.models import Document, EvidenceItem
from banso.core.state import UserQuery


class EvidenceExtractionRequest(BaseModel):
    """Structured request for extracting evidence from a document."""

    query: UserQuery
    document: Document
    max_items: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceExtractor(Protocol):
    """Extracts evidence items from documents."""

    async def extract(self, request: EvidenceExtractionRequest) -> list[EvidenceItem]:
        """Return evidence items extracted from a document."""
        ...

"""Document domain models."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from banso.source import Source


class Document(BaseModel):
    """A fetched and parsed document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    url: str
    title: str
    text: str
    source: Source | None = None
    published_at: datetime | None = None
    author: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """A claim or fact extracted from a document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    claim: str
    supporting_text: str | None = None
    source_url: str
    published_at: datetime | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

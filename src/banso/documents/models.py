"""Document domain models."""

from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, Field, StringConstraints

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


class DocumentEvidence(BaseModel):
    """Query-relevant text distilled from one document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

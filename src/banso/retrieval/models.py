"""Retrieval domain models."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from banso.source_types import SourceType


class Source(BaseModel):
    """Information source metadata."""

    name: str
    url: str | None = None
    type: SourceType = SourceType.UNKNOWN


class SearchResult(BaseModel):
    """A single result returned by a retrieval provider."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    url: str
    snippet: str | None = None
    source: Source | None = None
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

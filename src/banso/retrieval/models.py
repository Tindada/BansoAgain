"""Retrieval domain models."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from banso.source import Source


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


class RetrievalReportModel(BaseModel):
    """Strict base model for retrieval process reports."""

    model_config = ConfigDict(extra="forbid")


class SearchResultMergeReport(RetrievalReportModel):
    """Summary of new and reused results produced by one search."""

    candidate_count: int = Field(ge=0)
    new_result_count: int = Field(ge=0)
    reused_result_count: int = Field(ge=0)


class RetrievalFilterReport(RetrievalReportModel):
    """Summary of filtering decisions for one search."""

    input_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    dropped_empty_title: int = Field(default=0, ge=0)
    dropped_empty_url: int = Field(default=0, ge=0)
    dropped_invalid_url: int = Field(default=0, ge=0)
    dropped_duplicate_url: int = Field(default=0, ge=0)
    truncated_count: int = Field(default=0, ge=0)


class SourceClassificationRecord(RetrievalReportModel):
    """Trace-safe source classification for one search result."""

    search_result_id: str
    publisher_domain: str
    source_type: str
    classification_source: Literal["domain", "provider", "unknown"]


class SourceClassificationReport(RetrievalReportModel):
    """Summary of source classification decisions for one search."""

    input_count: int = Field(ge=0)
    recognized_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    classifications: list[SourceClassificationRecord] = Field(default_factory=list)


class SearchResultSelectionReport(RetrievalReportModel):
    """Candidate results and the ordered subset selected for processing."""

    candidate_ids: list[str]
    selected_ids: list[str]

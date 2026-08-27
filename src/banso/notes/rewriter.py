"""Research notes rewriting contracts."""

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from banso.source import Source


class NotesEvidenceGroup(BaseModel):
    """Complete extracted evidence available to a research notes rewrite."""

    document_ref: str
    title: str
    source_url: str
    source: Source | None = None
    published_at: datetime | None = None
    evidence_text: str = Field(min_length=1)


class NotesRewriteRequest(BaseModel):
    """Complete working state supplied to a research notes rewriter."""

    query: str
    language: str | None = None
    time_range: str | None = None
    reference_time: datetime
    current_notes: str
    research_history: list[dict[str, object]] = Field(default_factory=list)
    evidence_groups: list[NotesEvidenceGroup] = Field(default_factory=list)


class NotesRewriteResult(BaseModel):
    """Complete replacement research notes."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(max_length=32_000)


class NotesRewriter(Protocol):
    """Produces complete replacement research notes for the agent."""

    async def rewrite(self, request: NotesRewriteRequest) -> NotesRewriteResult:
        """Return rewritten notes from complete extracted evidence."""
        ...

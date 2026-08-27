"""Scratch rewriting contracts."""

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from banso.source import Source


class ScratchEvidenceGroup(BaseModel):
    """Complete extracted evidence available to a scratch rewrite."""

    document_ref: str
    title: str
    source_url: str
    source: Source | None = None
    published_at: datetime | None = None
    evidence_text: str = Field(min_length=1)


class ScratchRewriteRequest(BaseModel):
    """Complete working state supplied to a scratch rewriter."""

    query: str
    language: str | None = None
    time_range: str | None = None
    reference_time: datetime
    current_scratch: str
    research_history: list[dict[str, object]] = Field(default_factory=list)
    evidence_groups: list[ScratchEvidenceGroup] = Field(default_factory=list)


class ScratchRewriteResult(BaseModel):
    """A complete replacement scratch."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(max_length=32_000)


class ScratchRewriter(Protocol):
    """Produces a complete replacement for the agent's research scratch."""

    async def rewrite(self, request: ScratchRewriteRequest) -> ScratchRewriteResult:
        """Return a rewritten scratch from complete extracted evidence."""
        ...

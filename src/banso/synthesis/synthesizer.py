"""Synthesis interface."""

from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from banso.source import Source


class Citation(BaseModel):
    """A source-group reference used in the synthesized answer."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(pattern=r"^S[1-9]\d*$")
    document_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)


class SynthesisEvidenceGroup(BaseModel):
    """Evidence and source metadata grouped by document for synthesis."""

    document_id: str
    title: str
    source_url: str
    source: Source | None = None
    published_at: datetime | None = None
    evidence_text: str = Field(min_length=1)


class SynthesisRequest(BaseModel):
    """Structured request for synthesizing a final answer."""

    query: str
    language: str | None = None
    time_range: str | None = None
    reference_time: datetime
    evidence_groups: list[SynthesisEvidenceGroup] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SynthesisResult(BaseModel):
    """Structured synthesis output."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Synthesizer(Protocol):
    """Synthesizes evidence into a final answer."""

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Return a synthesized answer."""
        ...

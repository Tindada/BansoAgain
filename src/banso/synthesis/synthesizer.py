"""Synthesis interface."""

from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.core.observation import Citation
from banso.core.state import UserQuery
from banso.documents.models import EvidenceItem
from banso.retrieval.models import Source


class SynthesisEvidenceGroup(BaseModel):
    """Evidence and source metadata grouped by document for synthesis."""

    document_id: str
    title: str
    source_url: str
    source: Source | None = None
    published_at: datetime | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)


class SynthesisRequest(BaseModel):
    """Structured request for synthesizing a final answer."""

    query: UserQuery
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

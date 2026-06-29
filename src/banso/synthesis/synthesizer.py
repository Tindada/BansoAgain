"""Synthesis interface."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from banso.core.state import UserQuery
from banso.documents.models import Document, EvidenceItem


class SynthesisRequest(BaseModel):
    """Structured request for synthesizing a final answer."""

    query: UserQuery
    evidence: list[EvidenceItem] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SynthesisResult(BaseModel):
    """Structured synthesis output."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Synthesizer(Protocol):
    """Synthesizes evidence into a final answer."""

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Return a synthesized answer."""
        ...

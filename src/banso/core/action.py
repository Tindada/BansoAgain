"""Agent action models."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalRoute(StrEnum):
    """Semantic retrieval routes available to a research action."""

    WEB = "web"
    LOCAL = "local"


class AgentActionType(StrEnum):
    """Supported action types for the agent loop."""

    RESEARCH = "research"
    CURATE_EVIDENCE = "curate_evidence"
    FINISH = "finish"
    STOP = "stop"


class ResearchActionParams(BaseModel):
    """Strict parameters accepted by a research action."""

    model_config = ConfigDict(extra="forbid")

    query: str
    route: RetrievalRoute

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must be non-empty")
        return query


class AgentAction(BaseModel):
    """A structured decision selected by a policy."""

    type: AgentActionType
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None

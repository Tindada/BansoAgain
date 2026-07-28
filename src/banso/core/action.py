"""Agent action models."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, JsonValue


class AgentActionType(StrEnum):
    """Supported action types for the agent loop."""

    PLAN_SEARCH = "plan_search"
    SEARCH = "search"
    READ_DOCUMENT = "read_document"
    # RANK_DOCUMENTS = "rank_documents"
    EXTRACT_EVIDENCE = "extract_evidence"
    CURATE_EVIDENCE = "curate_evidence"
    FINISH = "finish"
    # ASK_CLARIFICATION = "ask_clarification"
    STOP = "stop"


class AgentAction(BaseModel):
    """A structured decision selected by a policy."""

    type: AgentActionType
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class Observation(BaseModel):
    """Structured result returned after executing an action."""

    data: dict[str, JsonValue] = Field(default_factory=dict)

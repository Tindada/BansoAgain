"""Agent observation and result models."""

from typing import Any

from pydantic import BaseModel, Field

from banso.core.action import AgentActionType
from banso.core.state import AgentState


class Observation(BaseModel):
    """Result returned after executing an action."""

    action_type: AgentActionType
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AgentResult(BaseModel):
    """Final result returned by an agent run."""

    final_answer: str | None = None
    state: AgentState
    metadata: dict[str, Any] = Field(default_factory=dict)

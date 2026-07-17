"""Agent observation and result models."""

from typing import Any

from pydantic import BaseModel, Field

from banso.core.state import AgentState


class Observation(BaseModel):
    """Result returned after executing an action."""

    data: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Final result returned by an agent run."""

    state: AgentState
    metadata: dict[str, Any] = Field(default_factory=dict)

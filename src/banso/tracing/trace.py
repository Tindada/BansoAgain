"""Trace models for recording agent execution."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from banso.core.action import AgentAction
from banso.core.result import AgentResult, Observation
from banso.core.state import AgentState, UserQuery


class TraceStep(BaseModel):
    """A single state-action-observation step in an agent run."""

    step_index: int
    state: AgentState
    action: AgentAction
    observation: Observation
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentTrace(BaseModel):
    """Complete execution trace for one agent run."""

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    query: UserQuery
    steps: list[TraceStep] = Field(default_factory=list)
    final_result: AgentResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

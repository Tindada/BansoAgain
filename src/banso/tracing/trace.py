"""Trace models for recording agent execution."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from banso.core.action import AgentAction
from banso.core.observation import Observation
from banso.core.result import AgentResult
from banso.core.state import AgentState, UserQuery


class TraceStep(BaseModel):
    """A single state-action-observation step in an agent run."""

    step_index: int
    state: AgentState
    action: AgentAction
    observation: Observation
    policy_duration_seconds: float | None = None
    executor_duration_seconds: float | None = None
    reducer_duration_seconds: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TraceFailure(BaseModel):
    """The point at which an agent run stopped unexpectedly."""

    phase: Literal["policy", "executor", "trace", "reducer"]
    step_index: int
    state: AgentState
    action: AgentAction | None = None
    observation: Observation | None = None
    error_type: str
    message: str
    phase_duration_seconds: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentTrace(BaseModel):
    """Complete execution trace for one agent run."""

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    query: UserQuery
    steps: list[TraceStep] = Field(default_factory=list)
    status: Literal["running", "completed", "failed"] = "running"
    failure: TraceFailure | None = None
    final_result: AgentResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

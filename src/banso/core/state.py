"""Agent state models."""

from typing import Any

from pydantic import BaseModel, Field

from banso.core.action import AgentActionType
from banso.core.observation import Observation


class ExecutionBudget(BaseModel):
    """Execution limits for a single agent run."""

    max_steps: int = 12
    max_searches: int = 3
    max_documents_to_read: int = 8


class UserQuery(BaseModel):
    """Original user request and lightweight query context."""

    text: str
    language: str | None = None
    region: str | None = None
    time_range: str | None = None


class PlannedSearch(BaseModel):
    """One query in an ordered search plan."""

    query: str
    intent: str = "general"


class SearchPlan(BaseModel):
    """Searches planned for one user query."""

    searches: list[PlannedSearch] = Field(default_factory=list)


class ActionHistoryEntry(BaseModel):
    """One completed action and its policy-visible observation."""

    step_index: int
    action_type: AgentActionType
    params: dict[str, Any] = Field(default_factory=dict)
    observation: Observation


class AgentState(BaseModel):
    """Mutable state observed by the policy at each step."""

    query: UserQuery
    current_step: int = 0
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    search_plan: SearchPlan | None = None
    action_history: list[ActionHistoryEntry] = Field(default_factory=list)
    search_result_ids: list[str] = Field(default_factory=list)
    search_result_index: dict[str, str] = Field(default_factory=dict)
    document_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    final_answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    last_action: AgentActionType | None = None
    done: bool = False

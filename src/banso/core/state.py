"""Agent state models."""

from pydantic import BaseModel, Field


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


class AgentState(BaseModel):
    """Mutable state observed by the policy at each step."""

    query: UserQuery
    current_step: int = 0
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    search_queries: list[str] = Field(default_factory=list)
    search_result_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    done: bool = False

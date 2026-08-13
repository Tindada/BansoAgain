"""Agent state models."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from banso.core.action import AgentAction, AgentActionType
from banso.core.observation import Observation


class ExecutionBudget(BaseModel):
    """Execution limits for a single agent run."""

    max_steps: int = 12
    max_researches: int = Field(default=3, ge=0)
    max_results_per_research: int = Field(default=4, ge=1)
    max_document_fetches: int = 8
    max_active_documents: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_active_document_limit(self) -> "ExecutionBudget":
        if self.max_active_documents is None:
            self.max_active_documents = self.max_document_fetches
        elif self.max_active_documents > self.max_document_fetches:
            raise ValueError("max_active_documents cannot exceed max_document_fetches")
        return self


class UserQuery(BaseModel):
    """Original user request and lightweight query context."""

    text: str
    language: str | None = None
    region: str | None = None
    time_range: str | None = None


class ActionHistoryEntry(BaseModel):
    """One completed action and its recorded runtime observation."""

    step_index: int
    action: AgentAction
    observation: Observation


class Failure(BaseModel):
    """A terminal resource-processing failure retained in agent state."""

    reason: str
    status_code: int | None = None


class SearchResultState(BaseModel):
    """Run-scoped processing state for one search result artifact."""

    document_id: str | None = None
    failure: Failure | None = None

    @model_validator(mode="after")
    def validate_fetch_state(self) -> "SearchResultState":
        if self.document_id is not None and self.failure is not None:
            raise ValueError("fetch result cannot contain both document_id and failure")
        return self


DocumentLifecycleStatus = Literal["active", "shelved", "unusable"]


class DocumentState(BaseModel):
    """Run-scoped processing state and evidence references for one document."""

    evidence_ids: list[str] = Field(default_factory=list)
    lifecycle_status: DocumentLifecycleStatus | None = None
    lifecycle_reason: str | None = None
    lifecycle_updated_at_step: int | None = Field(default=None, ge=0)


class AgentState(BaseModel):
    """Authoritative mutable record of runtime progress and artifact references."""

    query: UserQuery
    reference_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0)
    )
    current_step: int = 0
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    action_history: list[ActionHistoryEntry] = Field(default_factory=list)
    search_results: dict[str, SearchResultState] = Field(default_factory=dict)
    search_result_index: dict[str, str] = Field(default_factory=dict)
    documents: dict[str, DocumentState] = Field(default_factory=dict)
    document_index: dict[str, str] = Field(default_factory=dict)
    final_answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    last_action: AgentActionType | None = None
    done: bool = False

    @property
    def remaining_steps(self) -> int:
        """Return the number of actions remaining in the run."""
        return max(self.budget.max_steps - self.current_step, 0)

    @property
    def remaining_research_capacity(self) -> int:
        """Return the number of research actions remaining in the run."""
        completed = sum(
            entry.action.type == AgentActionType.RESEARCH
            for entry in self.action_history
        )
        return max(self.budget.max_researches - completed, 0)

    @property
    def remaining_document_capacity(self) -> int:
        """Return the number of unique documents the run may still collect."""
        return max(self.budget.max_document_fetches - len(self.documents), 0)

    @property
    def active_document_count(self) -> int:
        """Return the number of documents in the active working set."""
        return sum(
            document.lifecycle_status == "active"
            for document in self.documents.values()
        )

    @property
    def has_curatable_documents(self) -> bool:
        """Return whether any evidence-bearing document is available for curation."""
        return any(
            document.lifecycle_status in {"active", "shelved"}
            for document in self.documents.values()
        )

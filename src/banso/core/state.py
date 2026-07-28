"""Agent state models."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from banso.core.action import AgentActionType
from banso.core.observation import Observation


class ExecutionBudget(BaseModel):
    """Execution limits for a single agent run."""

    max_steps: int = 12
    max_searches: int = 3
    max_documents_to_read: int = 8
    max_active_documents: int | None = Field(default=None, ge=1)
    max_read_attempts: int = 2
    max_extraction_attempts: int = 2

    @model_validator(mode="after")
    def validate_active_document_limit(self) -> "ExecutionBudget":
        if self.max_active_documents is None:
            self.max_active_documents = self.max_documents_to_read
        elif self.max_active_documents > self.max_documents_to_read:
            raise ValueError("max_active_documents cannot exceed max_documents_to_read")
        return self


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
    """One completed action and its recorded runtime observation."""

    step_index: int
    action_type: AgentActionType
    params: dict[str, Any] = Field(default_factory=dict)
    observation: Observation


class Failure(BaseModel):
    """A resource-processing failure relevant to future decisions."""

    reason: str
    retryable: bool
    status_code: int | None = None


class ExtractProgress(BaseModel):
    """Current extraction lifecycle for one document."""

    attempt_count: int = Field(ge=1)
    failure: Failure | None = None


class SearchResultState(BaseModel):
    """Run-scoped processing state for one search result artifact."""

    attempt_count: int = Field(default=0, ge=0)
    document_id: str | None = None
    failure: Failure | None = None

    @model_validator(mode="after")
    def validate_read_state(self) -> "SearchResultState":
        if self.attempt_count == 0:
            if self.document_id is not None or self.failure is not None:
                raise ValueError("pending search result cannot contain a read outcome")
        elif (self.document_id is None) == (self.failure is None):
            raise ValueError("completed read must contain exactly one of document_id or failure")
        return self


DocumentLifecycleStatus = Literal["active", "shelved", "unusable"]


class DocumentState(BaseModel):
    """Run-scoped processing state and evidence references for one document."""

    extraction: ExtractProgress | None = None
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
    search_plan: SearchPlan | None = None
    action_history: list[ActionHistoryEntry] = Field(default_factory=list)
    search_results: dict[str, SearchResultState] = Field(default_factory=dict)
    search_result_index: dict[str, str] = Field(default_factory=dict)
    documents: dict[str, DocumentState] = Field(default_factory=dict)
    document_index: dict[str, str] = Field(default_factory=dict)
    final_answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    last_action: AgentActionType | None = None
    done: bool = False

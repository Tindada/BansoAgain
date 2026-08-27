"""Typed results produced by agent actions."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from banso.agent.action import AgentActionType, RetrievalRoute
from banso.retrieval.models import (
    RetrievalFilterReport,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)
from banso.synthesis.synthesizer import Citation


class ObservationModel(BaseModel):
    """Strict base model for values crossing the executor boundary."""

    model_config = ConfigDict(extra="forbid")


# Document fetch outcomes


class DocumentFetchFailure(ObservationModel):
    """Diagnostic details for a failed document fetch."""

    reason: str
    status_code: int | None = None
    url: str
    message: str
    source_error_type: str


class FetchSuccess(ObservationModel):
    """A search result that resolved to a document artifact."""

    status: Literal["success"] = "success"
    search_result_id: str
    document_id: str
    attempt_count: int = Field(default=1, ge=0)


class FetchFailure(ObservationModel):
    """A search result whose document fetch failed."""

    status: Literal["failure"] = "failure"
    search_result_id: str
    failure: DocumentFetchFailure
    attempt_count: int = Field(default=1, ge=1)


FetchOutcome = Annotated[
    FetchSuccess | FetchFailure,
    Field(discriminator="status"),
]


# Evidence extraction outcomes


class EvidenceExtractionFailure(ObservationModel):
    """Diagnostic details for a failed evidence extraction."""

    reason: str
    url: str
    message: str


class ExtractionSuccess(ObservationModel):
    """A document whose evidence extraction completed."""

    status: Literal["success"] = "success"
    document_id: str
    evidence_id: str | None = None
    attempt_count: int = Field(default=1, ge=1)


class ExtractionFailure(ObservationModel):
    """A document whose evidence extraction failed."""

    status: Literal["failure"] = "failure"
    document_id: str
    failure: EvidenceExtractionFailure
    attempt_count: int = Field(default=1, ge=1)


ExtractionOutcome = Annotated[
    ExtractionSuccess | ExtractionFailure,
    Field(discriminator="status"),
]


# Action observations


class ResearchObservationBase(ObservationModel):
    """Fields shared by all handled research action results."""

    type: Literal[AgentActionType.RESEARCH] = AgentActionType.RESEARCH
    query: str = Field(min_length=1)
    route: RetrievalRoute
    source_domains: list[str] | None = None


class CompletedResearchObservation(ResearchObservationBase):
    """Completed retrieval, selection, fetch, and extraction result."""

    status: Literal["completed"] = "completed"
    search_result_ids: list[str]
    retrieval_filter_report: RetrievalFilterReport
    source_classification_report: SourceClassificationReport
    search_result_merge_report: SearchResultMergeReport
    selection_report: SearchResultSelectionReport
    fetch_outcomes: list[FetchOutcome]
    extraction_outcomes: list[ExtractionOutcome]
    search_result_index_updates: dict[str, str]
    document_index_updates: dict[str, str]


class FailedResearchObservation(ResearchObservationBase):
    """Handled failure that prevented later research stages."""

    status: Literal["failed"] = "failed"
    stage: Literal["retrieval", "selection"]
    reason: str
    message: str
    source_error_type: str
    retryable: bool
    attempt_count: int = Field(ge=1)
    provider: str | None = None
    status_code: int | None = None


ResearchObservation = Annotated[
    CompletedResearchObservation | FailedResearchObservation,
    Field(discriminator="status"),
]


class FinishObservation(ObservationModel):
    """Final synthesized answer and its citations."""

    type: Literal[AgentActionType.FINISH] = AgentActionType.FINISH
    final_answer: str
    citations: list[Citation]


class StopObservation(ObservationModel):
    """Confirmation that execution stopped without synthesis."""

    type: Literal[AgentActionType.STOP] = AgentActionType.STOP


# Discriminated union and standalone parser


Observation = Annotated[
    ResearchObservation
    | FinishObservation
    | StopObservation,
    Field(discriminator="type"),
]

_OBSERVATION_ADAPTER = TypeAdapter(Observation)


def validate_observation(value: object) -> Observation:
    """Parse an observation value using its action type discriminator."""

    return _OBSERVATION_ADAPTER.validate_python(value)

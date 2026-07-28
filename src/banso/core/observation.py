"""Typed results produced by agent actions."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from banso.core.action import AgentActionType


class ObservationModel(BaseModel):
    """Strict base model for values crossing the executor boundary."""

    model_config = ConfigDict(extra="forbid")


# Shared value objects


class PlannedSearch(ObservationModel):
    """One query in an ordered search plan."""

    query: str
    intent: str = "general"


class SearchPlan(ObservationModel):
    """Searches planned for one user query."""

    searches: list[PlannedSearch] = Field(default_factory=list)


class Failure(ObservationModel):
    """A resource-processing failure relevant to future decisions."""

    reason: str
    retryable: bool
    status_code: int | None = None


# Search reports


class SearchResultMergeReport(ObservationModel):
    """Summary of new and reused results produced by one search."""

    candidate_count: int = Field(ge=0)
    new_result_count: int = Field(ge=0)
    reused_result_count: int = Field(ge=0)


class RetrievalFilterReport(ObservationModel):
    """Summary of filtering decisions for one search."""

    input_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    dropped_empty_title: int = Field(default=0, ge=0)
    dropped_empty_url: int = Field(default=0, ge=0)
    dropped_invalid_url: int = Field(default=0, ge=0)
    dropped_duplicate_url: int = Field(default=0, ge=0)
    truncated_count: int = Field(default=0, ge=0)


class SourceClassificationRecord(ObservationModel):
    """Trace-safe source classification for one search result."""

    search_result_id: str
    publisher_domain: str
    source_type: str
    classification_source: Literal["domain", "provider", "unknown"]


class SourceClassificationReport(ObservationModel):
    """Summary of source classification decisions for one search."""

    input_count: int = Field(ge=0)
    recognized_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    classifications: list[SourceClassificationRecord] = Field(default_factory=list)


# Document fetch outcomes


class DocumentFetchFailure(ObservationModel):
    """Diagnostic details for a failed document fetch."""

    reason: str
    retryable: bool
    status_code: int | None = None
    url: str
    message: str
    source_error_type: str


class FetchSuccess(ObservationModel):
    """A search result that resolved to a document artifact."""

    status: Literal["success"] = "success"
    search_result_id: str
    document_id: str


class FetchFailure(ObservationModel):
    """A search result whose document fetch failed."""

    status: Literal["failure"] = "failure"
    search_result_id: str
    failure: DocumentFetchFailure


FetchOutcome = Annotated[
    FetchSuccess | FetchFailure,
    Field(discriminator="status"),
]


# Evidence extraction outcomes


class EvidenceExtractionFailure(ObservationModel):
    """Diagnostic details for a failed evidence extraction."""

    reason: str
    retryable: bool
    url: str
    message: str


class ExtractionSuccess(ObservationModel):
    """A document whose evidence extraction completed."""

    status: Literal["success"] = "success"
    document_id: str
    evidence_ids: list[str]

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "ExtractionSuccess":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must be unique")
        return self


class ExtractionFailure(ObservationModel):
    """A document whose evidence extraction failed."""

    status: Literal["failure"] = "failure"
    document_id: str
    failure: EvidenceExtractionFailure


ExtractionOutcome = Annotated[
    ExtractionSuccess | ExtractionFailure,
    Field(discriminator="status"),
]


# Action observations


class PlanSearchObservation(ObservationModel):
    """Result of planning searches."""

    type: Literal[AgentActionType.PLAN_SEARCH] = AgentActionType.PLAN_SEARCH
    search_plan: SearchPlan


class SearchObservation(ObservationModel):
    """Result of executing one search."""

    type: Literal[AgentActionType.SEARCH] = AgentActionType.SEARCH
    search_queries: list[str]
    search_result_ids: list[str]
    search_result_index_updates: dict[str, str]
    search_result_merge_report: SearchResultMergeReport
    retrieval_filter_report: RetrievalFilterReport
    source_classification_report: SourceClassificationReport


class FetchDocumentsObservation(ObservationModel):
    """Result of fetching eligible search results."""

    type: Literal[AgentActionType.FETCH_DOCUMENTS] = AgentActionType.FETCH_DOCUMENTS
    fetch_outcomes: list[FetchOutcome]
    document_index_updates: dict[str, str]


class ExtractEvidenceObservation(ObservationModel):
    """Result of extracting evidence from eligible documents."""

    type: Literal[AgentActionType.EXTRACT_EVIDENCE] = AgentActionType.EXTRACT_EVIDENCE
    extraction_outcomes: list[ExtractionOutcome]


class CurateEvidenceObservation(ObservationModel):
    """Confirmation that evidence curation completed."""

    type: Literal[AgentActionType.CURATE_EVIDENCE] = AgentActionType.CURATE_EVIDENCE


class FinishObservation(ObservationModel):
    """Final synthesized answer and its citations."""

    type: Literal[AgentActionType.FINISH] = AgentActionType.FINISH
    final_answer: str
    citations: list[str]


class StopObservation(ObservationModel):
    """Confirmation that execution stopped without synthesis."""

    type: Literal[AgentActionType.STOP] = AgentActionType.STOP


# Discriminated union and standalone parser


Observation = Annotated[
    PlanSearchObservation
    | SearchObservation
    | FetchDocumentsObservation
    | ExtractEvidenceObservation
    | CurateEvidenceObservation
    | FinishObservation
    | StopObservation,
    Field(discriminator="type"),
]

_OBSERVATION_ADAPTER = TypeAdapter(Observation)


def validate_observation(value: object) -> Observation:
    """Parse an observation value using its action type discriminator."""

    return _OBSERVATION_ADAPTER.validate_python(value)

"""Typed results produced by agent actions."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from banso.core.action import AgentActionType, RetrievalRoute


class ObservationModel(BaseModel):
    """Strict base model for values crossing the executor boundary."""

    model_config = ConfigDict(extra="forbid")


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


class SearchResultSelectionReport(ObservationModel):
    """How one research action partitioned its ordered result candidates."""

    candidate_ids: list[str]
    selected_ids: list[str]
    deferred_ids: list[str]

    @model_validator(mode="after")
    def validate_partition(self) -> "SearchResultSelectionReport":
        groups = {
            "candidate_ids": self.candidate_ids,
            "selected_ids": self.selected_ids,
            "deferred_ids": self.deferred_ids,
        }
        for name, values in groups.items():
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique IDs")

        partition = [*self.selected_ids, *self.deferred_ids]
        if len(partition) != len(set(partition)):
            raise ValueError("selection groups must be disjoint")
        if set(partition) != set(self.candidate_ids):
            raise ValueError("selection groups must partition candidate_ids")
        candidate_order = {
            candidate_id: index
            for index, candidate_id in enumerate(self.candidate_ids)
        }
        for name in ("selected_ids", "deferred_ids"):
            values = groups[name]
            if values != sorted(values, key=candidate_order.__getitem__):
                raise ValueError(f"{name} must preserve candidate order")
        return self


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
    evidence_ids: list[str]
    attempt_count: int = Field(default=1, ge=1)

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
    attempt_count: int = Field(default=1, ge=1)


ExtractionOutcome = Annotated[
    ExtractionSuccess | ExtractionFailure,
    Field(discriminator="status"),
]


# Action observations


class RetrievalFailure(ObservationModel):
    """Diagnostic details for a failed retrieval operation."""

    provider: str
    reason: str
    status_code: int | None = None
    message: str
    source_error_type: str
    retryable: bool
    attempt_count: int = Field(ge=1)


class ResearchObservation(ObservationModel):
    """Combined retrieval, selection, fetch, and extraction result."""

    type: Literal[AgentActionType.RESEARCH] = AgentActionType.RESEARCH
    query: str = Field(min_length=1)
    route: RetrievalRoute
    retrieval_failure: RetrievalFailure | None = None
    search_result_ids: list[str]
    retrieval_filter_report: RetrievalFilterReport
    source_classification_report: SourceClassificationReport
    search_result_merge_report: SearchResultMergeReport
    selection_report: SearchResultSelectionReport
    fetch_outcomes: list[FetchOutcome]
    extraction_outcomes: list[ExtractionOutcome]
    search_result_index_updates: dict[str, str]
    document_index_updates: dict[str, str]

    @classmethod
    def from_retrieval_failure(
        cls,
        *,
        query: str,
        route: RetrievalRoute,
        failure: RetrievalFailure,
    ) -> "ResearchObservation":
        """Construct the canonical result of a failed retrieval."""
        return cls(
            query=query,
            route=route,
            retrieval_failure=failure,
            search_result_ids=[],
            retrieval_filter_report=RetrievalFilterReport(
                input_count=0,
                output_count=0,
            ),
            source_classification_report=SourceClassificationReport(
                input_count=0,
                recognized_count=0,
                unknown_count=0,
            ),
            search_result_merge_report=SearchResultMergeReport(
                candidate_count=0,
                new_result_count=0,
                reused_result_count=0,
            ),
            selection_report=SearchResultSelectionReport(
                candidate_ids=[],
                selected_ids=[],
                deferred_ids=[],
            ),
            fetch_outcomes=[],
            extraction_outcomes=[],
            search_result_index_updates={},
            document_index_updates={},
        )

    @model_validator(mode="after")
    def validate_failed_retrieval_has_no_artifacts(self) -> "ResearchObservation":
        if self.retrieval_failure is None:
            return self
        if (
            self.search_result_ids
            or self.retrieval_filter_report.input_count
            or self.retrieval_filter_report.output_count
            or self.source_classification_report.input_count
            or self.source_classification_report.recognized_count
            or self.source_classification_report.unknown_count
            or self.source_classification_report.classifications
            or self.search_result_merge_report.candidate_count
            or self.search_result_merge_report.new_result_count
            or self.search_result_merge_report.reused_result_count
            or self.selection_report.candidate_ids
            or self.selection_report.selected_ids
            or self.selection_report.deferred_ids
            or self.fetch_outcomes
            or self.extraction_outcomes
            or self.search_result_index_updates
            or self.document_index_updates
        ):
            raise ValueError("failed retrieval cannot contain artifacts or outcomes")
        return self


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
    ResearchObservation
    | CurateEvidenceObservation
    | FinishObservation
    | StopObservation,
    Field(discriminator="type"),
]

_OBSERVATION_ADAPTER = TypeAdapter(Observation)


def validate_observation(value: object) -> Observation:
    """Parse an observation value using its action type discriminator."""

    return _OBSERVATION_ADAPTER.validate_python(value)

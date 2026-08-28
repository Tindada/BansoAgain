"""Research context derived from agent state and stored artifacts."""

from collections import Counter
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from banso.artifacts.store import ArtifactStore
from banso.agent.action import RetrievalRoute
from banso.agent.observation import (
    CompletedResearchObservation,
    CompletedSearchObservation,
    ExtractionFailure,
    ExtractionSuccess,
    FailedResearchObservation,
    FailedSearchObservation,
    FetchFailure,
    ResearchObservation,
    ResearchObservationBase,
    SearchObservation,
    SearchObservationBase,
)
from banso.agent.state import ActionHistoryEntry, AgentState
from banso.documents.models import Document, DocumentEvidence
from banso.retrieval.models import SearchResult
from banso.retrieval.url_utils import publisher_domain
from banso.source import Source, SourceType


class SourceView(BaseModel):
    """Compact source identity exposed in the research context."""

    name: str | None = None
    domain: str
    type: SourceType


class EvidenceGroup(BaseModel):
    """Visible evidence text for one source document."""

    document_ref: str
    query_refs: list[str]
    document_title: str
    source: SourceView
    published_at: datetime | None = None
    evidence_preview: str | None = None
    evidence_truncated: bool


class QueryHistoryItemBase(BaseModel):
    """Facts shared by research and search history items."""

    query_ref: str
    query: str
    route: RetrievalRoute
    source_domains: list[str] | None = None


class FetchFailureSource(BaseModel):
    """Aggregated fetch failures for one publisher and failure kind."""

    domain: str
    reason: str
    status_code: int | None = None
    count: int = Field(ge=1)


class CompletedQueryHistoryItem(QueryHistoryItemBase):
    """Compact retrieval and optional read outcomes from a completed query."""

    status: Literal["completed"] = "completed"
    retrieved_results: int
    new_results: int
    reused_results: int
    selected_results: int | None = None
    fetch_successes: int | None = None
    fetch_failures: int | None = None
    fetch_failure_sources: list[FetchFailureSource] | None = None
    evidence_documents: int | None = None
    no_evidence_documents: int | None = None
    extraction_failures: int | None = None


class FailedQueryHistoryItem(QueryHistoryItemBase):
    """Trace-safe diagnostics from a failed query action."""

    status: Literal["failed"] = "failed"
    stage: Literal["retrieval", "selection"]
    reason: str
    status_code: int | None = None
    retryable: bool
    attempt_count: int


ResearchHistoryItem = CompletedQueryHistoryItem | FailedQueryHistoryItem
QueryObservation = ResearchObservation | SearchObservation
ReferencedQuery = tuple[str, ActionHistoryEntry, QueryObservation]


def _referenced_queries(state: AgentState) -> list[ReferencedQuery]:
    entries = [
        entry
        for entry in state.action_history
        if isinstance(
            entry.observation,
            (ResearchObservationBase, SearchObservationBase),
        )
    ]
    return [
        (f"Q{index}", entry, entry.observation)
        for index, entry in enumerate(entries, start=1)
    ]


def build_research_history(state: AgentState) -> list[ResearchHistoryItem]:
    """Build compact research and search outcomes from agent history."""
    return _build_research_history(_referenced_queries(state))


def _build_research_history(
    referenced_queries: list[ReferencedQuery],
) -> list[ResearchHistoryItem]:
    return [
        _build_research_history_item(query_ref, entry, observation)
        for query_ref, entry, observation in referenced_queries
    ]


def _build_query_refs(
    state: AgentState,
    referenced_queries: list[ReferencedQuery],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    result_query_refs: dict[str, list[str]] = {}
    document_query_refs: dict[str, list[str]] = {}
    for query_ref, _, observation in referenced_queries:
        if isinstance(observation, CompletedResearchObservation):
            search = observation.search
        elif isinstance(observation, CompletedSearchObservation):
            search = observation
        else:
            continue
        for result_id in search.search_result_ids:
            query_refs = result_query_refs.setdefault(result_id, [])
            if query_ref not in query_refs:
                query_refs.append(query_ref)
            document_id = state.search_results[result_id].document_id
            if document_id is not None:
                document_refs = document_query_refs.setdefault(document_id, [])
                if query_ref not in document_refs:
                    document_refs.append(query_ref)
    return result_query_refs, document_query_refs


def _build_research_history_item(
    query_ref: str,
    entry: ActionHistoryEntry,
    observation: QueryObservation,
) -> ResearchHistoryItem:
    if isinstance(observation, ResearchObservationBase):
        query = observation.query
        source_domains = observation.source_domains
    else:
        query = str(entry.action.params["query"])
        source_domains = entry.action.params.get("source_domains")
    if isinstance(
        observation,
        (FailedSearchObservation, FailedResearchObservation),
    ):
        return FailedQueryHistoryItem(
            query_ref=query_ref,
            query=query,
            route=observation.route,
            source_domains=source_domains,
            stage=(
                observation.stage
                if isinstance(observation, FailedResearchObservation)
                else "retrieval"
            ),
            reason=observation.reason,
            status_code=observation.status_code,
            retryable=observation.retryable,
            attempt_count=observation.attempt_count,
        )
    if isinstance(observation, CompletedSearchObservation):
        return CompletedQueryHistoryItem(
            query_ref=query_ref,
            query=query,
            route=observation.route,
            source_domains=source_domains,
            retrieved_results=len(observation.search_result_ids),
            new_results=observation.search_result_merge_report.new_result_count,
            reused_results=observation.search_result_merge_report.reused_result_count,
        )
    search = observation.search
    read = observation.read
    fetch_failures = [
        outcome
        for outcome in read.fetch_outcomes
        if isinstance(outcome, FetchFailure)
    ]
    extraction_failures = [
        outcome
        for outcome in read.extraction_outcomes
        if isinstance(outcome, ExtractionFailure)
    ]
    fetch_failure_counts = Counter(
        (
            publisher_domain(outcome.failure.url),
            outcome.failure.reason,
            outcome.failure.status_code,
        )
        for outcome in fetch_failures
    )
    evidence_documents = sum(
        isinstance(outcome, ExtractionSuccess) and outcome.evidence_id is not None
        for outcome in read.extraction_outcomes
    )
    no_evidence_documents = sum(
        isinstance(outcome, ExtractionSuccess) and outcome.evidence_id is None
        for outcome in read.extraction_outcomes
    )
    return CompletedQueryHistoryItem(
        query_ref=query_ref,
        query=query,
        route=search.route,
        source_domains=source_domains,
        retrieved_results=len(search.search_result_ids),
        new_results=search.search_result_merge_report.new_result_count,
        reused_results=search.search_result_merge_report.reused_result_count,
        selected_results=len(observation.selection_report.selected_ids),
        fetch_successes=len(read.fetch_outcomes) - len(fetch_failures),
        fetch_failures=len(fetch_failures),
        fetch_failure_sources=[
            FetchFailureSource(
                domain=domain,
                reason=reason,
                status_code=status_code,
                count=count,
            )
            for (domain, reason, status_code), count in sorted(
                fetch_failure_counts.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2] if item[0][2] is not None else -1,
                ),
            )
        ],
        evidence_documents=evidence_documents,
        no_evidence_documents=no_evidence_documents,
        extraction_failures=len(extraction_failures),
    )


class BudgetSummary(BaseModel):
    """Remaining execution capacity relevant to the next action."""

    remaining_steps: int
    remaining_researches: int
    max_results_per_research: int


class ArtifactSummary(BaseModel):
    """Collected artifact totals independent of context display limits."""

    search_result_count: int
    document_count: int
    evidence_document_count: int
    no_evidence_document_count: int


class UserQueryView(BaseModel):
    """User query fields relevant to the research process."""

    text: str
    region: str | None = None
    time_range: str | None = None


class CandidateResult(BaseModel):
    """Unread search result available to a later read action."""

    candidate_ref: str
    query_refs: list[str]
    title: str
    url: str
    snippet: str | None = None
    source: SourceView
    published_at: datetime | None = None


class ResearchContext(BaseModel):
    """Compact facts describing the current research run."""

    user_query: UserQueryView
    reference_time: datetime
    notes: str
    enabled_routes: list[RetrievalRoute]
    budget: BudgetSummary
    research_history: list[ResearchHistoryItem]
    artifacts: ArtifactSummary
    candidate_results: list[CandidateResult]
    evidence_groups: list[EvidenceGroup]


def document_reference_maps(state: AgentState) -> tuple[dict[str, str], dict[str, str]]:
    """Return stable rollout-local mappings between document IDs and LLM refs."""
    id_to_ref = {
        document_id: f"D{index}"
        for index, document_id in enumerate(state.documents, start=1)
    }
    return id_to_ref, {
        document_ref: document_id
        for document_id, document_ref in id_to_ref.items()
    }


def search_result_reference_maps(
    state: AgentState,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return stable rollout-local mappings between search result IDs and refs."""
    id_to_ref = {
        result_id: f"C{index}"
        for index, result_id in enumerate(state.search_results, start=1)
    }
    return id_to_ref, {
        result_ref: result_id
        for result_id, result_ref in id_to_ref.items()
    }


class ResearchContextBuilder:
    """Build deterministic research context from state and artifacts."""

    def __init__(
        self,
        store: ArtifactStore,
        enabled_routes: list[RetrievalRoute],
        *,
        max_evidence_preview_chars: int = 3000,
    ) -> None:
        if not enabled_routes:
            raise ValueError("enabled_routes must contain at least one route")
        if len(set(enabled_routes)) != len(enabled_routes):
            raise ValueError("enabled_routes must be unique")
        if max_evidence_preview_chars < 0:
            raise ValueError("max_evidence_preview_chars must be non-negative")

        self.store = store
        self.enabled_routes = list(enabled_routes)
        self.max_evidence_preview_chars = max_evidence_preview_chars

    def build(self, state: AgentState) -> ResearchContext:
        """Resolve state facts and artifacts into bounded research context."""
        evidence_document_ids = [
            document_id
            for document_id, document in state.documents.items()
            if document.evidence_id is not None
        ]
        documents = {
            document_id: self._load_artifact(document_id, Document)
            for document_id in evidence_document_ids
        }

        id_to_ref, _ = document_reference_maps(state)
        result_id_to_ref, _ = search_result_reference_maps(state)
        evidence_count = state.evidence_document_count
        referenced_queries = _referenced_queries(state)
        research_history = _build_research_history(referenced_queries)
        result_query_refs, document_query_refs = _build_query_refs(
            state,
            referenced_queries,
        )

        return ResearchContext(
            user_query=UserQueryView(
                text=state.query.text,
                region=state.query.region,
                time_range=state.query.time_range,
            ),
            reference_time=state.reference_time,
            notes=state.notes,
            enabled_routes=list(self.enabled_routes),
            budget=BudgetSummary(
                remaining_steps=state.remaining_steps,
                remaining_researches=state.remaining_research_capacity,
                max_results_per_research=state.budget.max_results_per_research,
            ),
            research_history=research_history,
            artifacts=ArtifactSummary(
                search_result_count=len(state.search_results),
                document_count=len(state.documents),
                evidence_document_count=evidence_count,
                no_evidence_document_count=len(state.documents) - evidence_count,
            ),
            candidate_results=[
                self._build_candidate_result(
                    result_id_to_ref[result_id],
                    result_query_refs.get(result_id, []),
                    self._load_artifact(result_id, SearchResult),
                )
                for result_id, result_state in state.search_results.items()
                if result_state.document_id is None and result_state.failure is None
            ],
            evidence_groups=[
                self._build_evidence_group(
                    id_to_ref[document_id],
                    document_query_refs.get(document_id, []),
                    documents[document_id],
                    state.documents[document_id].evidence_id,
                )
                for document_id in evidence_document_ids
            ],
        )

    def _build_evidence_group(
        self,
        document_ref: str,
        query_refs: list[str],
        document: Document,
        evidence_id: str | None,
    ) -> EvidenceGroup:
        evidence = (
            self.store.get(evidence_id, DocumentEvidence)
            if evidence_id is not None
            else None
        )
        evidence_preview = (
            evidence.text[: self.max_evidence_preview_chars]
            if evidence is not None
            else None
        )
        return EvidenceGroup(
            document_ref=document_ref,
            query_refs=query_refs,
            document_title=document.title,
            source=self._build_source(document.source, document.url),
            published_at=document.published_at,
            evidence_preview=evidence_preview,
            evidence_truncated=(
                evidence is not None
                and len(evidence.text) > self.max_evidence_preview_chars
            ),
        )

    def _build_candidate_result(
        self,
        candidate_ref: str,
        query_refs: list[str],
        result: SearchResult,
    ) -> CandidateResult:
        return CandidateResult(
            candidate_ref=candidate_ref,
            query_refs=query_refs,
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            source=self._build_source(result.source, result.url),
            published_at=result.published_at,
        )

    @staticmethod
    def _build_source(source: Source | None, item_url: str) -> SourceView:
        domain = publisher_domain(item_url)
        return SourceView(
            name=source.name if source is not None else domain or None,
            domain=domain,
            type=source.type if source is not None else SourceType.UNKNOWN,
        )

    def _load_artifact[T](self, artifact_id: str, artifact_type: type[T]) -> T:
        artifact = self.store.get(artifact_id, artifact_type)
        if artifact is None:
            raise ValueError(
                f"{artifact_type.__name__} artifact is missing or has the wrong type: "
                f"{artifact_id}"
            )
        return artifact

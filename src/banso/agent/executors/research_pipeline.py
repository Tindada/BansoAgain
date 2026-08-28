"""Composite research execution used by the news action executor."""

from collections.abc import Mapping
from dataclasses import dataclass

from banso.artifacts.store import ArtifactStore
from banso.agent.action import ResearchActionParams, RetrievalRoute
from banso.agent.executors.read import execute_read
from banso.agent.executors.retry import RetryPolicy, run_with_retry
from banso.agent.executors.search import SearchFailure, execute_search
from banso.agent.observation import (
    CompletedResearchObservation,
    FailedResearchObservation,
    ResearchObservation,
)
from banso.agent.selection.selector import (
    SearchResultSelection,
    SearchResultSelectionError,
    SearchResultSelectionRequest,
    SearchResultSelector,
)
from banso.agent.state import AgentState
from banso.documents.extractor import EvidenceExtractor
from banso.documents.fetcher import DocumentFetcher
from banso.retrieval.filter import RetrievalFilter
from banso.retrieval.models import SearchResult, SearchResultSelectionReport
from banso.retrieval.provider import RetrievalProvider, SearchRequest
from banso.retrieval.source_classifier import SourceClassifier
from banso.tracing.trace import start_span


@dataclass(frozen=True)
class ResearchRouteComponents:
    """Retrieval and document-fetch components for one semantic route."""

    retrieval_provider: RetrievalProvider
    document_fetcher: DocumentFetcher


async def execute_research(
    params: ResearchActionParams,
    state: AgentState,
    *,
    store: ArtifactStore,
    research_routes: Mapping[RetrievalRoute, ResearchRouteComponents],
    evidence_extractor: EvidenceExtractor,
    retrieval_filter: RetrievalFilter,
    source_classifier: SourceClassifier,
    search_result_selector: SearchResultSelector,
    max_extraction_concurrency: int,
    retry_policy: RetryPolicy,
) -> ResearchObservation:
    """Execute search, selection, and reading as one research action."""
    components = research_routes.get(params.route)
    if components is None:
        raise ValueError(f"research route is not enabled: {params.route.value}")

    search_outcome = await execute_search(
        SearchRequest(
            query=params.query,
            language=state.query.language,
            region=state.query.region,
            time_range=state.query.time_range,
            source_domains=params.source_domains,
        ),
        state,
        store=store,
        retrieval_provider=components.retrieval_provider,
        retrieval_filter=retrieval_filter,
        source_classifier=source_classifier,
        retry_policy=retry_policy,
        route=params.route,
    )
    if isinstance(search_outcome, SearchFailure):
        return FailedResearchObservation(
            query=params.query,
            route=params.route,
            source_domains=params.source_domains,
            stage="retrieval",
            provider=search_outcome.provider,
            reason=search_outcome.reason,
            status_code=search_outcome.status_code,
            message=search_outcome.message,
            source_error_type=search_outcome.source_error_type,
            retryable=search_outcome.retryable,
            attempt_count=search_outcome.attempt_count,
        )

    selection = await _select_results(
        params,
        state,
        search_outcome.search_result_ids,
        store=store,
        search_result_selector=search_result_selector,
        retry_policy=retry_policy,
    )
    if isinstance(selection, FailedResearchObservation):
        return selection
    selected_results, selection_report = selection

    read_result = await execute_read(
        selected_results,
        state,
        store=store,
        document_fetcher=components.document_fetcher,
        evidence_extractor=evidence_extractor,
        ignored_query_params=retrieval_filter.config.ignored_query_params,
        limit=state.budget.max_results_per_research,
        max_extraction_concurrency=max_extraction_concurrency,
        retry_policy=retry_policy,
        route=params.route,
    )

    return CompletedResearchObservation(
        query=params.query,
        route=params.route,
        source_domains=params.source_domains,
        search_result_ids=search_outcome.search_result_ids,
        retrieval_filter_report=search_outcome.retrieval_filter_report,
        source_classification_report=search_outcome.source_classification_report,
        search_result_merge_report=search_outcome.search_result_merge_report,
        selection_report=selection_report,
        fetch_outcomes=read_result.fetch_outcomes,
        extraction_outcomes=read_result.extraction_outcomes,
        search_result_index_updates=search_outcome.search_result_index_updates,
        document_index_updates=read_result.document_index_updates,
    )


async def _select_results(
    params: ResearchActionParams,
    state: AgentState,
    search_result_ids: list[str],
    *,
    store: ArtifactStore,
    search_result_selector: SearchResultSelector,
    retry_policy: RetryPolicy,
) -> tuple[list[SearchResult], SearchResultSelectionReport] | FailedResearchObservation:
    with start_span(
        "news.research.select",
        attributes={"route": params.route.value},
    ) as span:
        candidates = _load_selectable_candidates(
            search_result_ids,
            state,
            store=store,
        )
        if candidates:
            selection_request = SearchResultSelectionRequest(
                research_query=params.query,
                candidates=candidates,
                state=state,
            )
            selection_attempt = await run_with_retry(
                lambda: search_result_selector.select(selection_request),
                error_type=Exception,
                is_retryable=lambda error: isinstance(
                    error,
                    SearchResultSelectionError,
                ),
                policy=retry_policy,
            )
            if selection_attempt.error is not None:
                error = selection_attempt.error
                failure = FailedResearchObservation(
                    query=params.query,
                    route=params.route,
                    source_domains=params.source_domains,
                    stage="selection",
                    reason=(
                        "invalid_response"
                        if isinstance(error, SearchResultSelectionError)
                        else "unexpected_error"
                    ),
                    message=str(error),
                    source_error_type=type(error).__name__,
                    retryable=isinstance(error, SearchResultSelectionError),
                    attempt_count=selection_attempt.attempt_count,
                )
                span.set_attribute("outcome", "failure")
                span.set_output({"research_failure": failure})
                return failure
            if selection_attempt.value is None:
                raise AssertionError("successful selection returned no result")
            selected_results = _resolve_selection(
                selection_attempt.value,
                candidates,
            )
        else:
            selected_results = []
        selection_report = SearchResultSelectionReport(
            candidate_ids=[result.id for result in candidates],
            selected_ids=[result.id for result in selected_results],
        )
        span.set_attribute("outcome", "success")
        span.set_output({"selection_report": selection_report})
        return selected_results, selection_report


def _load_selectable_candidates(
    search_result_ids: list[str],
    state: AgentState,
    *,
    store: ArtifactStore,
) -> list[SearchResult]:
    candidates: list[SearchResult] = []
    for result_id in search_result_ids:
        result_state = state.search_results.get(result_id)
        if result_state is not None and (
            result_state.document_id is not None or result_state.failure is not None
        ):
            continue
        result = store.get(result_id, SearchResult)
        if result is None:
            raise ValueError(
                "SearchResult artifact is missing or has the wrong type: "
                f"{result_id}"
            )
        candidates.append(result)
    return candidates


def _resolve_selection(
    selection: SearchResultSelection,
    candidates: list[SearchResult],
) -> list[SearchResult]:
    selected_id_set = set(selection.selected_ids)
    unknown_ids = selected_id_set - {candidate.id for candidate in candidates}
    if unknown_ids:
        raise ValueError(
            "search result selector returned unknown IDs: "
            + ", ".join(sorted(unknown_ids))
        )
    return [candidate for candidate in candidates if candidate.id in selected_id_set]

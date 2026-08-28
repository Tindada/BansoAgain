"""Search execution helpers used by the news action executor."""

from banso.artifacts.store import ArtifactStore
from banso.agent.action import RetrievalRoute
from banso.agent.executors.retry import RetryPolicy, run_with_retry
from banso.agent.observation import (
    CompletedSearchObservation,
    FailedSearchObservation,
    SearchObservation,
)
from banso.agent.state import AgentState
from banso.retrieval.filter import RetrievalFilter
from banso.retrieval.models import SearchResult, SearchResultMergeReport
from banso.retrieval.provider import (
    RetrievalError,
    RetrievalProvider,
    SearchRequest,
)
from banso.retrieval.source_classifier import SourceClassifier
from banso.retrieval.url_utils import normalize_url
from banso.tracing.trace import start_span


async def execute_search(
    request: SearchRequest,
    state: AgentState,
    *,
    store: ArtifactStore,
    retrieval_provider: RetrievalProvider,
    retrieval_filter: RetrievalFilter,
    source_classifier: SourceClassifier,
    retry_policy: RetryPolicy,
    route: RetrievalRoute,
) -> SearchObservation:
    """Retrieve, normalize, classify, deduplicate, and store search results."""
    with start_span(
        "news.search.retrieve",
        input=request.model_dump(
            mode="json",
            include={"query", "source_domains"},
            exclude_none=True,
        ),
        attributes={"route": route.value},
    ) as span:
        attempt = await run_with_retry(
            lambda: retrieval_provider.search(request),
            error_type=RetrievalError,
            is_retryable=lambda error: error.retryable,
            policy=retry_policy,
        )
        if attempt.error is not None:
            error = attempt.error
            failure = FailedSearchObservation(
                route=route,
                provider=error.provider,
                reason=error.reason,
                status_code=error.status_code,
                message=error.message,
                source_error_type=error.source_error_type,
                retryable=error.retryable,
                attempt_count=attempt.attempt_count,
            )
            span.set_attribute("outcome", "failure")
            span.set_output({"search_failure": failure})
            return failure
        if attempt.value is None:
            raise AssertionError("successful retrieval returned no result")

        success = _process_results(
            attempt.value,
            state,
            store=store,
            retrieval_filter=retrieval_filter,
            source_classifier=source_classifier,
            route=route,
        )
        span.set_attribute("outcome", "success")
        span.set_output(
            {
                "search_result_ids": success.search_result_ids,
                "filter_report": success.retrieval_filter_report,
                "classification_report": success.source_classification_report,
                "merge_report": success.search_result_merge_report,
            }
        )
        return success


def _process_results(
    raw_results: list[SearchResult],
    state: AgentState,
    *,
    store: ArtifactStore,
    retrieval_filter: RetrievalFilter,
    source_classifier: SourceClassifier,
    route: RetrievalRoute,
) -> CompletedSearchObservation:
    filtered = retrieval_filter.apply(raw_results)
    classified = source_classifier.apply(filtered.results)
    index_updates: dict[str, str] = {}
    result_ids: list[str] = []
    new_result_count = 0
    reused_result_count = 0

    for result in classified.results:
        normalized_url = normalize_url(
            result.url,
            ignored_query_params=retrieval_filter.config.ignored_query_params,
        )
        existing_result_id = state.search_result_index.get(normalized_url)
        if existing_result_id is not None:
            result_ids.append(existing_result_id)
            reused_result_count += 1
            continue

        result_id = store.put(result)
        index_updates[normalized_url] = result_id
        result_ids.append(result_id)
        new_result_count += 1

    return CompletedSearchObservation(
        route=route,
        search_result_ids=result_ids,
        search_result_index_updates=index_updates,
        search_result_merge_report=SearchResultMergeReport(
            candidate_count=len(classified.results),
            new_result_count=new_result_count,
            reused_result_count=reused_result_count,
        ),
        retrieval_filter_report=filtered.report,
        source_classification_report=classified.report,
    )

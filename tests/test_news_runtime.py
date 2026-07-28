"""Smoke test for the basic news runtime workflow."""

import asyncio

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentRuntime,
    AgentState,
    DefaultStateReducer,
    ExecutionBudget,
    PlannedSearch,
    RuntimeExecutionError,
    SearchPlan,
    UserQuery,
)
from banso.core.action import AgentAction, AgentActionType
from banso.core.observation import (
    FetchDocumentsObservation,
    FinishObservation,
    PlanSearchObservation,
    SearchObservation,
)
from banso.core.state import SearchResultState
from banso.documents import (
    Document,
    DocumentFetcher,
    DocumentFetchError,
    FakeDocumentFetcher,
    FakeEvidenceExtractor,
)
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import (
    FakeRetrievalProvider,
    RetrievalProvider,
    SearchPlanningRequest,
    SearchQueryPlanner,
    SearchRequest,
    SearchResult,
    Source,
    SourceType,
)
from banso.synthesis import FakeSynthesizer
from banso.tracing import InMemoryTraceSink, Tracer


class DuplicateRetrievalProvider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        source = Source(name="Example News", type=SourceType.NEWS)
        return [
            SearchResult(
                title=f"First result for {request.query}",
                url="https://example.com/news?a=1&utm_source=test",
                rank=1,
                source=source,
            ),
            SearchResult(
                title=f"Duplicate result for {request.query}",
                url="https://example.com/news?utm_medium=test&a=1",
                rank=2,
                source=source,
            ),
            SearchResult(
                title=f"Second result for {request.query}",
                url="https://example.com/second",
                rank=3,
                source=source,
            ),
        ]


class CrossSearchDuplicateRetrievalProvider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        if request.query.startswith("first"):
            return [
                SearchResult(
                    title="First search title",
                    url="https://example.com/news?a=1&utm_source=first",
                    snippet="First search snippet",
                    rank=1,
                )
            ]
        return [
            SearchResult(
                title="Second search title",
                url="https://EXAMPLE.com/news?utm_medium=second&a=1#details",
                snippet="Second search snippet",
                rank=3,
            ),
            SearchResult(
                title="New result",
                url="https://example.com/new",
                snippet="New result snippet",
                rank=1,
            ),
        ]


class PartiallyBlockedDocumentFetcher(FakeDocumentFetcher):
    def __init__(self, status_code: int = 403) -> None:
        self.status_code = status_code

    async def fetch(self, request):
        if request.url.endswith("blocked"):
            raise DocumentFetchError(
                url=request.url,
                reason="http_status",
                message=f"HTTP {self.status_code} while fetching document",
                status_code=self.status_code,
                source_error_type="HTTPStatusError",
            )
        return await super().fetch(request)


class BlockedDocumentFetcher(FakeDocumentFetcher):
    async def fetch(self, request):
        raise DocumentFetchError(
            url=request.url,
            reason="http_status",
            message="HTTP 503 while fetching document",
            status_code=503,
            source_error_type="HTTPStatusError",
        )


class PartiallyBlockedRetrievalProvider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        source = Source(name="Example News", type=SourceType.NEWS)
        return [
            SearchResult(
                title="Blocked",
                url="https://example.com/blocked",
                rank=1,
                source=source,
            ),
            SearchResult(
                title="Fetchable",
                url="https://example.com/fetchable",
                rank=2,
                source=source,
            ),
        ]


class TwoQueryPlanner:
    async def plan(self, request: SearchPlanningRequest) -> SearchPlan:
        return SearchPlan(
            searches=[
                PlannedSearch(query="first unknown search"),
                PlannedSearch(query="second trusted search"),
            ][: request.max_searches]
        )


class MixedTrustRetrievalProvider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        if request.query.startswith("first"):
            return [
                SearchResult(
                    title=f"Early unknown {index}",
                    url=f"https://early-{index}.example/report",
                    rank=index,
                    metadata={"provider": "tavily", "score": score},
                )
                for index, score in enumerate([0.9, 0.8, 0.7], start=1)
            ]
        return [
            SearchResult(
                title="Later official release",
                url="https://openai.com/index/release",
                rank=1,
                metadata={"provider": "tavily", "score": 0.4},
            ),
            SearchResult(
                title="Later research paper",
                url="https://arxiv.org/abs/2607.00001",
                rank=2,
                metadata={"provider": "tavily", "score": 0.3},
            ),
            *[
                SearchResult(
                    title=f"Later unknown {index}",
                    url=f"https://later-{index}.example/report",
                    rank=index + 2,
                    metadata={"provider": "tavily", "score": score},
                )
                for index, score in enumerate([0.95, 0.85, 0.75], start=1)
            ],
        ]


def _news_executor(
    store: InMemoryArtifactStore,
    *,
    retrieval_provider: RetrievalProvider | None = None,
    document_fetcher: DocumentFetcher | None = None,
    search_query_planner: SearchQueryPlanner | None = None,
) -> NewsActionExecutor:
    return NewsActionExecutor(
        store=store,
        retrieval_provider=(
            retrieval_provider
            if retrieval_provider is not None
            else FakeRetrievalProvider()
        ),
        document_fetcher=(
            document_fetcher
            if document_fetcher is not None
            else FakeDocumentFetcher()
        ),
        evidence_extractor=FakeEvidenceExtractor(),
        synthesizer=FakeSynthesizer(),
        search_query_planner=search_query_planner,
    )


def _news_runtime(
    store: InMemoryArtifactStore,
    *,
    retrieval_provider: RetrievalProvider | None = None,
    document_fetcher: DocumentFetcher | None = None,
    search_query_planner: SearchQueryPlanner | None = None,
    tracer: Tracer | None = None,
) -> AgentRuntime:
    return AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=_news_executor(
            store,
            retrieval_provider=retrieval_provider,
            document_fetcher=document_fetcher,
            search_query_planner=search_query_planner,
        ),
        tracer=tracer,
    )


async def _run_news_runtime() -> None:
    store = InMemoryArtifactStore()
    trace_sink = InMemoryTraceSink()
    runtime = _news_runtime(
        store,
        tracer=Tracer(trace_sink),
    )

    output = await runtime.run(AgentState(query=UserQuery(text="latest AI news")))
    state = output.result.state
    spans = trace_sink.get_trace(output.trace_id)

    assert state.done is True
    assert [entry.action.type for entry in state.action_history] == [
        AgentActionType.PLAN_SEARCH,
        AgentActionType.SEARCH,
        AgentActionType.FETCH_DOCUMENTS,
        AgentActionType.EXTRACT_EVIDENCE,
        AgentActionType.FINISH,
    ]
    assert state.search_plan is not None
    assert state.search_plan.model_dump() == {
        "searches": [{"query": "latest AI news", "intent": "general"}]
    }
    plan_observation = state.action_history[0].observation
    assert isinstance(plan_observation, PlanSearchObservation)
    assert plan_observation.search_plan == state.search_plan
    search_observation = state.action_history[1].observation
    assert isinstance(search_observation, SearchObservation)
    assert search_observation.search_queries == ["latest AI news"]
    assert len(state.search_results) == 1
    assert len(state.documents) == 1
    assert sum(
        len(document.evidence_ids) for document in state.documents.values()
    ) == 1
    assert state.final_answer is not None
    assert "Fake summary for 'latest AI news'" in state.final_answer
    finish_observation = state.action_history[4].observation
    assert isinstance(finish_observation, FinishObservation)
    assert state.citations == finish_observation.citations
    assert "final_answer" not in output.result.model_dump()
    run_span = next(span for span in spans if span.name == "agent.run")
    assert run_span.output == {
        "result": output.result.model_dump(mode="json"),
    }
    assert run_span.status == "ok"
    assert all(
        span.duration_seconds >= 0
        for span in spans
        if span.name
        in {
            "agent.policy.select",
            "agent.action.execute",
            "agent.state.reduce",
        }
    )


async def _run_news_runtime_filters_search_results() -> None:
    store = InMemoryArtifactStore()
    runtime = _news_runtime(
        store,
        retrieval_provider=DuplicateRetrievalProvider(),
    )

    output = await runtime.run(AgentState(query=UserQuery(text="latest AI news")))
    state = output.result.state
    search_observation = output.result.state.action_history[1].observation

    assert len(state.search_results) == 2
    assert len(state.documents) == 2
    assert isinstance(search_observation, SearchObservation)
    assert search_observation.retrieval_filter_report.model_dump() == {
        "input_count": 3,
        "output_count": 2,
        "dropped_empty_title": 0,
        "dropped_empty_url": 0,
        "dropped_invalid_url": 0,
        "dropped_duplicate_url": 1,
        "truncated_count": 0,
    }
    classification_report = search_observation.source_classification_report
    assert classification_report.input_count == 2
    assert classification_report.recognized_count == 2
    assert classification_report.unknown_count == 0


async def _run_news_runtime_respects_document_fetch_budget() -> None:
    store = InMemoryArtifactStore()
    runtime = _news_runtime(
        store,
        retrieval_provider=DuplicateRetrievalProvider(),
    )

    output = await runtime.run(
        AgentState(
            query=UserQuery(text="latest AI news"),
            budget=ExecutionBudget(max_document_fetches=1),
        )
    )

    assert len(output.result.state.search_results) == 2
    assert len(output.result.state.documents) == 1


async def _run_news_runtime_preserves_search_order_when_fetching() -> None:
    store = InMemoryArtifactStore()
    runtime = _news_runtime(
        store,
        retrieval_provider=MixedTrustRetrievalProvider(),
        search_query_planner=TwoQueryPlanner(),
    )

    output = await runtime.run(
        AgentState(
            query=UserQuery(text="latest AI news"),
            budget=ExecutionBudget(max_searches=2, max_document_fetches=8),
        )
    )
    results = [
        store.get(result_id, SearchResult)
        for result_id in output.result.state.search_results
    ]
    documents = [
        store.get(document_id, Document)
        for document_id in output.result.state.documents
    ]

    assert len(output.result.state.search_results) == 8
    assert len(output.result.state.documents) == 8
    assert [result.title for result in results if result][:4] == [
        "Early unknown 1",
        "Early unknown 2",
        "Early unknown 3",
        "Later official release",
    ]
    assert [document.title for document in documents if document][:4] == [
        "Early unknown 1",
        "Early unknown 2",
        "Early unknown 3",
        "Later official release",
    ]


async def _run_news_runtime_deduplicates_across_searches() -> None:
    store = InMemoryArtifactStore()
    runtime = _news_runtime(
        store,
        retrieval_provider=CrossSearchDuplicateRetrievalProvider(),
        search_query_planner=TwoQueryPlanner(),
    )

    output = await runtime.run(
        AgentState(
            query=UserQuery(text="latest AI news"),
            budget=ExecutionBudget(max_searches=2, max_document_fetches=8),
        )
    )
    state = output.result.state
    search_observations = [
        entry.observation
        for entry in state.action_history
        if entry.action.type == AgentActionType.SEARCH
    ]
    stored_results = store.list(SearchResult)
    stored_documents = store.list(Document)
    result_ids = list(state.search_results)

    assert len(search_observations) == 2
    first_search = search_observations[0]
    second_search = search_observations[1]
    assert isinstance(first_search, SearchObservation)
    assert isinstance(second_search, SearchObservation)
    assert first_search.search_result_ids == [
        result_ids[0]
    ]
    assert second_search.search_result_ids == [
        result_ids[0],
        result_ids[1],
    ]
    assert first_search.search_result_merge_report.model_dump() == {
        "candidate_count": 1,
        "new_result_count": 1,
        "reused_result_count": 0,
    }
    assert second_search.search_result_merge_report.model_dump() == {
        "candidate_count": 2,
        "new_result_count": 1,
        "reused_result_count": 1,
    }
    assert len(state.search_results) == 2
    assert state.search_result_index == {
        "https://example.com/news?a=1": result_ids[0],
        "https://example.com/new": result_ids[1],
    }
    assert len(stored_results) == 2
    assert stored_results[0].title == "First search title"
    assert stored_results[0].snippet == "First search snippet"
    assert all(result.snippet != "Second search snippet" for result in stored_results)
    assert len(state.documents) == 2
    assert len(stored_documents) == 2


async def _run_news_runtime_skips_unfetchable_document(status_code: int) -> None:
    store = InMemoryArtifactStore()
    runtime = _news_runtime(
        store,
        retrieval_provider=PartiallyBlockedRetrievalProvider(),
        document_fetcher=PartiallyBlockedDocumentFetcher(status_code),
    )

    output = await runtime.run(AgentState(query=UserQuery(text="latest AI news")))
    fetch_observation = output.result.state.action_history[2].observation
    result_ids = list(output.result.state.search_results)
    document_id = next(iter(output.result.state.documents))
    failed_result_id = result_ids[0]

    assert output.result.state.done is True
    assert len(output.result.state.documents) == 1
    assert isinstance(fetch_observation, FetchDocumentsObservation)
    assert [
        outcome.model_dump(mode="json")
        for outcome in fetch_observation.fetch_outcomes
    ] == [
        {
            "status": "failure",
            "search_result_id": failed_result_id,
            "failure": {
                "url": "https://example.com/blocked",
                "status_code": status_code,
                "reason": "http_status",
                "retryable": 500 <= status_code < 600,
                "message": f"HTTP {status_code} while fetching document",
                "source_error_type": "HTTPStatusError",
            },
        },
        {
            "status": "success",
            "search_result_id": result_ids[1],
            "document_id": document_id,
        },
    ]
    failed_fetch = output.result.state.search_results[failed_result_id]
    assert failed_fetch.failure is not None
    assert (
        failed_fetch.failure.retryable
        is (status_code == 503)
    )
    assert failed_fetch.attempt_count == (
        2 if status_code == 503 else 1
    )


def test_news_runtime() -> None:
    asyncio.run(_run_news_runtime())


def test_news_runtime_filters_search_results() -> None:
    asyncio.run(_run_news_runtime_filters_search_results())


def test_news_runtime_respects_document_fetch_budget() -> None:
    asyncio.run(_run_news_runtime_respects_document_fetch_budget())


def test_news_runtime_preserves_search_order_when_fetching() -> None:
    asyncio.run(_run_news_runtime_preserves_search_order_when_fetching())


def test_news_runtime_deduplicates_across_searches() -> None:
    asyncio.run(_run_news_runtime_deduplicates_across_searches())


@pytest.mark.parametrize("status_code", [404, 503])
def test_news_runtime_skips_unfetchable_document(status_code: int) -> None:
    asyncio.run(_run_news_runtime_skips_unfetchable_document(status_code))


async def _run_document_fetch_reports_failed_when_all_documents_fail() -> None:
    store = InMemoryArtifactStore()
    search_result = SearchResult(
        title="Unavailable",
        url="https://example.com/unavailable",
    )
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        search_results={
            store.put(search_result): SearchResultState()
        },
    )
    executor = _news_executor(
        store,
        document_fetcher=BlockedDocumentFetcher(),
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.FETCH_DOCUMENTS),
        state,
    )

    assert isinstance(observation, FetchDocumentsObservation)
    assert [
        outcome.model_dump(mode="json")
        for outcome in observation.fetch_outcomes
    ] == [
        {
            "status": "failure",
            "search_result_id": search_result.id,
            "failure": {
                "url": search_result.url,
                "status_code": 503,
                "reason": "http_status",
                "retryable": True,
                "message": "HTTP 503 while fetching document",
                "source_error_type": "HTTPStatusError",
            },
        }
    ]
    assert observation.document_index_updates == {}


def test_document_fetch_reports_failed_when_all_documents_fail() -> None:
    asyncio.run(_run_document_fetch_reports_failed_when_all_documents_fail())


class BrokenDocumentFetcher(FakeDocumentFetcher):
    async def fetch(self, request):
        raise TypeError("fetcher implementation bug")


def test_news_runtime_does_not_hide_unknown_fetcher_errors() -> None:
    store = InMemoryArtifactStore()
    trace_sink = InMemoryTraceSink()
    runtime = _news_runtime(
        store,
        document_fetcher=BrokenDocumentFetcher(),
        tracer=Tracer(trace_sink),
    )

    with pytest.raises(RuntimeExecutionError) as caught:
        asyncio.run(runtime.run(AgentState(query=UserQuery(text="latest AI news"))))

    assert isinstance(caught.value.original_error, TypeError)
    spans = trace_sink.get_trace(caught.value.trace_id)
    failed_executor = next(
        span
        for span in spans
        if span.name == "agent.action.execute" and span.status == "error"
    )
    assert failed_executor.error is not None
    assert failed_executor.error.error_type == "TypeError"
    assert failed_executor.attributes["step_index"] == 2
    assert len(
        [span for span in spans if span.name == "agent.step" and span.status == "ok"]
    ) == 2


class TransientDocumentFetcher(FakeDocumentFetcher):
    def __init__(self) -> None:
        self.call_count = 0

    async def fetch(self, request):
        self.call_count += 1
        if self.call_count == 1:
            raise DocumentFetchError(
                url=request.url,
                reason="timeout",
                message="temporary timeout",
                source_error_type="ReadTimeout",
            )
        return await super().fetch(request)


class RedirectingDocumentFetcher(FakeDocumentFetcher):
    def __init__(self) -> None:
        self.call_count = 0

    async def fetch(self, request):
        self.call_count += 1
        document = await super().fetch(request)
        document.url = "https://official.example/article?utm_source=redirect"
        return document


async def _run_document_fetch_retries_then_stops_after_success() -> None:
    store = InMemoryArtifactStore()
    result = SearchResult(
        title="Article",
        url="https://example.com/article",
    )
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        search_results={store.put(result): SearchResultState()},
    )
    fetcher = TransientDocumentFetcher()
    executor = _news_executor(store, document_fetcher=fetcher)
    reducer = DefaultStateReducer()
    action = AgentAction(type=AgentActionType.FETCH_DOCUMENTS)

    first = await executor.execute(action, state)
    state = reducer.apply(state, action, first)
    assert state.search_results[result.id].attempt_count == 1
    assert state.search_results[result.id].failure is not None
    assert state.search_results[result.id].failure.retryable is True

    second = await executor.execute(action, state)
    state = reducer.apply(state, action, second)
    assert state.search_results[result.id].attempt_count == 2
    assert state.search_results[result.id].failure is None
    assert len(state.documents) == 1

    third = await executor.execute(action, state)
    assert isinstance(third, FetchDocumentsObservation)
    assert third.fetch_outcomes == []
    assert third.document_index_updates == {}
    assert fetcher.call_count == 2


def test_document_fetch_retries_then_stops_after_success() -> None:
    asyncio.run(_run_document_fetch_retries_then_stops_after_success())


async def _run_document_fetch_reuses_redirect_target() -> None:
    store = InMemoryArtifactStore()
    results = [
        SearchResult(
            title="Redirect",
            url="https://short.example/article",
        ),
        SearchResult(
            title="Canonical",
            url="https://official.example/article",
        ),
    ]
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        search_results={
            store.put(result): SearchResultState() for result in results
        },
    )
    fetcher = RedirectingDocumentFetcher()
    executor = _news_executor(store, document_fetcher=fetcher)
    action = AgentAction(type=AgentActionType.FETCH_DOCUMENTS)

    observation = await executor.execute(action, state)
    state = DefaultStateReducer().apply(state, action, observation)

    assert fetcher.call_count == 1
    assert len(store.list(Document)) == 1
    assert len(state.documents) == 1
    document_id = next(iter(state.documents))
    assert state.search_results[results[0].id].document_id == document_id
    assert state.search_results[results[1].id].document_id == document_id
    assert state.document_index == {
        "https://official.example/article": document_id,
    }


def test_document_fetch_reuses_redirect_target() -> None:
    asyncio.run(_run_document_fetch_reuses_redirect_target())

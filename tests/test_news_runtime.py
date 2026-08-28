"""Integration tests for the atomic news research workflow."""

import asyncio
from collections.abc import Iterable

import pytest

from banso.artifacts.store import InMemoryArtifactStore
from banso.agent.action import (
    AgentAction,
    AgentActionType,
    RetrievalRoute,
)
from banso.agent.observation import (
    CompletedResearchObservation,
    FailedResearchObservation,
)
from banso.agent.reducer import DefaultStateReducer
from banso.agent.runtime import AgentRuntime
from banso.agent.executors.retry import RetryPolicy
from banso.agent.selection.selector import (
    SearchResultSelection,
    SearchResultSelectionError,
    SearchResultSelectionRequest,
    SearchResultSelector,
)
from banso.agent.state import AgentState, ExecutionBudget, UserQuery
from banso.documents.extractor import EvidenceExtractionRequest
from banso.documents.fetcher import DocumentFetchRequest
from banso.documents.models import Document, DocumentEvidence
from banso.agent.executors.news_executor import NewsActionExecutor
from banso.agent.executors.research_pipeline import ResearchRouteComponents
from banso.retrieval.models import SearchResult
from banso.retrieval.provider import RetrievalError, SearchRequest
from banso.synthesis.synthesizer import Citation, SynthesisRequest, SynthesisResult
from banso.tracing.trace import InMemoryTraceSink, Tracer


class SequencePolicy:
    def __init__(self, actions: Iterable[AgentAction]) -> None:
        self.actions = list(actions)

    async def select_action(self, state: AgentState) -> AgentAction:
        return self.actions[state.current_step]


class StaticRetrievalProvider:
    def __init__(self, prefix: str, count: int = 1) -> None:
        self.prefix = prefix
        self.count = count
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        self.requests.append(request)
        return [
            SearchResult(
                id=f"{self.prefix}-result-{index}",
                title=f"Result {index}",
                url=f"https://example.com/{self.prefix}/{index}",
                snippet=f"Snippet {index}",
                rank=index,
            )
            for index in range(1, self.count + 1)
        ]


class FailingRetrievalProvider:
    def __init__(self, *, failures: int, retryable: bool) -> None:
        self.failures = failures
        self.retryable = retryable
        self.attempt_count = 0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        self.attempt_count += 1
        if self.attempt_count <= self.failures:
            raise RetrievalError(
                provider="test",
                reason="transport" if self.retryable else "http_status",
                status_code=None if self.retryable else 400,
                message="retrieval failed",
                source_error_type="TestError",
            )
        return [
            SearchResult(
                id="recovered-result",
                title="Recovered",
                url="https://example.com/recovered",
            )
        ]


class RecordingFetcher:
    def __init__(self) -> None:
        self.requests: list[DocumentFetchRequest] = []

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        self.requests.append(request)
        suffix = request.url.rsplit("/", 1)[-1]
        return Document(
            id=f"document-{len(self.requests)}",
            url=request.url,
            title=request.title or suffix,
            text=f"Body for {suffix}",
            source=request.source,
        )


class EvidenceExtractor:
    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> str:
        return f"Claim from {request.document.title}"


class RecordingSynthesizer:
    def __init__(self) -> None:
        self.requests: list[SynthesisRequest] = []

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        return SynthesisResult(
            answer="answer",
            citations=[
                Citation(
                    reference=f"S{index}",
                    document_id=group.document_id,
                    source_url=group.source_url,
                )
                for index, group in enumerate(request.evidence_groups, start=1)
            ],
        )


class SecondResultSelector:
    def __init__(self) -> None:
        self.requests: list[SearchResultSelectionRequest] = []

    async def select(
        self,
        request: SearchResultSelectionRequest,
    ) -> SearchResultSelection:
        self.requests.append(request)
        return SearchResultSelection(selected_ids=[request.candidates[1].id])


class SequencedSelector:
    def __init__(self, outcomes: list[Exception | SearchResultSelection]) -> None:
        self.outcomes = outcomes
        self.attempt_count = 0

    async def select(
        self,
        request: SearchResultSelectionRequest,
    ) -> SearchResultSelection:
        outcome = self.outcomes[self.attempt_count]
        self.attempt_count += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _executor(
    store: InMemoryArtifactStore,
    routes: dict[RetrievalRoute, ResearchRouteComponents],
    synthesizer: RecordingSynthesizer | None = None,
    search_result_selector: SearchResultSelector | None = None,
    retry_policy: RetryPolicy | None = None,
) -> NewsActionExecutor:
    return NewsActionExecutor(
        store=store,
        research_routes=routes,
        evidence_extractor=EvidenceExtractor(),
        synthesizer=synthesizer or RecordingSynthesizer(),
        search_result_selector=search_result_selector,
        retry_policy=retry_policy,
    )


def _web_executor(provider) -> NewsActionExecutor:
    return _executor(
        InMemoryArtifactStore(),
        {
            RetrievalRoute.WEB: ResearchRouteComponents(
                provider,
                RecordingFetcher(),
            )
        },
    )


def test_research_routes_and_bounds_new_documents_atomically() -> None:
    store = InMemoryArtifactStore()
    web = StaticRetrievalProvider("web", count=3)
    local = StaticRetrievalProvider("local")
    web_fetcher = RecordingFetcher()
    local_fetcher = RecordingFetcher()
    executor = _executor(
        store,
        {
            RetrievalRoute.WEB: ResearchRouteComponents(web, web_fetcher),
            RetrievalRoute.LOCAL: ResearchRouteComponents(local, local_fetcher),
        },
    )
    action = AgentAction(
        type=AgentActionType.RESEARCH,
        params={"query": "latest news", "route": "web"},
    )

    state = AgentState(
        query=UserQuery(text="news"),
        budget=ExecutionBudget(max_results_per_research=2),
    )
    observation = asyncio.run(executor.execute(action, state))

    assert isinstance(observation, CompletedResearchObservation)
    assert len(observation.search_result_ids) == 3
    assert observation.selection_report.selected_ids == observation.search_result_ids
    assert len(observation.fetch_outcomes) == 2
    assert len(observation.extraction_outcomes) == 2
    assert len(web_fetcher.requests) == 2
    assert not local.requests
    assert not local_fetcher.requests


def test_injected_selector_controls_which_candidates_are_processed() -> None:
    store = InMemoryArtifactStore()
    provider = StaticRetrievalProvider("web", count=3)
    fetcher = RecordingFetcher()
    selector = SecondResultSelector()
    executor = _executor(
        store,
        {RetrievalRoute.WEB: ResearchRouteComponents(provider, fetcher)},
        search_result_selector=selector,
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "specific gap", "route": "web"},
            ),
            AgentState(query=UserQuery(text="overall question")),
        )
    )

    selected_id = observation.search_result_ids[1]
    assert selector.requests[0].research_query == "specific gap"
    assert [result.id for result in selector.requests[0].candidates] == (
        observation.search_result_ids
    )
    assert observation.selection_report.selected_ids == [selected_id]
    assert fetcher.requests[0].url.endswith("/2")


def test_selector_validation_failure_retries_once_and_recovers() -> None:
    provider = StaticRetrievalProvider("web")
    fetcher = RecordingFetcher()
    selector = SequencedSelector(
        [
            SearchResultSelectionError("invalid response"),
            SearchResultSelection(selected_ids=["web-result-1"]),
        ]
    )
    executor = _executor(
        InMemoryArtifactStore(),
        {RetrievalRoute.WEB: ResearchRouteComponents(provider, fetcher)},
        search_result_selector=selector,
        retry_policy=RetryPolicy(delay_seconds=0),
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "query", "route": "web"},
            ),
            AgentState(query=UserQuery(text="question")),
        )
    )

    assert isinstance(observation, CompletedResearchObservation)
    assert selector.attempt_count == 2
    assert len(fetcher.requests) == 1


def test_selector_validation_failure_becomes_handled_research_failure() -> None:
    provider = StaticRetrievalProvider("web")
    fetcher = RecordingFetcher()
    selector = SequencedSelector(
        [
            SearchResultSelectionError("invalid response one"),
            SearchResultSelectionError("invalid response two"),
        ]
    )
    executor = _executor(
        InMemoryArtifactStore(),
        {RetrievalRoute.WEB: ResearchRouteComponents(provider, fetcher)},
        search_result_selector=selector,
        retry_policy=RetryPolicy(delay_seconds=0),
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "query", "route": "web"},
            ),
            AgentState(query=UserQuery(text="question")),
        )
    )

    assert isinstance(observation, FailedResearchObservation)
    assert observation.stage == "selection"
    assert observation.reason == "invalid_response"
    assert observation.attempt_count == 2
    assert selector.attempt_count == 2
    assert fetcher.requests == []


def test_unexpected_selector_failure_is_not_retried() -> None:
    fetcher = RecordingFetcher()
    selector = SequencedSelector([RuntimeError("selector crashed")])
    executor = _executor(
        InMemoryArtifactStore(),
        {
            RetrievalRoute.WEB: ResearchRouteComponents(
                StaticRetrievalProvider("web"),
                fetcher,
            )
        },
        search_result_selector=selector,
        retry_policy=RetryPolicy(delay_seconds=0),
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "query", "route": "web"},
            ),
            AgentState(query=UserQuery(text="question")),
        )
    )

    assert isinstance(observation, FailedResearchObservation)
    assert observation.reason == "unexpected_error"
    assert observation.source_error_type == "RuntimeError"
    assert observation.retryable is False
    assert observation.attempt_count == 1
    assert selector.attempt_count == 1
    assert fetcher.requests == []


def test_runtime_researches_then_finishes_from_evidence() -> None:
    store = InMemoryArtifactStore()
    provider = StaticRetrievalProvider("web", count=2)
    fetcher = RecordingFetcher()
    synthesizer = RecordingSynthesizer()
    sink = InMemoryTraceSink()
    tracer = Tracer(sink)
    runtime = AgentRuntime(
        policy=SequencePolicy(
            [
                AgentAction(
                    type=AgentActionType.RESEARCH,
                    params={
                        "query": "query",
                        "route": "web",
                        "source_domains": ["x.com"],
                    },
                ),
                AgentAction(type=AgentActionType.FINISH),
            ]
        ),
        executor=_executor(
            store,
            {
                RetrievalRoute.WEB: ResearchRouteComponents(provider, fetcher),
            },
            synthesizer,
        ),
        tracer=tracer,
    )

    output = asyncio.run(runtime.run(AgentState(query=UserQuery(text="query"))))
    state = output.result.state

    assert state.done is True
    assert state.final_answer == "answer"
    assert len(state.search_results) == 2
    assert len(state.documents) == 2
    evidence = store.list(DocumentEvidence)
    assert len(evidence) == 2
    assert {document.evidence_id for document in state.documents.values()} == {
        item.id for item in evidence
    }
    assert len(synthesizer.requests[0].evidence_groups) == 2
    spans = sink.get_trace(output.trace_id)
    span_names = {span.name for span in spans}
    assert {
        "news.research.retrieve",
        "news.research.select",
        "news.research.fetch",
        "news.research.extract",
    } <= span_names
    assert provider.requests[0].source_domains == ["x.com"]
    research = state.action_history[0].observation
    assert research.source_domains == ["x.com"]
    retrieve_span = next(span for span in spans if span.name == "news.research.retrieve")
    assert retrieve_span.input["source_domains"] == ["x.com"]


def test_retrieval_failure_is_recorded_without_artifacts_and_consumes_budget() -> None:
    provider = FailingRetrievalProvider(failures=2, retryable=True)
    sink = InMemoryTraceSink()
    runtime = AgentRuntime(
        policy=SequencePolicy(
            [
                AgentAction(
                    type=AgentActionType.RESEARCH,
                    params={
                        "query": "query",
                        "route": "web",
                        "source_domains": ["x.com"],
                    },
                ),
                AgentAction(type=AgentActionType.STOP),
            ]
        ),
        executor=_web_executor(provider),
        tracer=Tracer(sink),
    )

    output = asyncio.run(runtime.run(AgentState(query=UserQuery(text="query"))))
    state = output.result.state
    failure = state.action_history[0].observation

    assert provider.attempt_count == 2
    assert isinstance(failure, FailedResearchObservation)
    assert failure.stage == "retrieval"
    assert failure.reason == "transport"
    assert failure.attempt_count == 2
    assert failure.source_domains == ["x.com"]
    assert state.current_step == 2
    assert state.remaining_research_capacity == state.budget.max_researches - 1
    assert state.search_results == {}
    assert state.documents == {}
    spans = sink.get_trace(output.trace_id)
    retrieve_span = next(span for span in spans if span.name == "news.research.retrieve")
    assert retrieve_span.attributes["outcome"] == "failure"
    assert retrieve_span.input["source_domains"] == ["x.com"]


def test_terminal_results_do_not_reenter_result_selection() -> None:
    store = InMemoryArtifactStore()
    provider = StaticRetrievalProvider("web")
    fetcher = RecordingFetcher()
    executor = _executor(
        store,
        {RetrievalRoute.WEB: ResearchRouteComponents(provider, fetcher)},
    )
    state = AgentState(query=UserQuery(text="query"))
    first_action = AgentAction(
        type=AgentActionType.RESEARCH,
        params={"query": "first", "route": "web"},
    )
    first_observation = asyncio.run(executor.execute(first_action, state))
    state = DefaultStateReducer().apply(state, first_action, first_observation)

    second_observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "second", "route": "web"},
            ),
            state,
        )
    )

    assert second_observation.selection_report.candidate_ids == []
    assert second_observation.fetch_outcomes == []


def test_unprocessed_results_do_not_enter_the_next_research() -> None:
    store = InMemoryArtifactStore()
    provider = StaticRetrievalProvider("web", count=3)
    executor = _executor(
        store,
        {
            RetrievalRoute.WEB: ResearchRouteComponents(
                provider,
                RecordingFetcher(),
            )
        },
    )
    state = AgentState(
        query=UserQuery(text="query"),
        budget=ExecutionBudget(max_results_per_research=1),
    )
    first_action = AgentAction(
        type=AgentActionType.RESEARCH,
        params={"query": "first", "route": "web"},
    )
    first_observation = asyncio.run(executor.execute(first_action, state))
    state = DefaultStateReducer().apply(state, first_action, first_observation)
    provider.count = 0

    second_observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "second", "route": "web"},
            ),
            state,
        )
    )

    assert second_observation.search_result_ids == []
    assert second_observation.selection_report.candidate_ids == []
    assert second_observation.fetch_outcomes == []


def test_research_rejects_a_disabled_route() -> None:
    store = InMemoryArtifactStore()
    executor = _executor(
        store,
        {
            RetrievalRoute.WEB: ResearchRouteComponents(
                StaticRetrievalProvider("web"),
                RecordingFetcher(),
            )
        },
    )

    try:
        asyncio.run(
            executor.execute(
                AgentAction(
                    type=AgentActionType.RESEARCH,
                    params={"query": "query", "route": "local"},
                ),
                AgentState(query=UserQuery(text="query")),
            )
        )
    except ValueError as error:
        assert "not enabled" in str(error)
    else:
        raise AssertionError("disabled route was accepted")


def test_executor_requires_positive_extraction_concurrency() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        NewsActionExecutor(
            store=InMemoryArtifactStore(),
            research_routes={
                RetrievalRoute.WEB: ResearchRouteComponents(
                    StaticRetrievalProvider("web"),
                    RecordingFetcher(),
                )
            },
            evidence_extractor=EvidenceExtractor(),
            synthesizer=RecordingSynthesizer(),
            max_extraction_concurrency=0,
        )

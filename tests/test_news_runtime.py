"""Integration tests for the atomic news research workflow."""

import asyncio
from collections.abc import Iterable

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentAction,
    AgentActionType,
    AgentRuntime,
    AgentState,
    DefaultStateReducer,
    ExecutionBudget,
    RetrievalRoute,
    UserQuery,
)
from banso.core.observation import ResearchObservation
from banso.documents import (
    Document,
    DocumentFetchError,
    DocumentFetchRequest,
    EvidenceExtractionRequest,
    EvidenceItem,
)
from banso.executors import NewsActionExecutor, ResearchRouteComponents
from banso.retrieval import SearchRequest, SearchResult
from banso.synthesis import SynthesisRequest, SynthesisResult
from banso.tracing import InMemoryTraceSink, Tracer


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


class RecordingFetcher:
    def __init__(self, *, redirect_url: str | None = None) -> None:
        self.redirect_url = redirect_url
        self.requests: list[DocumentFetchRequest] = []

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        self.requests.append(request)
        suffix = request.url.rsplit("/", 1)[-1]
        return Document(
            id=f"document-{len(self.requests)}",
            url=self.redirect_url or request.url,
            title=request.title or suffix,
            text=f"Body for {suffix}",
            source=request.source,
        )


class FlakyFetcher:
    def __init__(self) -> None:
        self.attempt_count = 0

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        self.attempt_count += 1
        if self.attempt_count == 1:
            raise DocumentFetchError(
                url=request.url,
                reason="timeout",
                message="timed out",
                source_error_type="TimeoutError",
            )
        return Document(
            id="document-after-retry",
            url=request.url,
            title=request.title,
            text="Body after retry",
            source=request.source,
        )


class EvidenceExtractor:
    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        return [
            EvidenceItem(
                id=f"evidence-{request.document.id}",
                document_id=request.document.id,
                claim=f"Claim from {request.document.title}",
                source_url=request.document.url,
            )
        ]


class RecordingSynthesizer:
    def __init__(self) -> None:
        self.requests: list[SynthesisRequest] = []

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        return SynthesisResult(
            answer="answer",
            citations=[item.source_url for item in request.evidence],
        )


def _executor(
    store: InMemoryArtifactStore,
    routes: dict[RetrievalRoute, ResearchRouteComponents],
    synthesizer: RecordingSynthesizer | None = None,
) -> NewsActionExecutor:
    return NewsActionExecutor(
        store=store,
        research_routes=routes,
        evidence_extractor=EvidenceExtractor(),
        synthesizer=synthesizer or RecordingSynthesizer(),
    )


def test_research_routes_and_processes_selected_results_atomically() -> None:
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

    assert isinstance(observation, ResearchObservation)
    assert len(observation.search_result_ids) == 3
    assert (
        observation.selection_report.selected_ids
        == observation.search_result_ids[:2]
    )
    assert (
        observation.selection_report.deferred_ids
        == observation.search_result_ids[2:]
    )
    assert len(observation.fetch_outcomes) == 2
    assert len(observation.extraction_outcomes) == 2
    assert len(web_fetcher.requests) == 2
    assert not local.requests
    assert not local_fetcher.requests


def test_runtime_researches_then_finishes_from_active_evidence() -> None:
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
                    params={"query": "query", "route": "web"},
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
    assert all(
        document.lifecycle_status == "active"
        for document in state.documents.values()
    )
    assert len(synthesizer.requests[0].evidence) == 2
    span_names = {span.name for span in sink.get_trace(output.trace_id)}
    assert {
        "news.research.retrieve",
        "news.research.select",
        "news.research.fetch",
        "news.research.extract",
    } <= span_names


def test_fetch_retries_within_the_research_action() -> None:
    store = InMemoryArtifactStore()
    fetcher = FlakyFetcher()
    executor = _executor(
        store,
        {
            RetrievalRoute.WEB: ResearchRouteComponents(
                StaticRetrievalProvider("web"),
                fetcher,
            )
        },
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "query", "route": "web"},
            ),
            AgentState(query=UserQuery(text="query")),
        )
    )

    assert fetcher.attempt_count == 2
    assert observation.fetch_outcomes[0].attempt_count == 2
    assert len(observation.extraction_outcomes) == 1


def test_global_document_budget_bounds_research_selection() -> None:
    store = InMemoryArtifactStore()
    provider = StaticRetrievalProvider("web", count=4)
    fetcher = RecordingFetcher()
    executor = _executor(
        store,
        {RetrievalRoute.WEB: ResearchRouteComponents(provider, fetcher)},
    )
    state = AgentState(
        query=UserQuery(text="query"),
        budget=ExecutionBudget(max_document_fetches=2),
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "query", "route": "web"},
            ),
            state,
        )
    )

    assert len(observation.selection_report.selected_ids) == 2
    assert len(observation.selection_report.deferred_ids) == 2
    assert len(fetcher.requests) == 2


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


def test_deferred_results_reenter_the_next_result_selection() -> None:
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
    expected_result_id = first_observation.selection_report.deferred_ids[0]
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
    assert second_observation.selection_report.selected_ids == [expected_result_id]
    assert second_observation.fetch_outcomes[0].search_result_id == expected_result_id


def test_result_limit_bounds_fetch_and_extraction_together() -> None:
    store = InMemoryArtifactStore()
    provider = StaticRetrievalProvider("web", count=3)
    fetcher = RecordingFetcher()
    executor = _executor(
        store,
        {RetrievalRoute.WEB: ResearchRouteComponents(provider, fetcher)},
    )
    state = AgentState(
        query=UserQuery(text="query"),
        budget=ExecutionBudget(
            max_results_per_research=1,
        ),
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "query", "route": "web"},
            ),
            state,
        )
    )

    assert len(observation.selection_report.selected_ids) == 1
    assert len(observation.selection_report.deferred_ids) == 2
    assert len(observation.fetch_outcomes) == 1
    assert len(observation.extraction_outcomes) == 1


def test_redirected_documents_are_deduplicated_within_research() -> None:
    store = InMemoryArtifactStore()
    provider = StaticRetrievalProvider("web", count=2)
    fetcher = RecordingFetcher(redirect_url="https://example.com/canonical")
    executor = _executor(
        store,
        {RetrievalRoute.WEB: ResearchRouteComponents(provider, fetcher)},
    )
    state = AgentState(query=UserQuery(text="query"))
    action = AgentAction(
        type=AgentActionType.RESEARCH,
        params={"query": "query", "route": "web"},
    )

    observation = asyncio.run(executor.execute(action, state))
    next_state = DefaultStateReducer().apply(state, action, observation)

    assert len(next_state.documents) == 1
    assert {
        result.document_id for result in next_state.search_results.values()
    } == set(next_state.documents)


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

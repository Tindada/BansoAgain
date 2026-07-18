"""Smoke test for the basic news runtime workflow."""

import asyncio

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentRuntime,
    AgentState,
    ExecutionBudget,
    PlannedSearch,
    RuntimeExecutionError,
    SearchPlan,
    UserQuery,
)
from banso.core.action import AgentAction, AgentActionType
from banso.documents import (
    Document,
    DocumentReadError,
    FakeDocumentReader,
    FakeEvidenceExtractor,
)
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import (
    FakeRetrievalProvider,
    SearchPlanningRequest,
    SearchRequest,
    SearchResult,
    Source,
    SourceType,
)
from banso.synthesis import FakeSynthesizer


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


class PartiallyBlockedDocumentReader(FakeDocumentReader):
    def __init__(self, status_code: int = 403) -> None:
        self.status_code = status_code

    async def read(self, request):
        if request.url.endswith("blocked"):
            raise DocumentReadError(
                url=request.url,
                reason="http_status",
                message=f"HTTP {self.status_code} while reading document",
                status_code=self.status_code,
                source_error_type="HTTPStatusError",
            )
        return await super().read(request)


class BlockedDocumentReader(FakeDocumentReader):
    async def read(self, request):
        raise DocumentReadError(
            url=request.url,
            reason="http_status",
            message="HTTP 503 while reading document",
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
                title="Readable",
                url="https://example.com/readable",
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


async def _run_news_runtime() -> None:
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
        ),
    )

    output = await runtime.run(AgentState(query=UserQuery(text="latest AI news")))
    state = output.result.state

    assert state.done is True
    assert [step.action.type for step in output.trace.steps] == [
        AgentActionType.PLAN_SEARCH,
        AgentActionType.SEARCH,
        AgentActionType.READ_DOCUMENT,
        AgentActionType.EXTRACT_EVIDENCE,
        AgentActionType.SYNTHESIZE,
        AgentActionType.STOP,
    ]
    assert state.search_plan is not None
    assert state.search_plan.model_dump() == {
        "searches": [{"query": "latest AI news", "intent": "general"}]
    }
    assert output.trace.steps[0].observation.data["search_plan"] == (
        state.search_plan.model_dump(mode="json")
    )
    assert state.action_history[1].observation.data["search_queries"] == [
        "latest AI news"
    ]
    assert len(state.search_result_ids) == 1
    assert len(state.document_ids) == 1
    assert len(state.evidence_ids) == 1
    assert state.final_answer is not None
    assert "Fake summary for 'latest AI news'" in state.final_answer
    synthesis_observation = output.trace.steps[4].observation
    assert state.citations == synthesis_observation.data["citations"]
    assert "final_answer" not in output.result.model_dump()
    assert output.trace.final_result == output.result
    assert output.trace.status == "completed"
    assert output.trace.failure is None
    assert all(
        step.policy_duration_seconds is not None
        and step.policy_duration_seconds >= 0
        and step.executor_duration_seconds is not None
        and step.executor_duration_seconds >= 0
        and step.reducer_duration_seconds is not None
        and step.reducer_duration_seconds >= 0
        for step in output.trace.steps
    )


async def _run_news_runtime_filters_search_results() -> None:
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=DuplicateRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
        ),
    )

    output = await runtime.run(AgentState(query=UserQuery(text="latest AI news")))
    state = output.result.state
    search_observation = output.trace.steps[1].observation

    assert len(state.search_result_ids) == 2
    assert len(state.document_ids) == 2
    assert search_observation.data["retrieval_filter_report"] == {
        "input_count": 3,
        "output_count": 2,
        "dropped_empty_title": 0,
        "dropped_empty_url": 0,
        "dropped_duplicate_url": 1,
        "truncated_count": 0,
    }
    classification_report = search_observation.data[
        "source_classification_report"
    ]
    assert classification_report["input_count"] == 2
    assert classification_report["recognized_count"] == 2
    assert classification_report["unknown_count"] == 0


async def _run_news_runtime_respects_document_read_budget() -> None:
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=DuplicateRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
        ),
    )

    output = await runtime.run(
        AgentState(
            query=UserQuery(text="latest AI news"),
            budget=ExecutionBudget(max_documents_to_read=1),
        )
    )

    assert len(output.result.state.search_result_ids) == 2
    assert len(output.result.state.document_ids) == 1


async def _run_news_runtime_preserves_search_order_when_reading() -> None:
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=MixedTrustRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
            search_query_planner=TwoQueryPlanner(),
        ),
    )

    output = await runtime.run(
        AgentState(
            query=UserQuery(text="latest AI news"),
            budget=ExecutionBudget(max_searches=2, max_documents_to_read=8),
        )
    )
    results = [
        store.get(result_id, SearchResult)
        for result_id in output.result.state.search_result_ids
    ]
    documents = [
        store.get(document_id, Document)
        for document_id in output.result.state.document_ids
    ]

    assert len(output.result.state.search_result_ids) == 8
    assert len(output.result.state.document_ids) == 8
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


async def _run_news_runtime_skips_unreadable_document(status_code: int) -> None:
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=PartiallyBlockedRetrievalProvider(),
            document_reader=PartiallyBlockedDocumentReader(status_code),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
        ),
    )

    output = await runtime.run(AgentState(query=UserQuery(text="latest AI news")))
    read_observation = output.trace.steps[2].observation

    assert output.result.state.done is True
    assert len(output.result.state.document_ids) == 1
    assert read_observation.data["document_read_failures"] == [
        {
            "search_result_id": output.result.state.search_result_ids[0],
            "url": "https://example.com/blocked",
            "status_code": status_code,
            "reason": "http_status",
            "message": f"HTTP {status_code} while reading document",
            "source_error_type": "HTTPStatusError",
        }
    ]
    assert read_observation.data["successfully_read_document_count"] == 1
    assert read_observation.data["failed_document_count"] == 1


def test_news_runtime() -> None:
    asyncio.run(_run_news_runtime())


def test_news_runtime_filters_search_results() -> None:
    asyncio.run(_run_news_runtime_filters_search_results())


def test_news_runtime_respects_document_read_budget() -> None:
    asyncio.run(_run_news_runtime_respects_document_read_budget())


def test_news_runtime_preserves_search_order_when_reading() -> None:
    asyncio.run(_run_news_runtime_preserves_search_order_when_reading())


@pytest.mark.parametrize("status_code", [401, 403, 404, 410, 500, 521])
def test_news_runtime_skips_unreadable_document(status_code: int) -> None:
    asyncio.run(_run_news_runtime_skips_unreadable_document(status_code))


async def _run_document_read_reports_failed_when_all_documents_fail() -> None:
    store = InMemoryArtifactStore()
    search_result = SearchResult(
        title="Unavailable",
        url="https://example.com/unavailable",
    )
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        search_result_ids=[store.put(search_result)],
    )
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_reader=BlockedDocumentReader(),
        evidence_extractor=FakeEvidenceExtractor(),
        synthesizer=FakeSynthesizer(),
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.READ_DOCUMENT),
        state,
    )

    assert observation.data["successfully_read_document_count"] == 0
    assert observation.data["failed_document_count"] == 1
    assert observation.data["document_ids"] == []


def test_document_read_reports_failed_when_all_documents_fail() -> None:
    asyncio.run(_run_document_read_reports_failed_when_all_documents_fail())


class BrokenDocumentReader(FakeDocumentReader):
    async def read(self, request):
        raise TypeError("reader implementation bug")


def test_news_runtime_does_not_hide_unknown_reader_errors() -> None:
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=BrokenDocumentReader(),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
        ),
    )

    with pytest.raises(RuntimeExecutionError) as caught:
        asyncio.run(runtime.run(AgentState(query=UserQuery(text="latest AI news"))))

    assert isinstance(caught.value.original_error, TypeError)
    assert caught.value.trace.status == "failed"
    assert caught.value.trace.failure is not None
    assert caught.value.trace.failure.phase == "executor"
    assert caught.value.trace.failure.error_type == "TypeError"
    assert caught.value.trace.failure.step_index == 2
    assert len(caught.value.trace.steps) == 2

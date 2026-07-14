"""Smoke test for the basic news runtime workflow."""

import asyncio

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentRuntime, AgentState, ExecutionBudget, UserQuery
from banso.core.action import AgentActionType
from banso.documents import (
    DocumentHTTPStatusError,
    FakeDocumentReader,
    FakeEvidenceExtractor,
)
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import (
    FakeRetrievalProvider,
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
            raise DocumentHTTPStatusError(
                url=request.url,
                status_code=self.status_code,
            )
        return await super().read(request)


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
    assert state.search_queries == ["latest AI news"]
    assert len(state.search_result_ids) == 1
    assert len(state.document_ids) == 1
    assert len(state.evidence_ids) == 1
    assert output.result.final_answer is not None
    assert "Fake summary for 'latest AI news'" in output.result.final_answer
    assert output.trace.final_result == output.result
    assert all(
        step.duration_seconds is not None and step.duration_seconds >= 0
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
    evaluation_report = search_observation.data[
        "search_result_evaluation_report"
    ]
    assert evaluation_report["accepted_count"] == 2
    assert evaluation_report["rejected_count"] == 0


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
        }
    ]


def test_news_runtime() -> None:
    asyncio.run(_run_news_runtime())


def test_news_runtime_filters_search_results() -> None:
    asyncio.run(_run_news_runtime_filters_search_results())


def test_news_runtime_respects_document_read_budget() -> None:
    asyncio.run(_run_news_runtime_respects_document_read_budget())


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_news_runtime_skips_unreadable_document(status_code: int) -> None:
    asyncio.run(_run_news_runtime_skips_unreadable_document(status_code))


def test_news_runtime_does_not_hide_other_http_errors() -> None:
    with pytest.raises(DocumentHTTPStatusError, match="HTTP 500"):
        asyncio.run(_run_news_runtime_skips_unreadable_document(500))

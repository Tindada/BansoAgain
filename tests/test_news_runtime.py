"""Smoke test for the basic news runtime workflow."""

import asyncio

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentRuntime, AgentState, UserQuery
from banso.core.action import AgentActionType
from banso.documents import FakeDocumentReader, FakeEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import FakeRetrievalProvider, SearchRequest, SearchResult
from banso.synthesis import FakeSynthesizer


class DuplicateRetrievalProvider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"First result for {request.query}",
                url="https://example.com/news?a=1&utm_source=test",
                rank=1,
            ),
            SearchResult(
                title=f"Duplicate result for {request.query}",
                url="https://example.com/news?utm_medium=test&a=1",
                rank=2,
            ),
            SearchResult(
                title=f"Second result for {request.query}",
                url="https://example.com/second",
                rank=3,
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
        AgentActionType.SEARCH,
        AgentActionType.READ_DOCUMENT,
        AgentActionType.EXTRACT_EVIDENCE,
        AgentActionType.SYNTHESIZE,
        AgentActionType.STOP,
    ]
    assert state.search_queries == ["latest AI news"]
    assert len(state.search_result_ids) == 1
    assert len(state.document_ids) == 1
    assert len(state.evidence_ids) == 1
    assert output.result.final_answer is not None
    assert "Fake summary for 'latest AI news'" in output.result.final_answer
    assert output.trace.final_result == output.result


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
    search_observation = output.trace.steps[0].observation

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


def test_news_runtime() -> None:
    asyncio.run(_run_news_runtime())


def test_news_runtime_filters_search_results() -> None:
    asyncio.run(_run_news_runtime_filters_search_results())

"""Smoke test for the basic news runtime workflow."""

import asyncio

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentRuntime, AgentState, UserQuery
from banso.core.action import AgentActionType
from banso.documents import FakeDocumentReader, FakeEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import FakeRetrievalProvider
from banso.synthesis import FakeSynthesizer


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


def test_news_runtime() -> None:
    asyncio.run(_run_news_runtime())

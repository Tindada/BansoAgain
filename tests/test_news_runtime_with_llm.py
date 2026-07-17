"""Integration test for the news runtime with LLM-backed components."""

import asyncio

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentRuntime, AgentState, UserQuery
from banso.core.action import AgentActionType
from banso.documents import FakeDocumentReader, LLMEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.llm import FakeLLMClient
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import FakeRetrievalProvider
from banso.synthesis import LLMSynthesizer


async def _run_news_runtime_with_llm() -> None:
    evidence_client = FakeLLMClient(
        content=(
            "["
            '{"claim":"The fake document reports on latest AI news.",'
            '"supporting_text":"This is a fake document body for Fake result for latest AI news.",'
            '"confidence":0.8}'
            "]"
        )
    )
    synthesis_client = FakeLLMClient(content="LLM synthesized news summary.")
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=LLMEvidenceExtractor(
                client=evidence_client,
                model="fake-evidence-model",
            ),
            synthesizer=LLMSynthesizer(
                client=synthesis_client,
                model="fake-synthesis-model",
            ),
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
    assert len(state.search_result_ids) == 1
    assert len(state.document_ids) == 1
    assert len(state.evidence_ids) == 1
    assert state.final_answer == "LLM synthesized news summary."

    assert len(evidence_client.requests) == 1
    assert evidence_client.requests[0].model == "fake-evidence-model"
    evidence_prompt = evidence_client.requests[0].messages[1].content
    assert "latest AI news" in evidence_prompt
    assert "Fake result for latest AI news" in evidence_prompt

    assert len(synthesis_client.requests) == 1
    assert synthesis_client.requests[0].model == "fake-synthesis-model"
    synthesis_prompt = synthesis_client.requests[0].messages[1].content
    assert "latest AI news" in synthesis_prompt
    assert "The fake document reports on latest AI news." in synthesis_prompt
    assert "https://example.com/news/fake-result" in synthesis_prompt


def test_news_runtime_with_llm_components() -> None:
    asyncio.run(_run_news_runtime_with_llm())

"""Manual smoke check for the news runtime with a real LLM client.

Retrieval and document reading remain fake. Evidence extraction and synthesis
use OpenAISDKLLMClient.

Run with:
UV_CACHE_DIR=.uv-cache uv run python scripts/real_llm_news_runtime_check.py
"""

import asyncio
import os

from dotenv import load_dotenv

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentRuntime, AgentState, UserQuery
from banso.documents import (
    Document,
    DocumentReadRequest,
    EvidenceItem,
    LLMEvidenceExtractor,
)
from banso.executors import NewsActionExecutor
from banso.llm import (
    ThinkingTagStrippingLLMClient,
    build_external_llm_client_from_env,
    build_vllm_llm_client_from_env,
)
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import FakeRetrievalProvider
from banso.synthesis import LLMSynthesizer
from banso.tracing import InMemoryTraceSink, Tracer


class SampleNewsDocumentReader:
    async def read(self, request: DocumentReadRequest) -> Document:
        return Document(
            url=request.url,
            title=request.title or "Sample AI news report",
            source=request.source,
            text=(
                "OpenAI announced a new AI model on Monday, saying the model "
                "is designed to improve reasoning and coding tasks. The company "
                "said the release will be available to developers through an API. "
                "Analysts said the launch could intensify competition among AI "
                "labs and cloud providers."
            ),
            metadata=request.metadata,
        )


async def main() -> None:
    load_dotenv()

    query = os.getenv("BANSO_NEWS_QUERY", "latest AI news")
    evidence_llm_client = ThinkingTagStrippingLLMClient(
        build_vllm_llm_client_from_env()
    )
    external_llm_client = build_external_llm_client_from_env()
    store = InMemoryArtifactStore()
    trace_sink = InMemoryTraceSink()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=SampleNewsDocumentReader(),
            evidence_extractor=LLMEvidenceExtractor(client=evidence_llm_client),
            synthesizer=LLMSynthesizer(client=external_llm_client),
        ),
        tracer=Tracer(trace_sink),
    )

    output = await runtime.run(AgentState(query=UserQuery(text=query)))
    state = output.result.state

    print("done:", state.done)
    print("trace steps:", len(state.action_history))
    print("actions:", [entry.action_type.value for entry in state.action_history])
    print("search result ids:", state.search_result_ids)
    print("document ids:", state.document_ids)
    print("evidence ids:", state.evidence_ids)
    print("evidence:")
    for index, evidence_id in enumerate(state.evidence_ids, start=1):
        evidence = store.get(evidence_id, EvidenceItem)
        if evidence is None:
            print(f"{index}. missing evidence artifact: {evidence_id}")
            continue
        print(f"{index}. claim: {evidence.claim}")
        print(f"   supporting_text: {evidence.supporting_text}")
        print(f"   confidence: {evidence.confidence}")
        print(f"   source_url: {evidence.source_url}")
    print("final answer:", state.final_answer)
    print("observations:")
    for entry in state.action_history:
        print(f"- {entry.action_type.value}: {entry.observation.data}")


if __name__ == "__main__":
    asyncio.run(main())

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
from banso.llm import OpenAISDKLLMClient
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import FakeRetrievalProvider
from banso.synthesis import LLMSynthesizer


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


def build_llm_client() -> OpenAISDKLLMClient:
    base_url = os.getenv("BANSO_LLM_BASE_URL")
    api_key = os.getenv("BANSO_LLM_API_KEY") or "dummy"
    model = os.getenv("BANSO_LLM_MODEL")
    timeout = float(os.getenv("BANSO_LLM_TIMEOUT_SECONDS", "60"))

    if not model:
        raise RuntimeError("BANSO_LLM_MODEL is required in .env")

    return OpenAISDKLLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )


async def main() -> None:
    load_dotenv()

    query = os.getenv("BANSO_NEWS_QUERY", "latest AI news")
    llm_client = build_llm_client()
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=SampleNewsDocumentReader(),
            evidence_extractor=LLMEvidenceExtractor(client=llm_client),
            synthesizer=LLMSynthesizer(client=llm_client),
        ),
    )

    output = await runtime.run(AgentState(query=UserQuery(text=query)))
    state = output.result.state

    print("done:", state.done)
    print("trace steps:", len(output.trace.steps))
    print("actions:", [step.action.type.value for step in output.trace.steps])
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
    print("final answer:", output.result.final_answer)
    print("observations:")
    for step in output.trace.steps:
        print(f"- {step.action.type.value}: {step.observation.data}")
        if step.observation.error:
            print(f"  error: {step.observation.error}")


if __name__ == "__main__":
    asyncio.run(main())

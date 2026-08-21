"""Manual smoke check for the news runtime with a real LLM client.

Retrieval and document fetching remain fake. Evidence extraction and synthesis
use OpenAISDKLLMClient.

Run with:
UV_CACHE_DIR=.uv-cache uv run python scripts/real_llm_news_runtime_check.py
"""

import asyncio
import os

from dotenv import load_dotenv

from banso.artifacts.store import InMemoryArtifactStore
from banso.core.action import RetrievalRoute
from banso.core.runtime import AgentRuntime
from banso.core.state import AgentState, UserQuery
from banso.documents.fetcher import DocumentFetchRequest
from banso.documents.llm_extractor import LLMEvidenceExtractor
from banso.documents.models import Document, EvidenceItem
from banso.executors.news_executor import NewsActionExecutor
from banso.executors.research_pipeline import ResearchRouteComponents
from banso.llm.config import (
    build_external_llm_client_from_env,
    build_vllm_llm_client_from_env,
)
from banso.llm.openai_sdk_client import ThinkingModeLLMClient
from banso.policies.llm_news_policy import LLMNewsPolicy
from banso.research_context import ResearchContextBuilder
from banso.retrieval.fake import FakeRetrievalProvider
from banso.synthesis.llm_synthesizer import LLMSynthesizer
from banso.tracing.trace import InMemoryTraceSink, Tracer


class SampleNewsDocumentFetcher:
    async def fetch(self, request: DocumentFetchRequest) -> Document:
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
    evidence_llm_client = ThinkingModeLLMClient(
        build_vllm_llm_client_from_env()
    )
    external_llm_client = build_external_llm_client_from_env()
    store = InMemoryArtifactStore()
    trace_sink = InMemoryTraceSink()
    tracer = Tracer(trace_sink)
    runtime = AgentRuntime(
        policy=LLMNewsPolicy(
            evidence_llm_client,
            ResearchContextBuilder(store, [RetrievalRoute.WEB]),
        ),
        executor=NewsActionExecutor(
            store=store,
            research_routes={
                RetrievalRoute.WEB: ResearchRouteComponents(
                    retrieval_provider=FakeRetrievalProvider(),
                    document_fetcher=SampleNewsDocumentFetcher(),
                )
            },
            evidence_extractor=LLMEvidenceExtractor(client=evidence_llm_client),
            synthesizer=LLMSynthesizer(client=external_llm_client),
        ),
        tracer=tracer,
    )

    output = await runtime.run(AgentState(query=UserQuery(text=query)))
    state = output.result.state

    print("done:", state.done)
    print("trace steps:", len(state.action_history))
    print("actions:", [entry.action.type.value for entry in state.action_history])
    print("search result ids:", list(state.search_results))
    print("document ids:", list(state.documents))
    evidence_ids = [
        evidence_id
        for document in state.documents.values()
        for evidence_id in document.evidence_ids
    ]
    print("evidence ids:", evidence_ids)
    print("evidence:")
    for index, evidence_id in enumerate(evidence_ids, start=1):
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
        print(
            f"- {entry.action.type.value}: "
            f"{entry.observation.model_dump(mode='json')}"
        )


if __name__ == "__main__":
    asyncio.run(main())

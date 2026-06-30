"""Manual end-to-end smoke check for the real news runtime path.

This uses real Tavily search, real HTTP document reading, and a real LLM client.

Run with:
UV_CACHE_DIR=.uv-cache uv run python scripts/real_news_runtime_check.py
"""

import asyncio
import os

from dotenv import load_dotenv

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentRuntime, AgentState, UserQuery
from banso.documents import (
    Document,
    EvidenceItem,
    HTTPDocumentReader,
    LLMEvidenceExtractor,
)
from banso.executors import NewsActionExecutor
from banso.llm import OpenAISDKLLMClient
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import SearchResult, TavilyRetrievalProvider
from banso.synthesis import LLMSynthesizer


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


def build_tavily_provider() -> TavilyRetrievalProvider:
    api_key = os.getenv("BANSO_TAVILY_API_KEY")
    base_url = os.getenv("BANSO_TAVILY_BASE_URL", "https://api.tavily.com")

    if not api_key:
        raise RuntimeError("BANSO_TAVILY_API_KEY is required in .env")

    return TavilyRetrievalProvider(
        api_key=api_key,
        base_url=base_url,
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
            retrieval_provider=build_tavily_provider(),
            document_reader=HTTPDocumentReader(),
            evidence_extractor=LLMEvidenceExtractor(client=llm_client),
            synthesizer=LLMSynthesizer(client=llm_client),
        ),
    )

    output = await runtime.run(AgentState(query=UserQuery(text=query)))
    state = output.result.state

    print("done:", state.done)
    print("trace steps:", len(output.trace.steps))
    print("actions:", [step.action.type.value for step in output.trace.steps])
    print("search queries:", state.search_queries)
    print("search result ids:", state.search_result_ids)
    print("document ids:", state.document_ids)
    print("evidence ids:", state.evidence_ids)

    print("search results:")
    for index, result_id in enumerate(state.search_result_ids, start=1):
        result = store.get(result_id, SearchResult)
        if result is None:
            print(f"{index}. missing search result artifact: {result_id}")
            continue
        print(f"{index}. title: {result.title}")
        print(f"   url: {result.url}")
        print(f"   rank: {result.rank}")
        print(f"   score: {result.metadata.get('score')}")
        print(f"   snippet: {result.snippet}")

    print("documents:")
    for index, document_id in enumerate(state.document_ids, start=1):
        document = store.get(document_id, Document)
        if document is None:
            print(f"{index}. missing document artifact: {document_id}")
            continue
        print(f"{index}. title: {document.title}")
        print(f"   url: {document.url}")
        print(f"   text chars: {len(document.text)}")
        print(f"   preview: {document.text[:300]}")

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

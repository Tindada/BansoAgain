"""Manual end-to-end smoke check for the real news runtime path.

This uses real Tavily search, real HTTP document reading, and a real LLM client.

Run with:
UV_CACHE_DIR=.uv-cache uv run python scripts/real_news_runtime_check.py
"""

import argparse
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
from banso.llm import (
    ThinkingTagStrippingLLMClient,
    build_external_llm_client_from_env,
    build_vllm_llm_client_from_env,
)
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import SearchResult, TavilyRetrievalProvider
from banso.synthesis import LLMSynthesizer


def build_tavily_provider() -> TavilyRetrievalProvider:
    api_key = os.getenv("BANSO_TAVILY_API_KEY")
    base_url = os.getenv("BANSO_TAVILY_BASE_URL", "https://api.tavily.com")

    if not api_key:
        raise RuntimeError("BANSO_TAVILY_API_KEY is required in .env")

    return TavilyRetrievalProvider(
        api_key=api_key,
        base_url=base_url,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print artifact IDs, document previews, evidence, and observations",
    )
    return parser.parse_args()


async def main(*, verbose: bool = False) -> None:
    load_dotenv()

    query = os.getenv("BANSO_NEWS_QUERY", "latest AI news")
    max_extraction_concurrency = int(
        os.getenv("BANSO_MAX_EXTRACTION_CONCURRENCY", "3")
    )
    evidence_llm_client = ThinkingTagStrippingLLMClient(
        build_vllm_llm_client_from_env()
    )
    external_llm_client = build_external_llm_client_from_env()
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=build_tavily_provider(),
            document_reader=HTTPDocumentReader(),
            evidence_extractor=LLMEvidenceExtractor(client=evidence_llm_client),
            synthesizer=LLMSynthesizer(client=external_llm_client),
            max_extraction_concurrency=max_extraction_concurrency,
        ),
    )

    output = await runtime.run(AgentState(query=UserQuery(text=query)))
    state = output.result.state

    print("done:", state.done)
    print("trace steps:", len(output.trace.steps))
    print("actions:", [step.action.type.value for step in output.trace.steps])
    print("timings:")
    for step in output.trace.steps:
        duration = step.duration_seconds or 0.0
        print(f"- {step.action.type.value}: {duration:.2f}s")
    total_duration = sum(step.duration_seconds or 0.0 for step in output.trace.steps)
    print(f"total action time: {total_duration:.2f}s")
    print("search queries:", state.search_queries)
    print("search results:", len(state.search_result_ids))
    print("documents:", len(state.document_ids))
    print("evidence items:", len(state.evidence_ids))
    print("final answer:", output.result.final_answer)

    synthesis_step = next(
        (step for step in output.trace.steps if step.action.type.value == "synthesize"),
        None,
    )
    citations = synthesis_step.observation.data.get("citations", []) if synthesis_step else []
    if citations:
        print("citations:")
        for citation in citations:
            print("-", citation)

    if not verbose:
        return

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
        print(f"   extraction strategy: {document.metadata.get('extraction_strategy')}")
        print(f"   raw HTML chars: {document.metadata.get('raw_html_chars')}")
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

    print("observations:")
    for step in output.trace.steps:
        print(f"- {step.action.type.value}: {step.observation.data}")
        if step.observation.error:
            print(f"  error: {step.observation.error}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(verbose=args.verbose))

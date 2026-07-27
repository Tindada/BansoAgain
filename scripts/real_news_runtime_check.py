"""Manual end-to-end smoke check for the real news runtime path.

This uses real Tavily search, real HTTP document reading, and a real LLM client.

Run with:
UV_CACHE_DIR=.uv-cache uv run python scripts/real_news_runtime_check.py
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from banso.apps.real_news import build_real_news_runtime
from banso.core import AgentActionType, AgentState, UserQuery
from banso.documents import Document, EvidenceItem
from banso.retrieval import SearchResult


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
    bundle = build_real_news_runtime()
    runtime = bundle.runtime
    store = bundle.store

    output = await runtime.run(AgentState(query=UserQuery(text=query)))
    state = output.result.state
    spans = bundle.trace_sink.get_trace(output.trace_id)
    action_spans = [
        span for span in spans if span.name == "agent.action.execute"
    ]

    print("done:", state.done)
    print("trace steps:", len(state.action_history))
    print("actions:", [entry.action_type.value for entry in state.action_history])
    print("timings:")
    for span in action_spans:
        print(f"- {span.attributes['action_type']}: {span.duration_seconds:.2f}s")
    total_duration = sum(span.duration_seconds for span in action_spans)
    print(f"total action time: {total_duration:.2f}s")
    search_queries = [
        entry.observation.data["search_queries"][0]
        for entry in state.action_history
        if entry.action_type == AgentActionType.SEARCH
    ]
    print("search queries:", search_queries)
    print("search results:", len(state.search_results))
    print("documents:", len(state.documents))
    print(
        "evidence items:",
        sum(len(document.evidence_ids) for document in state.documents.values()),
    )
    print("final answer:", state.final_answer)

    if state.citations:
        print("citations:")
        for citation in state.citations:
            print("-", citation)

    if not verbose:
        return

    print("search result ids:", list(state.search_results))
    print("document ids:", list(state.documents))
    print(
        "evidence ids:",
        [
            evidence_id
            for document in state.documents.values()
            for evidence_id in document.evidence_ids
        ],
    )

    print("search results:")
    for index, result_id in enumerate(state.search_results, start=1):
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
    for index, document_id in enumerate(state.documents, start=1):
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
    evidence_ids = [
        evidence_id
        for document in state.documents.values()
        for evidence_id in document.evidence_ids
    ]
    for index, evidence_id in enumerate(evidence_ids, start=1):
        evidence = store.get(evidence_id, EvidenceItem)
        if evidence is None:
            print(f"{index}. missing evidence artifact: {evidence_id}")
            continue
        print(f"{index}. claim: {evidence.claim}")
        print(f"   supporting_text: {evidence.supporting_text}")
        print(f"   confidence: {evidence.confidence}")
        print(f"   source_url: {evidence.source_url}")

    print("observations:")
    for entry in state.action_history:
        print(f"- {entry.action_type.value}: {entry.observation.data}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(verbose=args.verbose))

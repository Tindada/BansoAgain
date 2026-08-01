"""Construction helpers for the real news runtime."""

import os
from dataclasses import dataclass
from pathlib import Path

from banso.artifacts import InMemoryArtifactStore
from banso.corpus import (
    CorpusAwareDocumentFetcher,
    CorpusSearchMode,
    LanceCorpusIndex,
    LocalCorpusRetrievalProvider,
    SQLiteCorpusStore,
    SourceRegistry,
)
from banso.corpus.config import build_embedding_provider_from_env
from banso.core import AgentRuntime
from banso.documents import HTTPDocumentFetcher, LLMEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.llm import (
    ThinkingTagStrippingLLMClient,
    TracingLLMClient,
    build_external_llm_client_from_env,
    build_vllm_llm_client_from_env,
)
from banso.policies import (
    LLMNewsPolicy,
    NewsPolicyContextBuilder,
    NewsRuleBasedPolicy,
)
from banso.retrieval import (
    LLMSearchQueryPlanner,
    SourceClassifier,
    SourceClassifierConfig,
    TavilyRetrievalProvider,
)
from banso.synthesis import LLMSynthesizer
from banso.tracing import InMemoryTraceSink, Tracer


@dataclass(frozen=True)
class RealNewsRuntimeBundle:
    """A real runtime and its run-scoped artifact store."""

    runtime: AgentRuntime
    store: InMemoryArtifactStore
    trace_sink: InMemoryTraceSink


def build_tavily_provider_from_env() -> TavilyRetrievalProvider:
    """Build the real retrieval provider from environment variables."""

    api_key = os.getenv("BANSO_TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("BANSO_TAVILY_API_KEY is required in .env")

    return TavilyRetrievalProvider(
        api_key=api_key,
        base_url=os.getenv("BANSO_TAVILY_BASE_URL", "https://api.tavily.com"),
    )


def build_real_news_runtime() -> RealNewsRuntimeBundle:
    """Build a fresh real news runtime from environment variables."""

    policy_name = os.getenv("BANSO_NEWS_POLICY", "rule_based").strip().casefold()
    if policy_name not in {"rule_based", "llm"}:
        raise RuntimeError(
            "BANSO_NEWS_POLICY must be 'rule_based' or 'llm', "
            f"got {policy_name!r}"
        )

    retrieval_provider_name = os.getenv("BANSO_NEWS_RETRIEVAL_PROVIDER", "tavily")
    if retrieval_provider_name not in {"tavily", "local"}:
        raise RuntimeError("BANSO_NEWS_RETRIEVAL_PROVIDER must be 'tavily' or 'local'")

    if retrieval_provider_name == "local":
        corpus_search_mode = CorpusSearchMode(os.getenv("BANSO_CORPUS_SEARCH_MODE", "hybrid"))

    registry = SourceRegistry.load(
        Path(os.getenv("BANSO_CORPUS_REGISTRY_PATH", "config/trusted_sources.json"))
    )
    source_classifier = SourceClassifier(
        SourceClassifierConfig(source_domains=registry.source_type_by_domain())
    )

    local_llm_client = TracingLLMClient(
        ThinkingTagStrippingLLMClient(build_vllm_llm_client_from_env())
    )
    external_llm_client = TracingLLMClient(build_external_llm_client_from_env())
    store = InMemoryArtifactStore()
    trace_sink = InMemoryTraceSink()
    policy = (
        LLMNewsPolicy(
            client=local_llm_client,
            context_builder=NewsPolicyContextBuilder(store),
        )
        if policy_name == "llm"
        else NewsRuleBasedPolicy()
    )
    if retrieval_provider_name == "tavily":
        retrieval_provider = build_tavily_provider_from_env()
        document_fetcher = HTTPDocumentFetcher()
    else:
        index = LanceCorpusIndex(
            Path(os.getenv("BANSO_CORPUS_INDEX_PATH", "data/corpus.lance")),
            embedding_provider=(
                None
                if corpus_search_mode == CorpusSearchMode.BM25
                else build_embedding_provider_from_env()
            ),
        )
        corpus_store = SQLiteCorpusStore(
            Path(os.getenv("BANSO_CORPUS_DATABASE_PATH", "data/corpus.sqlite3"))
        )
        retrieval_provider = LocalCorpusRetrievalProvider(
            index,
            corpus_store,
            registry,
            mode=corpus_search_mode,
        )
        document_fetcher = CorpusAwareDocumentFetcher(
            corpus_store,
            HTTPDocumentFetcher(),
        )

    runtime = AgentRuntime(
        policy=policy,
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=retrieval_provider,
            document_fetcher=document_fetcher,
            evidence_extractor=LLMEvidenceExtractor(client=local_llm_client),
            synthesizer=LLMSynthesizer(client=external_llm_client),
            search_query_planner=LLMSearchQueryPlanner(client=external_llm_client),
            source_classifier=source_classifier,
            max_extraction_concurrency=int(
                os.getenv("BANSO_MAX_EXTRACTION_CONCURRENCY", "3")
            ),
        ),
        tracer=Tracer(trace_sink),
    )
    return RealNewsRuntimeBundle(
        runtime=runtime,
        store=store,
        trace_sink=trace_sink,
    )

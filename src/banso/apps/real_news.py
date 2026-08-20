"""Construction helpers for the real news runtime."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from banso.core import AgentRuntime, RetrievalRoute
from banso.documents import HTTPDocumentFetcher, LLMEvidenceExtractor
from banso.executors import NewsActionExecutor, ResearchRouteComponents
from banso.llm import (
    ThinkingModeLLMClient,
    TracingLLMClient,
    build_external_llm_client_from_env,
    build_vllm_llm_client_from_env,
)
from banso.policies import LLMNewsPolicy, NewsPolicyContextBuilder
from banso.retrieval import (
    SourceClassifier,
    SourceClassifierConfig,
    TavilyRetrievalProvider,
)
from banso.synthesis import LLMSynthesizer, Synthesizer
from banso.tracing import InMemoryTraceSink, Tracer


@dataclass(frozen=True)
class RealNewsRuntimeBundle:
    """A real runtime and its run-scoped artifact store."""

    runtime: AgentRuntime
    store: InMemoryArtifactStore
    trace_sink: InMemoryTraceSink


def build_tavily_provider_from_env() -> TavilyRetrievalProvider:
    """Build the web retrieval provider from environment variables."""
    api_key = os.getenv("BANSO_TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("BANSO_TAVILY_API_KEY is required in .env")
    return TavilyRetrievalProvider(
        api_key=api_key,
        base_url=os.getenv("BANSO_TAVILY_BASE_URL", "https://api.tavily.com"),
    )


def enabled_retrieval_routes_from_env() -> list[RetrievalRoute]:
    """Parse the explicitly enabled semantic retrieval routes."""
    value = os.getenv("BANSO_NEWS_RETRIEVAL_ROUTES", "web").strip().casefold()
    allowed = {
        "web": [RetrievalRoute.WEB],
        "local": [RetrievalRoute.LOCAL],
        "local,web": [RetrievalRoute.LOCAL, RetrievalRoute.WEB],
    }
    try:
        return allowed[value]
    except KeyError as error:
        raise RuntimeError(
            "BANSO_NEWS_RETRIEVAL_ROUTES must be 'web', 'local', or 'local,web'"
        ) from error


def build_real_news_runtime(
    *,
    synthesizer_class: type[Synthesizer] = LLMSynthesizer,
    extraction_thinking_extra_body: dict[str, Any] | None = None,
) -> RealNewsRuntimeBundle:
    """Build a fresh LLM-policy news runtime from environment variables."""
    enabled_routes = enabled_retrieval_routes_from_env()
    registry = SourceRegistry.load(
        Path(os.getenv("BANSO_CORPUS_REGISTRY_PATH", "config/trusted_sources.json"))
    )
    source_classifier = SourceClassifier(
        SourceClassifierConfig(source_domains=registry.source_type_by_domain())
    )

    base_local_llm_client = build_vllm_llm_client_from_env()
    policy_llm_client = TracingLLMClient(
        ThinkingModeLLMClient(base_local_llm_client)
    )
    extraction_llm_client = TracingLLMClient(
        ThinkingModeLLMClient(
            base_local_llm_client,
            thinking_extra_body=extraction_thinking_extra_body,
        )
    )
    external_llm_client = TracingLLMClient(build_external_llm_client_from_env())
    store = InMemoryArtifactStore()
    trace_sink = InMemoryTraceSink()
    tracer = Tracer(trace_sink)
    research_routes: dict[RetrievalRoute, ResearchRouteComponents] = {}

    if RetrievalRoute.WEB in enabled_routes:
        research_routes[RetrievalRoute.WEB] = ResearchRouteComponents(
            retrieval_provider=build_tavily_provider_from_env(),
            document_fetcher=HTTPDocumentFetcher(),
        )

    if RetrievalRoute.LOCAL in enabled_routes:
        corpus_search_mode = CorpusSearchMode(
            os.getenv("BANSO_CORPUS_SEARCH_MODE", "vector")
        )
        corpus_store = SQLiteCorpusStore(
            Path(os.getenv("BANSO_CORPUS_DATABASE_PATH", "data/corpus.sqlite3"))
        )
        index = LanceCorpusIndex(
            Path(os.getenv("BANSO_CORPUS_INDEX_PATH", "data/corpus.lance")),
            embedding_provider=(
                None
                if corpus_search_mode == CorpusSearchMode.BM25
                else build_embedding_provider_from_env()
            ),
        )
        research_routes[RetrievalRoute.LOCAL] = ResearchRouteComponents(
            retrieval_provider=LocalCorpusRetrievalProvider(
                index,
                corpus_store,
                registry,
                mode=corpus_search_mode,
            ),
            document_fetcher=CorpusAwareDocumentFetcher(
                corpus_store,
                HTTPDocumentFetcher(),
            ),
        )

    policy = LLMNewsPolicy(
        client=policy_llm_client,
        context_builder=NewsPolicyContextBuilder(store, enabled_routes),
    )
    runtime = AgentRuntime(
        policy=policy,
        executor=NewsActionExecutor(
            store=store,
            research_routes=research_routes,
            evidence_extractor=LLMEvidenceExtractor(client=extraction_llm_client),
            synthesizer=synthesizer_class(client=external_llm_client),
            source_classifier=source_classifier,
            max_extraction_concurrency=int(
                os.getenv("BANSO_MAX_EXTRACTION_CONCURRENCY", "4")
            ),
        ),
        tracer=tracer,
    )
    return RealNewsRuntimeBundle(
        runtime=runtime,
        store=store,
        trace_sink=trace_sink,
    )

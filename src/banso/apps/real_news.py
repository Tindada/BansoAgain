"""Construction helpers for the real news runtime."""

import os
from dataclasses import dataclass
from pathlib import Path

from banso.artifacts.store import InMemoryArtifactStore
from banso.corpus.agent import (
    CorpusAwareDocumentFetcher,
    LocalCorpusRetrievalProvider,
)
from banso.corpus.config import build_embedding_provider_from_env
from banso.corpus.indexing.index import CorpusSearchMode, LanceCorpusIndex
from banso.corpus.ingestion.registry import SourceRegistry
from banso.corpus.sqlite_store import SQLiteCorpusStore
from banso.agent.action import RetrievalRoute
from banso.agent.runtime import AgentRuntime
from banso.documents.http_fetcher import HTTPDocumentFetcher
from banso.documents.fetcher import DocumentFetcher
from banso.documents.jina_fetcher import JinaDocumentFetcher
from banso.documents.llm_extractor import LLMEvidenceExtractor
from banso.agent.executors.news_executor import NewsActionExecutor
from banso.agent.executors.research_pipeline import ResearchRouteComponents
from banso.llm.client import LLMClient
from banso.llm.config import (
    build_external_llm_client_from_env,
    build_vllm_llm_client_from_env,
    extraction_llm_extra_body_from_env,
)
from banso.llm.openai_sdk_client import ThinkingModeLLMClient
from banso.llm.tracing import TracingLLMClient
from banso.agent.policies.llm_atomic_policy import LLMAtomicPolicy
from banso.agent.policies.llm_news_policy import LLMNewsPolicy
from banso.agent.research_context import ResearchContextBuilder
from banso.agent.selection.llm_selector import LLMSearchResultSelector
from banso.retrieval.source_classifier import (
    SourceClassifier,
    SourceClassifierConfig,
)
from banso.retrieval.tavily_provider import TavilyRetrievalProvider
from banso.notes.llm_rewriter import LLMNotesRewriter
from banso.synthesis.synthesizer import Synthesizer
from banso.synthesis.llm_synthesizer import LLMSynthesizer
from banso.tracing.trace import InMemoryTraceSink, Tracer


@dataclass(frozen=True)
class RealNewsRuntimeBundle:
    """A real runtime and its run-scoped artifact store."""

    runtime: AgentRuntime
    store: InMemoryArtifactStore
    trace_sink: InMemoryTraceSink


DEFAULT_NEWS_POLICY = "search_read"


def build_tavily_provider_from_env() -> TavilyRetrievalProvider:
    """Build the web retrieval provider from environment variables."""
    api_key = os.getenv("BANSO_TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("BANSO_TAVILY_API_KEY is required in .env")
    return TavilyRetrievalProvider(
        api_key=api_key,
        base_url=os.getenv("BANSO_TAVILY_BASE_URL", "https://api.tavily.com"),
    )


def build_document_fetcher_from_env() -> DocumentFetcher:
    """Build the remote document fetcher selected by environment variables."""
    provider = os.getenv("BANSO_DOCUMENT_FETCHER", "http").strip().casefold()
    if provider == "http":
        return HTTPDocumentFetcher()
    elif provider == "jina":
        return JinaDocumentFetcher(api_key=os.getenv("BANSO_JINA_API_KEY"))
    else:
        raise RuntimeError("BANSO_DOCUMENT_FETCHER must be 'http' or 'jina'")


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


def build_news_policy_from_env(
    client: LLMClient,
    context_builder: ResearchContextBuilder,
) -> LLMNewsPolicy:
    """Build the configured atomic or search/read policy."""
    value = os.getenv("BANSO_NEWS_POLICY", DEFAULT_NEWS_POLICY).strip().casefold()
    policy_types = {
        "atomic": LLMAtomicPolicy,
        "search_read": LLMNewsPolicy,
    }
    try:
        policy_type = policy_types[value]
    except KeyError as error:
        raise RuntimeError(
            "BANSO_NEWS_POLICY must be 'atomic' or 'search_read'"
        ) from error
    return policy_type(client=client, context_builder=context_builder)


def build_real_news_runtime(
    *,
    synthesizer_class: type[Synthesizer] = LLMSynthesizer,
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
    agent_llm_client = TracingLLMClient(
        ThinkingModeLLMClient(base_local_llm_client)
    )
    extraction_llm_client = TracingLLMClient(
        ThinkingModeLLMClient(
            base_local_llm_client,
            request_extra_body=extraction_llm_extra_body_from_env(),
        )
    )
    external_llm_client = TracingLLMClient(build_external_llm_client_from_env())
    store = InMemoryArtifactStore()
    trace_sink = InMemoryTraceSink()
    tracer = Tracer(trace_sink)
    research_routes: dict[RetrievalRoute, ResearchRouteComponents] = {}
    document_fetcher = build_document_fetcher_from_env()

    if RetrievalRoute.WEB in enabled_routes:
        research_routes[RetrievalRoute.WEB] = ResearchRouteComponents(
            retrieval_provider=build_tavily_provider_from_env(),
            document_fetcher=document_fetcher,
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
                document_fetcher,
            ),
        )

    context_builder = ResearchContextBuilder(store, enabled_routes)
    policy = build_news_policy_from_env(agent_llm_client, context_builder)
    runtime = AgentRuntime(
        policy=policy,
        executor=NewsActionExecutor(
            store=store,
            research_routes=research_routes,
            evidence_extractor=LLMEvidenceExtractor(client=extraction_llm_client),
            synthesizer=synthesizer_class(client=external_llm_client),
            source_classifier=source_classifier,
            search_result_selector=LLMSearchResultSelector(
                agent_llm_client,
                context_builder,
            ),
            max_extraction_concurrency=int(
                os.getenv("BANSO_MAX_EXTRACTION_CONCURRENCY", "4")
            ),
            notes_rewriter=LLMNotesRewriter(agent_llm_client),
        ),
        tracer=tracer,
    )
    return RealNewsRuntimeBundle(
        runtime=runtime,
        store=store,
        trace_sink=trace_sink,
    )

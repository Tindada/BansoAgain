"""Construction helpers for the real news runtime."""

import os
from dataclasses import dataclass

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentRuntime
from banso.documents import HTTPDocumentReader, LLMEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.llm import (
    ThinkingTagStrippingLLMClient,
    build_external_llm_client_from_env,
    build_vllm_llm_client_from_env,
)
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import TavilyRetrievalProvider
from banso.synthesis import LLMSynthesizer


@dataclass(frozen=True)
class RealNewsRuntimeBundle:
    """A real runtime and its run-scoped artifact store."""

    runtime: AgentRuntime
    store: InMemoryArtifactStore


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

    evidence_llm_client = ThinkingTagStrippingLLMClient(
        build_vllm_llm_client_from_env()
    )
    external_llm_client = build_external_llm_client_from_env()
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=build_tavily_provider_from_env(),
            document_reader=HTTPDocumentReader(),
            evidence_extractor=LLMEvidenceExtractor(client=evidence_llm_client),
            synthesizer=LLMSynthesizer(client=external_llm_client),
            max_extraction_concurrency=int(
                os.getenv("BANSO_MAX_EXTRACTION_CONCURRENCY", "3")
            ),
        ),
    )
    return RealNewsRuntimeBundle(runtime=runtime, store=store)

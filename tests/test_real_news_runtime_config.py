"""Tests for real news runtime route configuration."""

import pytest

import banso.apps.real_news as real_news
from banso.apps.real_news import enabled_retrieval_routes_from_env
from banso.agent.action import RetrievalRoute
from banso.agent.executors.news_executor import NewsActionExecutor
from banso.agent.policies.llm_news_policy import LLMNewsPolicy
from banso.agent.selection.llm_selector import LLMSearchResultSelector


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, [RetrievalRoute.WEB]),
        ("web", [RetrievalRoute.WEB]),
        ("local", [RetrievalRoute.LOCAL]),
        ("local,web", [RetrievalRoute.LOCAL, RetrievalRoute.WEB]),
    ],
)
def test_enabled_retrieval_routes(value, expected, monkeypatch) -> None:
    if value is None:
        monkeypatch.delenv("BANSO_NEWS_RETRIEVAL_ROUTES", raising=False)
    else:
        monkeypatch.setenv("BANSO_NEWS_RETRIEVAL_ROUTES", value)

    assert enabled_retrieval_routes_from_env() == expected


@pytest.mark.parametrize("value", ["", "tavily", "web,local", "local,local"])
def test_enabled_retrieval_routes_rejects_unknown_shapes(value, monkeypatch) -> None:
    monkeypatch.setenv("BANSO_NEWS_RETRIEVAL_ROUTES", value)

    with pytest.raises(RuntimeError, match="must be"):
        enabled_retrieval_routes_from_env()


class _Registry:
    def source_type_by_domain(self):
        return {}


class _LLMClient:
    async def generate(self, request):
        raise AssertionError("LLM should not be called while building the runtime")


def _patch_common_runtime_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(real_news.SourceRegistry, "load", lambda path: _Registry())
    monkeypatch.setattr(real_news, "build_vllm_llm_client_from_env", _LLMClient)
    monkeypatch.setattr(real_news, "build_external_llm_client_from_env", _LLMClient)


def test_web_only_runtime_builds_only_the_web_route(monkeypatch) -> None:
    _patch_common_runtime_dependencies(monkeypatch)
    monkeypatch.setenv("BANSO_NEWS_RETRIEVAL_ROUTES", "web")
    web_provider = object()
    monkeypatch.setattr(
        real_news,
        "build_tavily_provider_from_env",
        lambda: web_provider,
    )
    monkeypatch.setattr(
        real_news,
        "LanceCorpusIndex",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local route must not be initialized")
        ),
    )

    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setenv(
        "BANSO_EXTRACTION_LLM_EXTRA_BODY",
        '{"chat_template_kwargs":{"enable_thinking":false}}',
    )
    bundle = real_news.build_real_news_runtime()

    assert isinstance(bundle.runtime.policy, LLMNewsPolicy)
    assert isinstance(bundle.runtime.executor, NewsActionExecutor)
    selector = bundle.runtime.executor.research_pipeline.search_result_selector
    assert isinstance(selector, LLMSearchResultSelector)
    assert selector.client is bundle.runtime.policy.client
    assert selector.context_builder is bundle.runtime.policy.context_builder
    extractor = bundle.runtime.executor.research_pipeline.evidence_extractor
    assert extractor.client.client.request_extra_body == extra_body
    assert set(bundle.runtime.executor.research_routes) == {RetrievalRoute.WEB}
    assert (
        bundle.runtime.executor.research_routes[RetrievalRoute.WEB].retrieval_provider
        is web_provider
    )


def test_local_only_runtime_does_not_require_or_build_tavily(monkeypatch) -> None:
    _patch_common_runtime_dependencies(monkeypatch)
    monkeypatch.setenv("BANSO_NEWS_RETRIEVAL_ROUTES", "local")
    monkeypatch.setenv("BANSO_CORPUS_SEARCH_MODE", "bm25")
    monkeypatch.delenv("BANSO_TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        real_news,
        "build_tavily_provider_from_env",
        lambda: (_ for _ in ()).throw(
            AssertionError("web route must not be initialized")
        ),
    )
    monkeypatch.setattr(real_news, "SQLiteCorpusStore", lambda path: "store")
    monkeypatch.setattr(
        real_news,
        "LanceCorpusIndex",
        lambda path, embedding_provider: "index",
    )
    monkeypatch.setattr(
        real_news,
        "LocalCorpusRetrievalProvider",
        lambda *args, **kwargs: "local-provider",
    )
    monkeypatch.setattr(
        real_news,
        "CorpusAwareDocumentFetcher",
        lambda *args: "local-fetcher",
    )

    bundle = real_news.build_real_news_runtime()

    assert set(bundle.runtime.executor.research_routes) == {RetrievalRoute.LOCAL}

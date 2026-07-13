"""Search and retrieval provider interfaces."""

from banso.retrieval.fake import FakeRetrievalProvider
from banso.retrieval.evaluator import (
    SearchResultEvaluation,
    SearchResultEvaluationResult,
    SearchResultEvaluator,
    SearchResultEvaluatorConfig,
)
from banso.retrieval.filter import (
    RetrievalFilter,
    RetrievalFilterConfig,
    RetrievalFilterReport,
    RetrievalFilterResult,
    normalize_url,
)
from banso.retrieval.models import SearchResult, Source, SourceType
from banso.retrieval.provider import RetrievalProvider, SearchRequest
from banso.retrieval.tavily_provider import TavilyRetrievalProvider

__all__ = [
    "FakeRetrievalProvider",
    "RetrievalFilter",
    "RetrievalFilterConfig",
    "RetrievalFilterReport",
    "RetrievalFilterResult",
    "RetrievalProvider",
    "SearchRequest",
    "SearchResult",
    "SearchResultEvaluation",
    "SearchResultEvaluationResult",
    "SearchResultEvaluator",
    "SearchResultEvaluatorConfig",
    "Source",
    "SourceType",
    "TavilyRetrievalProvider",
    "normalize_url",
]

"""Search and retrieval provider interfaces."""

from banso.retrieval.fake import FakeRetrievalProvider
from banso.retrieval.filter import (
    RetrievalFilter,
    RetrievalFilterConfig,
    RetrievalFilterReport,
    RetrievalFilterResult,
)
from banso.retrieval.models import SearchResult, Source, SourceType
from banso.retrieval.provider import RetrievalProvider, SearchRequest
from banso.retrieval.source_classifier import (
    SourceClassification,
    SourceClassificationResult,
    SourceClassifier,
    SourceClassifierConfig,
)
from banso.retrieval.tavily_provider import TavilyRetrievalProvider
from banso.retrieval.url_utils import normalize_url

__all__ = [
    "FakeRetrievalProvider",
    "RetrievalFilter",
    "RetrievalFilterConfig",
    "RetrievalFilterReport",
    "RetrievalFilterResult",
    "RetrievalProvider",
    "SearchRequest",
    "SearchResult",
    "Source",
    "SourceClassification",
    "SourceClassificationResult",
    "SourceClassifier",
    "SourceClassifierConfig",
    "SourceType",
    "TavilyRetrievalProvider",
    "normalize_url",
]

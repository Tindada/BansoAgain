"""Document fetching, ranking, and evidence extraction."""

from banso.documents.extractor import (
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceExtractor,
)
from banso.documents.fake import FakeDocumentFetcher, FakeEvidenceExtractor
from banso.documents.http_fetcher import HTTPDocumentFetcher
from banso.documents.llm_extractor import LLMEvidenceExtractor
from banso.documents.models import Document, EvidenceItem
from banso.documents.parser import (
    DocumentParseError,
    DocumentParser,
    ParsedDocument,
)
from banso.documents.fetcher import (
    DocumentFetchError,
    DocumentFetcher,
    DocumentFetchRequest,
)

__all__ = [
    "Document",
    "DocumentFetchError",
    "DocumentFetcher",
    "DocumentFetchRequest",
    "DocumentParseError",
    "DocumentParser",
    "EvidenceExtractionError",
    "EvidenceExtractionRequest",
    "EvidenceExtractor",
    "EvidenceItem",
    "FakeDocumentFetcher",
    "FakeEvidenceExtractor",
    "HTTPDocumentFetcher",
    "LLMEvidenceExtractor",
    "ParsedDocument",
]

"""Document fetching and evidence extraction contracts."""

from banso.documents.extractor import (
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceExtractor,
)
from banso.documents.models import Document, EvidenceItem
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
    "EvidenceExtractionError",
    "EvidenceExtractionRequest",
    "EvidenceExtractor",
    "EvidenceItem",
]

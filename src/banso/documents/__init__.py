"""Document fetching and evidence extraction contracts."""

from banso.documents.extractor import (
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceExtractor,
)
from banso.documents.models import Document, DocumentEvidence
from banso.documents.fetcher import (
    DocumentFetchError,
    DocumentFetcher,
    DocumentFetchRequest,
)

__all__ = [
    "Document",
    "DocumentEvidence",
    "DocumentFetchError",
    "DocumentFetcher",
    "DocumentFetchRequest",
    "EvidenceExtractionError",
    "EvidenceExtractionRequest",
    "EvidenceExtractor",
]

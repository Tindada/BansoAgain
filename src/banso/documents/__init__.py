"""Document reading, ranking, and evidence extraction."""

from banso.documents.extractor import (
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceExtractor,
)
from banso.documents.fake import FakeDocumentReader, FakeEvidenceExtractor
from banso.documents.http_reader import HTTPDocumentReader
from banso.documents.llm_extractor import LLMEvidenceExtractor
from banso.documents.models import Document, EvidenceItem
from banso.documents.reader import (
    DocumentReadError,
    DocumentReader,
    DocumentReadRequest,
)

__all__ = [
    "Document",
    "DocumentReadError",
    "DocumentReader",
    "DocumentReadRequest",
    "EvidenceExtractionError",
    "EvidenceExtractionRequest",
    "EvidenceExtractor",
    "EvidenceItem",
    "FakeDocumentReader",
    "FakeEvidenceExtractor",
    "HTTPDocumentReader",
    "LLMEvidenceExtractor",
]

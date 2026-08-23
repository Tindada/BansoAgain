"""Fake document components for local smoke tests."""

from banso.documents.extractor import EvidenceExtractionRequest
from banso.documents.models import Document
from banso.documents.fetcher import DocumentFetchRequest


class FakeDocumentFetcher:
    """Returns deterministic documents without fetching remote content."""

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        title = request.title or "Fake document"
        return Document(
            url=request.url,
            title=title,
            text=f"This is a fake document body for {title}.",
            source=request.source,
        )


class FakeEvidenceExtractor:
    """Returns deterministic evidence text without LLM extraction."""

    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> str:
        return _build_fake_evidence(request.query, request.document)


def _build_fake_evidence(query: str, document: Document) -> str:
    return f"Fake evidence for '{query}' from '{document.title}'."

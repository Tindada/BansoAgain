"""Fake document components for local smoke tests."""

from banso.core.state import UserQuery
from banso.documents.extractor import EvidenceExtractionRequest
from banso.documents.models import Document, EvidenceItem
from banso.documents.reader import DocumentReadRequest


class FakeDocumentReader:
    """Returns deterministic documents without fetching remote content."""

    async def read(self, request: DocumentReadRequest) -> Document:
        title = request.title or "Fake document"
        return Document(
            url=request.url,
            title=title,
            text=f"This is a fake document body for {title}.",
            source=request.source,
        )


class FakeEvidenceExtractor:
    """Returns deterministic evidence items without LLM extraction."""

    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        claim = _build_fake_claim(request.query, request.document)
        return [
            EvidenceItem(
                document_id=request.document.id,
                claim=claim,
                supporting_text=request.document.text,
                source_url=request.document.url,
                published_at=request.document.published_at,
                confidence=1.0,
            )
        ][: request.max_items_per_chunk]


def _build_fake_claim(query: UserQuery, document: Document) -> str:
    return f"Fake evidence for '{query.text}' from '{document.title}'."

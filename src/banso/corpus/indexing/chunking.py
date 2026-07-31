"""Deterministic paragraph-aware chunking for corpus documents."""

from dataclasses import dataclass
from datetime import datetime

from banso.corpus.models import CorpusDocument, CorpusDocumentStatus


@dataclass(frozen=True)
class CorpusChunk:
    """A searchable text chunk with its document provenance."""

    id: str
    document_id: str
    source_id: str
    url: str
    title: str | None
    published_at: datetime | None
    content_hash: str
    position: int
    text: str


def chunk_document(
    document: CorpusDocument,
    *,
    max_chars: int = 1200,
) -> tuple[CorpusChunk, ...]:
    """Split an active document without crossing the configured size limit."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if document.status != CorpusDocumentStatus.ACTIVE:
        raise ValueError("only active corpus documents can be chunked")
    if document.text is None or document.content_hash is None:
        raise ValueError("active corpus document is missing text or content hash")

    paragraphs = (
        segment
        for paragraph in map(str.strip, document.text.splitlines())
        if paragraph
        for segment in _split_oversized(paragraph, max_chars=max_chars)
    )
    texts: list[str] = []
    current: list[str] = []
    current_chars = 0
    for paragraph in paragraphs:
        separator_chars = 2 if current else 0
        if current and current_chars + separator_chars + len(paragraph) > max_chars:
            texts.append("\n\n".join(current))
            current = []
            current_chars = 0
            separator_chars = 0
        current.append(paragraph)
        current_chars += separator_chars + len(paragraph)
    if current:
        texts.append("\n\n".join(current))

    return tuple(
        CorpusChunk(
            id=f"{document.id}:{position}",
            document_id=document.id,
            source_id=document.source_id,
            url=document.url,
            title=document.title,
            published_at=document.published_at,
            content_hash=document.content_hash,
            position=position,
            text=text,
        )
        for position, text in enumerate(texts)
    )


def _split_oversized(text: str, *, max_chars: int) -> tuple[str, ...]:
    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return tuple(parts)

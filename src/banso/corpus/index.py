"""Rebuildable LanceDB BM25 index for the local corpus."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.index import FTS

from banso.corpus.chunking import CorpusChunk, chunk_document
from banso.corpus.models import CorpusDocumentStatus
from banso.corpus.sqlite_store import SQLiteCorpusStore

_TABLE_NAME = "corpus_chunks"
_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("title", pa.string()),
        pa.field("published_at", pa.timestamp("us", tz="UTC")),
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class CorpusSearchResult:
    """A BM25-ranked corpus chunk."""

    chunk: CorpusChunk
    score: float


class LanceCorpusIndex:
    """Full-text index derived entirely from the SQLite corpus."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_chunk_chars: int = 1200,
    ) -> None:
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be greater than zero")
        self._database = lancedb.connect(str(path))
        self._max_chunk_chars = max_chunk_chars

    def rebuild(self, store: SQLiteCorpusStore) -> int:
        """Replace the index with chunks from all active corpus documents."""

        chunks = tuple(
            chunk
            for document in store.list_documents(
                status=CorpusDocumentStatus.ACTIVE
            )
            for chunk in chunk_document(
                document,
                max_chars=self._max_chunk_chars,
            )
        )
        data = pa.Table.from_pylist(
            [_chunk_to_record(chunk) for chunk in chunks],
            schema=_SCHEMA,
        )
        table = self._database.create_table(
            _TABLE_NAME,
            data=data,
            mode="overwrite",
        )
        table.create_index("text", config=FTS(), replace=True)
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[CorpusSearchResult, ...]:
        """Return BM25-ranked chunks for a non-blank query."""

        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        rows = (
            self._database.open_table(_TABLE_NAME)
            .search(query, query_type="fts", fts_columns="text")
            .limit(limit)
            .to_list()
        )
        return tuple(
            CorpusSearchResult(
                chunk=_record_to_chunk(row),
                score=float(row["_score"]),
            )
            for row in rows
        )


def _chunk_to_record(chunk: CorpusChunk) -> dict[str, object]:
    published_at = chunk.published_at
    if published_at is not None:
        published_at = published_at.astimezone(timezone.utc)
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "source_id": chunk.source_id,
        "url": chunk.url,
        "title": chunk.title,
        "published_at": published_at,
        "content_hash": chunk.content_hash,
        "position": chunk.position,
        "text": chunk.text,
    }


def _record_to_chunk(record: dict[str, object]) -> CorpusChunk:
    return CorpusChunk(
        id=str(record["id"]),
        document_id=str(record["document_id"]),
        source_id=str(record["source_id"]),
        url=str(record["url"]),
        title=str(record["title"]) if record["title"] is not None else None,
        published_at=record["published_at"],
        content_hash=str(record["content_hash"]),
        position=record["position"],
        text=str(record["text"]),
    )

"""Rebuildable LanceDB hybrid index for the local corpus."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Sequence

import lancedb
import pyarrow as pa
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from banso.corpus.chunking import CorpusChunk, chunk_document
from banso.corpus.embeddings import EmbeddingProvider
from banso.corpus.models import CorpusDocumentStatus
from banso.corpus.sqlite_store import SQLiteCorpusStore

_TABLE_NAME = "corpus_chunks"
_EMBEDDING_MODEL_METADATA = b"banso.embedding_model"
_EMBEDDING_DIMENSIONS_METADATA = b"banso.embedding_dimensions"
_CHUNK_FIELDS = [
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


class CorpusSearchMode(StrEnum):
    """Available local corpus ranking strategies."""

    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class CorpusSearchResult:
    """A ranked corpus chunk whose score increases with relevance."""

    chunk: CorpusChunk
    score: float


class LanceCorpusIndex:
    """Text and vector index derived entirely from the SQLite corpus."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        max_chunk_chars: int = 1200,
    ) -> None:
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be greater than zero")
        self._database = lancedb.connect(str(path))
        self._embedding_provider = embedding_provider
        self._max_chunk_chars = max_chunk_chars

    def rebuild(self, store: SQLiteCorpusStore) -> int:
        """Replace the index and reuse unchanged compatible vectors."""

        provider = self._require_embedding_provider()
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
        reusable_vectors = self._load_reusable_vectors(provider)
        vectors: list[tuple[float, ...] | None] = []
        missing_texts: list[str] = []
        for chunk in chunks:
            vector = reusable_vectors.get((chunk.id, chunk.content_hash))
            vectors.append(vector)
            if vector is None:
                missing_texts.append(chunk.text)

        generated_vectors = iter(provider.embed_documents(missing_texts))
        complete_vectors = tuple(
            vector
            if vector is not None
            else next(generated_vectors)
            for vector in vectors
        )

        data = pa.Table.from_pylist(
            [
                _chunk_to_record(chunk, vector=vector)
                for chunk, vector in zip(chunks, complete_vectors, strict=True)
            ],
            schema=_schema_for(provider),
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
        mode: CorpusSearchMode = CorpusSearchMode.HYBRID,
    ) -> tuple[CorpusSearchResult, ...]:
        """Return chunks ranked by the selected retrieval mode."""

        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        mode = CorpusSearchMode(mode)
        table = self._database.open_table(_TABLE_NAME)
        if mode == CorpusSearchMode.BM25:
            rows = (
                table.search(query, query_type="fts", fts_columns="text")
                .limit(limit)
                .to_list()
            )
        else:
            provider = self._require_embedding_provider()
            if not _embedding_metadata_matches(table.schema, provider):
                raise ValueError(
                    "embedding provider model or dimensions do not match "
                    "the corpus index"
                )
            query_vector = provider.embed_query(query)
            if mode == CorpusSearchMode.VECTOR:
                rows = (
                    table.search(
                        list(query_vector),
                        vector_column_name="vector",
                        query_type="vector",
                    )
                    .distance_type("cosine")
                    .limit(limit)
                    .to_list()
                )
            else:
                rows = (
                    table.search(
                        query_type="hybrid",
                        vector_column_name="vector",
                        fts_columns="text",
                    )
                    .vector(list(query_vector))
                    .text(query)
                    .distance_type("cosine")
                    .rerank(RRFReranker(K=60))
                    .limit(limit)
                    .to_list()
                )
        return tuple(
            CorpusSearchResult(
                chunk=_record_to_chunk(row),
                score=_result_score(row, mode=mode),
            )
            for row in rows
        )

    def _require_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            raise RuntimeError(
                "an embedding provider is required for index rebuild, "
                "vector search, and hybrid search"
            )
        return self._embedding_provider

    def _load_reusable_vectors(
        self,
        provider: EmbeddingProvider,
    ) -> dict[tuple[str, str], tuple[float, ...]]:
        if _TABLE_NAME not in self._database.list_tables().tables:
            return {}
        table = self._database.open_table(_TABLE_NAME)
        if not _embedding_metadata_matches(table.schema, provider):
            return {}

        vectors: dict[tuple[str, str], tuple[float, ...]] = {}
        rows = table.to_arrow().select(["id", "content_hash", "vector"]).to_pylist()
        for row in rows:
            vectors[(str(row["id"]), str(row["content_hash"]))] = tuple(
                row["vector"]
            )
        return vectors


def _schema_for(provider: EmbeddingProvider) -> pa.Schema:
    return pa.schema(
        [
            *_CHUNK_FIELDS,
            pa.field(
                "vector",
                pa.list_(pa.float32(), provider.dimensions),
                nullable=False,
            ),
        ],
        metadata={
            _EMBEDDING_MODEL_METADATA: provider.model.encode("utf-8"),
            _EMBEDDING_DIMENSIONS_METADATA: str(provider.dimensions).encode("ascii"),
        },
    )


def _chunk_to_record(
    chunk: CorpusChunk,
    *,
    vector: Sequence[float],
) -> dict[str, object]:
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
        "vector": vector,
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


def _embedding_metadata_matches(
    schema: pa.Schema,
    provider: EmbeddingProvider,
) -> bool:
    metadata = schema.metadata or {}
    return (
        metadata.get(_EMBEDDING_MODEL_METADATA) == provider.model.encode("utf-8")
        and metadata.get(_EMBEDDING_DIMENSIONS_METADATA)
        == str(provider.dimensions).encode("ascii")
    )


def _result_score(
    record: dict[str, object],
    *,
    mode: CorpusSearchMode,
) -> float:
    if mode == CorpusSearchMode.BM25:
        return float(record["_score"])
    if mode == CorpusSearchMode.VECTOR:
        return 1.0 - float(record["_distance"])
    return float(record["_relevance_score"])

"""Tests for corpus chunking and the rebuildable LanceDB hybrid index."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest

from banso.corpus.indexing.chunking import chunk_document
from banso.corpus.indexing.index import CorpusSearchMode, LanceCorpusIndex
from banso.corpus.models import (
    CorpusDocument,
    CorpusDocumentStatus,
    CorpusDocumentWrite,
)
from banso.corpus.sqlite_store import SQLiteCorpusStore


class _FakeEmbeddingProvider:
    model = "test-embedding"
    dimensions = 3

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        self.document_calls.append(tuple(texts))
        return tuple(_semantic_vector(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.query_calls.append(text)
        return _semantic_vector(text)


def _semantic_vector(text: str) -> tuple[float, ...]:
    text = text.casefold()
    if "automobile" in text or "car" in text or "vehicle" in text:
        return (1.0, 0.0, 0.0)
    if "apple" in text or "fruit" in text:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _active_document(
    store: SQLiteCorpusStore,
    *,
    url: str = "https://example.org/reports/1",
    text: str,
) -> CorpusDocument:
    return store.upsert(
        CorpusDocumentWrite(
            source_id="example-official",
            url=url,
            status=CorpusDocumentStatus.ACTIVE,
            title="Official report",
            text=text,
            published_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        )
    )


def test_chunking_is_paragraph_aware_bounded_and_stable(tmp_path: Path) -> None:
    with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
        document = _active_document(
            store,
            text=(
                "First paragraph.\n"
                "Second paragraph.\n"
                "supercalifragilisticexpialidocious"
            ),
        )
        chunks = chunk_document(document, max_chars=20)

    assert [chunk.text for chunk in chunks] == [
        "First paragraph.",
        "Second paragraph.",
        "supercalifragilistic",
        "expialidocious",
    ]
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert [chunk.id for chunk in chunks] == [
        f"{document.id}:0",
        f"{document.id}:1",
        f"{document.id}:2",
        f"{document.id}:3",
    ]
    assert chunk_document(document, max_chars=20) == chunks


def test_search_switches_between_bm25_vector_and_hybrid(tmp_path: Path) -> None:
    provider = _FakeEmbeddingProvider()
    index_path = tmp_path / "lancedb"
    with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
        semantic_only = _active_document(
            store,
            text="The vehicle regulator published a car safety report.",
        )
        lexical_and_semantic = _active_document(
            store,
            url="https://example.org/reports/2",
            text="The automobile safety agency published its findings.",
        )
        _active_document(
            store,
            url="https://example.org/reports/3",
            text="The fruit board published an apple market report.",
        )
        index = LanceCorpusIndex(index_path, embedding_provider=provider)
        assert index.rebuild(store) == 3

    bm25 = index.search("automobile", mode=CorpusSearchMode.BM25)
    assert [result.chunk.document_id for result in bm25] == [
        lexical_and_semantic.id
    ]
    assert provider.query_calls == []

    vector = index.search(
        "automobile",
        limit=2,
        mode=CorpusSearchMode.VECTOR,
    )
    assert {result.chunk.document_id for result in vector} == {
        semantic_only.id,
        lexical_and_semantic.id,
    }
    assert all(
        left.score >= right.score for left, right in zip(vector, vector[1:])
    )

    hybrid = index.search("automobile", mode=CorpusSearchMode.HYBRID)
    assert hybrid[0].chunk.document_id == lexical_and_semantic.id
    assert provider.query_calls == ["automobile", "automobile"]

    bm25_only = LanceCorpusIndex(index_path)
    assert bm25_only.search(
        "automobile",
        mode=CorpusSearchMode.BM25,
    )
    with pytest.raises(RuntimeError, match="embedding provider is required"):
        bm25_only.search("automobile", mode=CorpusSearchMode.VECTOR)


def test_rebuild_reuses_vectors_and_replaces_changed_documents(
    tmp_path: Path,
) -> None:
    provider = _FakeEmbeddingProvider()
    index_path = tmp_path / "lancedb"
    with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
        original = _active_document(
            store,
            text="The quasar observatory released its official findings.",
        )
        hidden = _active_document(
            store,
            url="https://example.org/reports/hidden",
            text="This document contains the archivalkeyword.",
        )
        index = LanceCorpusIndex(
            index_path,
            embedding_provider=provider,
            max_chunk_chars=80,
        )

        assert index.rebuild(store) == 2
        assert provider.document_calls == [
            (original.text, hidden.text),
        ]
        assert index.rebuild(store) == 2
        assert provider.document_calls[-1] == ()

        updated = _active_document(
            store,
            text="The nebula observatory released revised official findings.",
        )
        store.upsert(
            CorpusDocumentWrite(
                source_id=hidden.source_id,
                url=hidden.url,
                status=CorpusDocumentStatus.INACTIVE,
                failure_reason="withdrawn",
            )
        )

        assert updated.id == original.id
        assert index.rebuild(store) == 1
        assert provider.document_calls[-1] == (updated.text,)
        assert index.search("quasar", mode=CorpusSearchMode.BM25) == ()
        assert index.search("archivalkeyword", mode=CorpusSearchMode.BM25) == ()
        nebula_results = index.search("nebula", mode=CorpusSearchMode.BM25)
        assert len(nebula_results) == 1
        assert nebula_results[0].chunk.document_id == updated.id

        store.upsert(
            CorpusDocumentWrite(
                source_id=updated.source_id,
                url=updated.url,
                status=CorpusDocumentStatus.INACTIVE,
                failure_reason="withdrawn",
            )
        )
        assert index.rebuild(store) == 0
        assert provider.document_calls[-1] == ()
        assert index.search("nebula", mode=CorpusSearchMode.BM25) == ()

    mismatched = _FakeEmbeddingProvider()
    mismatched.model = "other-embedding"
    mismatched_index = LanceCorpusIndex(
        index_path,
        embedding_provider=mismatched,
    )
    with pytest.raises(ValueError, match="do not match"):
        mismatched_index.search("nebula", mode=CorpusSearchMode.VECTOR)

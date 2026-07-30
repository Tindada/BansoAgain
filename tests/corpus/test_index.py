"""Tests for corpus chunking and the rebuildable LanceDB BM25 index."""

from datetime import datetime, timezone
from pathlib import Path

from banso.corpus import (
    CorpusDocument,
    CorpusDocumentStatus,
    CorpusDocumentWrite,
    LanceCorpusIndex,
    SQLiteCorpusStore,
    chunk_document,
)


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


def test_rebuild_replaces_updated_and_inactive_documents(tmp_path: Path) -> None:
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
        index = LanceCorpusIndex(tmp_path / "lancedb", max_chunk_chars=80)

        assert index.rebuild(store) == 2
        quasar_results = index.search("quasar")
        assert len(quasar_results) == 1
        assert quasar_results[0].chunk.document_id == original.id
        assert quasar_results[0].chunk.published_at == original.published_at

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
        assert index.search("quasar") == ()
        assert index.search("archivalkeyword") == ()
        nebula_results = index.search("nebula")
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
        assert index.search("nebula") == ()

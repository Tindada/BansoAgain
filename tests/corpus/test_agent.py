"""Tests for adapting the local corpus to Agent component interfaces."""

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, call

import pytest

from banso.corpus.agent import (
    CorpusAwareDocumentFetcher,
    LocalCorpusRetrievalProvider,
)
from banso.corpus.indexing.chunking import chunk_document
from banso.corpus.indexing.index import (
    CorpusSearchMode,
    CorpusSearchResult,
    LanceCorpusIndex,
)
from banso.corpus.ingestion.registry import SourceRegistry, TrustedSource
from banso.corpus.models import CorpusDocument, CorpusDocumentStatus
from banso.corpus.sqlite_store import SQLiteCorpusStore
from banso.documents.fetcher import DocumentFetchRequest
from banso.documents.models import Document
from banso.retrieval.models import Source
from banso.retrieval.provider import SearchRequest

_NOW = datetime(2026, 7, 31, tzinfo=timezone.utc)


def _document(
    path: str,
    *,
    domain: str = "example.org",
    source_id: str = "example",
    status: CorpusDocumentStatus = CorpusDocumentStatus.ACTIVE,
    text: str = "Indexed passage.",
) -> CorpusDocument:
    url = f"https://{domain}/{path}"
    return CorpusDocument(
        id=path,
        source_id=source_id,
        url=url,
        canonical_url=url,
        status=status,
        title="Trusted report",
        text=text,
        media_type="text/html",
        published_at=_NOW,
        fetched_at=_NOW,
        content_hash=f"hash-{path}",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _result(
    document: CorpusDocument,
    score: float,
    *,
    position: int = 0,
    max_chars: int = 1200,
    content_hash: str | None = None,
) -> CorpusSearchResult:
    chunk = chunk_document(document, max_chars=max_chars)[position]
    if content_hash is not None:
        chunk = replace(chunk, content_hash=content_hash)
    return CorpusSearchResult(chunk=chunk, score=score)


@pytest.mark.anyio
async def test_local_provider_returns_only_current_approved_documents() -> None:
    current = _document(
        "reports/current",
        text="First relevant passage.\n\nSecond relevant passage.",
    )
    stale = _document("reports/stale")
    indexed_inactive = _document("reports/inactive")
    inactive = indexed_inactive.model_copy(
        update={"status": CorpusDocumentStatus.INACTIVE}
    )
    outside_scope = _document("private/report")
    missing_source = _document(
        "report",
        domain="removed.example.org",
        source_id="removed",
    )
    best = _result(current, 0.9, max_chars=24)

    index = Mock(spec=LanceCorpusIndex)
    index.search.return_value = (
        best,
        _result(current, 0.8, position=1, max_chars=24),
        _result(stale, 0.7, content_hash="outdated-content"),
        _result(indexed_inactive, 0.6),
        _result(outside_scope, 0.5),
        _result(missing_source, 0.4),
    )
    documents = {
        document.id: document
        for document in (
            current,
            stale,
            inactive,
            outside_scope,
            missing_source,
        )
    }
    store = Mock(spec=SQLiteCorpusStore)
    store.get.side_effect = documents.get
    registry = SourceRegistry(
        schema_version=2,
        sources=(
            TrustedSource(
                id="example",
                name="Example Institute",
                source_type="research",
                allowed_domains=("example.org",),
                allowed_path_prefixes=("/reports",),
            ),
        ),
    )
    provider = LocalCorpusRetrievalProvider(
        index,
        store,
        registry,
        mode=CorpusSearchMode.BM25,
    )

    results = await provider.search(SearchRequest(query="report", max_results=6))

    index.search.assert_called_once_with(
        "report",
        limit=6,
        mode=CorpusSearchMode.BM25,
    )
    assert len(results) == 1
    result = results[0]
    assert result.title == "Trusted report"
    assert result.url == current.url
    assert result.snippet == best.chunk.text
    assert result.source == Source(
        name="Example Institute",
        url="https://example.org",
    )
    assert result.published_at == current.published_at
    assert result.rank == 1
    assert result.metadata == {
        "provider": "local_corpus",
        "score": 0.9,
        "corpus_document_id": current.id,
        "corpus_chunk_id": best.chunk.id,
        "corpus_source_id": current.source_id,
        "corpus_content_hash": current.content_hash,
        "corpus_search_mode": "bm25",
    }


@pytest.mark.anyio
async def test_corpus_aware_fetcher_prefers_active_local_document() -> None:
    stored = _document(
        "reports/current",
        text="Stored authoritative text.",
    )
    store = Mock(spec=SQLiteCorpusStore)
    store.get_by_url.return_value = stored
    fallback = Mock()
    fallback.fetch = AsyncMock()
    source = Source(name="Example Institute")
    request = DocumentFetchRequest(
        url=stored.url,
        title="Search result title",
        source=source,
        metadata={"search_result_id": "result-1"},
    )

    document = await CorpusAwareDocumentFetcher(store, fallback).fetch(request)

    fallback.fetch.assert_not_awaited()
    assert document.url == stored.url
    assert document.title == stored.title
    assert document.text == stored.text
    assert document.source == source
    assert document.published_at == stored.published_at
    assert document.metadata == {
        "search_result_id": "result-1",
        "fetcher": "local_corpus",
        "corpus_document_id": stored.id,
        "corpus_source_id": stored.source_id,
        "corpus_content_hash": stored.content_hash,
        "corpus_media_type": stored.media_type,
        "corpus_fetched_at": stored.fetched_at.isoformat(),
    }


@pytest.mark.anyio
async def test_corpus_aware_fetcher_delegates_missing_and_inactive_documents() -> None:
    inactive = _document(
        "reports/inactive",
        status=CorpusDocumentStatus.INACTIVE,
    )
    requests = (
        DocumentFetchRequest(url=inactive.url, title="Inactive"),
        DocumentFetchRequest(
            url="https://example.org/reports/missing",
            title="Missing",
        ),
    )
    store = Mock(spec=SQLiteCorpusStore)
    store.get_by_url.side_effect = (inactive, None)
    fallback = Mock()
    fallback.fetch = AsyncMock(
        return_value=Document(
            url="https://example.org/network",
            title="Network",
            text="Network",
        )
    )
    fetcher = CorpusAwareDocumentFetcher(store, fallback)

    for request in requests:
        await fetcher.fetch(request)

    assert fallback.fetch.await_args_list == [call(request) for request in requests]

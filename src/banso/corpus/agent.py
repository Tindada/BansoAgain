"""Adapters between the local corpus and Agent component interfaces."""

from banso.corpus.indexing.index import (
    CorpusSearchMode,
    CorpusSearchResult,
    LanceCorpusIndex,
)
from banso.corpus.ingestion.registry import SourceRegistry, TrustedSource
from banso.corpus.models import CorpusDocument, CorpusDocumentStatus
from banso.corpus.sqlite_store import SQLiteCorpusStore
from banso.documents.fetcher import DocumentFetchRequest, DocumentFetcher
from banso.documents.models import Document
from banso.retrieval.models import SearchResult
from banso.retrieval.provider import SearchRequest
from banso.retrieval.url_utils import publisher_home_url
from banso.source import Source


class LocalCorpusRetrievalProvider:
    """Expose the trusted local corpus through the Agent retrieval interface."""

    def __init__(
        self,
        index: LanceCorpusIndex,
        store: SQLiteCorpusStore,
        registry: SourceRegistry,
        *,
        mode: CorpusSearchMode = CorpusSearchMode.VECTOR,
    ) -> None:
        self._index = index
        self._store = store
        self._registry = registry
        self._mode = mode

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Return current, in-scope corpus documents ranked by their best chunk."""

        candidates = self._index.search(
            request.query,
            limit=request.max_results,
            mode=self._mode,
        )
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for candidate in candidates:
            current = self._current_document(candidate)
            if current is None or current.canonical_url in seen_urls:
                continue
            source = self._current_source(current)
            if source is None:
                continue

            seen_urls.add(current.canonical_url)
            results.append(
                SearchResult(
                    title=current.title or current.url,
                    url=current.url,
                    snippet=candidate.chunk.text,
                    source=Source(
                        name=source.name,
                        url=publisher_home_url(current.url),
                    ),
                    published_at=current.published_at,
                    rank=len(results) + 1,
                    metadata={
                        "provider": "local_corpus",
                        "score": candidate.score,
                        "corpus_document_id": current.id,
                        "corpus_chunk_id": candidate.chunk.id,
                        "corpus_source_id": current.source_id,
                        "corpus_content_hash": current.content_hash,
                        "corpus_search_mode": self._mode.value,
                    },
                )
            )
        return results

    def _current_document(
        self,
        result: CorpusSearchResult,
    ) -> CorpusDocument | None:
        document = self._store.get(result.chunk.document_id)
        if (
            document is None
            or document.status != CorpusDocumentStatus.ACTIVE
            or document.source_id != result.chunk.source_id
            or document.content_hash != result.chunk.content_hash
        ):
            return None
        return document

    def _current_source(
        self,
        document: CorpusDocument,
    ) -> TrustedSource | None:
        source = self._registry.get(document.source_id)
        if source is None or not source.enabled or not source.contains_url(document.url):
            return None
        return source


class CorpusAwareDocumentFetcher:
    """Read active corpus documents locally before using another fetcher."""

    def __init__(
        self,
        store: SQLiteCorpusStore,
        fallback: DocumentFetcher,
    ) -> None:
        self._store = store
        self._fallback = fallback

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        """Return a local active document or delegate the original request."""

        document = self._store.get_by_url(request.url)
        if document is None or document.status != CorpusDocumentStatus.ACTIVE:
            return await self._fallback.fetch(request)

        return Document(
            url=document.url,
            title=document.title or request.title or document.url,
            text=document.text,
            source=request.source,
            published_at=document.published_at,
            metadata={
                **request.metadata,
                "fetcher": "local_corpus",
                "corpus_document_id": document.id,
                "corpus_source_id": document.source_id,
                "corpus_content_hash": document.content_hash,
                "corpus_media_type": document.media_type,
                "corpus_fetched_at": (
                    document.fetched_at.isoformat()
                    if document.fetched_at is not None
                    else None
                ),
            },
        )

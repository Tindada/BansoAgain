"""Background synchronization for one trusted corpus source."""

from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from banso.corpus.ingestion.discovery import (
    DiscoveryParseError,
    parse_feed_urls,
    parse_sitemap_urls,
)
from banso.corpus.ingestion.discovery_fetcher import (
    DiscoveryEndpointFetcher,
    DiscoveryFetchResult,
)
from banso.corpus.ingestion.page_fetcher import CorpusPageFetcher
from banso.corpus.ingestion.registry import TrustedSource
from banso.corpus.ingestion.robots import RobotsChecker, RobotsDecision
from banso.corpus.models import (
    CorpusDocument,
    CorpusDocumentStatus,
    CorpusDocumentWrite,
    DiscoveryEndpointState,
)
from banso.corpus.sqlite_store import SQLiteCorpusStore
from banso.documents.parser import DocumentParseError


@dataclass(frozen=True)
class CorpusSyncFailure:
    """A discovery or page URL that could not be synchronized."""

    url: str
    reason: str


@dataclass(frozen=True)
class CorpusSyncResult:
    """Documents reached and recoverable failures from one source sync."""

    documents: tuple[CorpusDocument, ...]
    failures: tuple[CorpusSyncFailure, ...]


class CorpusSyncService:
    """Synchronize one trusted source into the latest-version corpus."""

    def __init__(
        self,
        store: SQLiteCorpusStore,
        *,
        discovery_fetcher: DiscoveryEndpointFetcher | None = None,
        robots_checker: RobotsChecker | None = None,
        page_fetcher: CorpusPageFetcher | None = None,
    ) -> None:
        self._store = store
        self._discovery_fetcher = discovery_fetcher or DiscoveryEndpointFetcher()
        self._robots_checker = robots_checker or RobotsChecker()
        self._page_fetcher = page_fetcher or CorpusPageFetcher()

    async def sync_source(self, source: TrustedSource) -> CorpusSyncResult:
        """Discover and ingest changed pages for one enabled source."""

        if not source.enabled:
            raise ValueError(f"cannot sync disabled source: {source.id}")

        urls: dict[str, None] = {}
        failures: list[CorpusSyncFailure] = []

        for endpoint in source.feeds:
            try:
                result, content = await self._fetch_discovery(endpoint, source)
                for url in parse_feed_urls(
                    content,
                    document_url=result.final_url,
                ):
                    urls.setdefault(url, None)
                self._store.upsert_discovery_endpoint(
                    result.state,
                    content=content,
                )
            except (httpx.HTTPError, DiscoveryParseError, ValueError) as error:
                failures.append(_failure(endpoint, error))

        sitemap_queue = list(source.sitemaps)
        seen_sitemaps: set[str] = set()
        while sitemap_queue:
            endpoint = sitemap_queue.pop(0)
            if endpoint in seen_sitemaps:
                continue
            seen_sitemaps.add(endpoint)
            try:
                result, content = await self._fetch_discovery(endpoint, source)
                discovery = parse_sitemap_urls(
                    content,
                    document_url=result.final_url,
                )
                self._store.upsert_discovery_endpoint(
                    result.state,
                    content=content,
                )
                for url in discovery.content_urls:
                    urls.setdefault(url, None)
                sitemap_queue.extend(
                    url
                    for url in discovery.sitemap_urls
                    if url not in seen_sitemaps
                    and _source_domain_contains(source, url)
                )
            except (httpx.HTTPError, DiscoveryParseError, ValueError) as error:
                failures.append(_failure(endpoint, error))

        documents: list[CorpusDocument] = []
        for url in urls:
            if not source.contains_url(url):
                continue
            document, failure = await self._ingest_page(source, url)
            if document is not None:
                documents.append(document)
            if failure is not None:
                failures.append(failure)

        return CorpusSyncResult(
            documents=tuple(documents),
            failures=tuple(failures),
        )

    async def _fetch_discovery(
        self,
        endpoint: str,
        source: TrustedSource,
    ) -> tuple[DiscoveryFetchResult, bytes]:
        state = self._store.get_discovery_endpoint(endpoint)
        cached_content = self._store.get_discovery_content(endpoint)
        result = await self._discovery_fetcher.fetch(
            state or DiscoveryEndpointState(url=endpoint)
        )
        if not _source_domain_contains(source, result.final_url):
            raise ValueError("discovery redirect is outside the source domains")
        content = result.content if result.content is not None else cached_content
        if content is None:
            raise ValueError("received HTTP 304 without cached discovery content")
        return result, content

    async def _ingest_page(
        self,
        source: TrustedSource,
        url: str,
    ) -> tuple[CorpusDocument | None, CorpusSyncFailure | None]:
        existing = self._store.get_by_url(url)
        robots = await self._robots_checker.check(url)
        if robots == RobotsDecision.DEFERRED:
            return existing, CorpusSyncFailure(url=url, reason="robots_deferred")
        if robots == RobotsDecision.DISALLOWED:
            return (
                self._store.upsert(
                    CorpusDocumentWrite(
                        source_id=source.id,
                        url=url,
                        status=CorpusDocumentStatus.INACTIVE,
                        failure_reason="robots_disallowed",
                    )
                ),
                None,
            )

        try:
            result = await self._page_fetcher.fetch(
                url,
                etag=existing.etag if existing is not None else None,
                last_modified=(
                    existing.last_modified if existing is not None else None
                ),
            )
        except (httpx.HTTPError, DocumentParseError, ValueError) as error:
            failure = _failure(url, error)
            return (
                self._preserve_or_deactivate(source, url, existing, failure.reason),
                failure,
            )

        if result is None:
            if existing is not None:
                return existing, None
            failure = CorpusSyncFailure(
                url=url,
                reason="received HTTP 304 without a stored document",
            )
            return (
                self._preserve_or_deactivate(source, url, existing, failure.reason),
                failure,
            )
        if not source.contains_url(result.final_url):
            return (
                self._store.upsert(
                    CorpusDocumentWrite(
                        source_id=source.id,
                        url=url,
                        status=CorpusDocumentStatus.INACTIVE,
                        failure_reason="redirect_outside_source_scope",
                    )
                ),
                None,
            )
        if not result.document.text.strip():
            failure = CorpusSyncFailure(url=url, reason="no_extractable_text")
            return (
                self._preserve_or_deactivate(source, url, existing, failure.reason),
                failure,
            )

        return (
            self._store.upsert(
                CorpusDocumentWrite(
                    source_id=source.id,
                    url=url,
                    status=CorpusDocumentStatus.ACTIVE,
                    title=result.document.title
                    or (existing.title if existing is not None else None),
                    text=result.document.text,
                    media_type=result.media_type,
                    published_at=(
                        existing.published_at if existing is not None else None
                    ),
                    fetched_at=result.fetched_at,
                    etag=result.etag,
                    last_modified=result.last_modified,
                )
            ),
            None,
        )

    def _preserve_or_deactivate(
        self,
        source: TrustedSource,
        url: str,
        existing: CorpusDocument | None,
        reason: str,
    ) -> CorpusDocument:
        if existing is not None and existing.status == CorpusDocumentStatus.ACTIVE:
            return existing
        return self._store.upsert(
            CorpusDocumentWrite(
                source_id=source.id,
                url=url,
                status=CorpusDocumentStatus.INACTIVE,
                failure_reason=reason,
            )
        )


def _source_domain_contains(source: TrustedSource, url: str) -> bool:
    return urlsplit(url).hostname in source.allowed_domains


def _failure(url: str, error: Exception) -> CorpusSyncFailure:
    return CorpusSyncFailure(
        url=url,
        reason=str(error) or type(error).__name__,
    )

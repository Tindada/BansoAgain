"""Background synchronization for one trusted corpus source."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import httpx

from banso.corpus.ingestion.discovery import (
    DiscoveredURL,
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
        max_page_concurrency: int = 4,
    ) -> None:
        if max_page_concurrency <= 0:
            raise ValueError("max_page_concurrency must be greater than zero")
        self._store = store
        self._discovery_fetcher = discovery_fetcher or DiscoveryEndpointFetcher()
        self._robots_checker = robots_checker or RobotsChecker()
        self._page_fetcher = page_fetcher or CorpusPageFetcher()
        self._max_page_concurrency = max_page_concurrency

    async def sync_source(self, source: TrustedSource) -> CorpusSyncResult:
        """Discover and ingest changed pages for one enabled source."""

        if not source.enabled:
            raise ValueError(f"cannot sync disabled source: {source.id}")

        urls: dict[str, DiscoveredURL] = {}
        failures: list[CorpusSyncFailure] = []

        for endpoint in source.feeds:
            try:
                result, content = await self._fetch_discovery(endpoint, source)
                for discovered in parse_feed_urls(
                    content,
                    discovery_url=result.final_url,
                ):
                    _remember_discovered_url(urls, discovered)
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
                    discovery_url=result.final_url,
                )
                self._store.upsert_discovery_endpoint(
                    result.state,
                    content=content,
                )
                for discovered in discovery.content_urls:
                    _remember_discovered_url(urls, discovered)
                sitemap_queue.extend(
                    url
                    for url in discovery.sitemap_urls
                    if url not in seen_sitemaps
                    and _source_domain_contains(source, url)
                )
            except (httpx.HTTPError, DiscoveryParseError, ValueError) as error:
                failures.append(_failure(endpoint, error))

        origin_limits: dict[tuple[str, str, int | None], asyncio.Semaphore] = {}
        page_requests: list[tuple[DiscoveredURL, asyncio.Semaphore]] = []
        for discovered in urls.values():
            if not source.contains_url(discovered.url):
                continue
            origin = _origin(discovered.url)
            limit = origin_limits.get(origin)
            if limit is None:
                limit = asyncio.Semaphore(self._max_page_concurrency)
                origin_limits[origin] = limit
            page_requests.append((discovered, limit))

        page_results = await asyncio.gather(
            *(
                self._ingest_page_with_limit(source, discovered, limit)
                for discovered, limit in page_requests
            )
        )

        documents: list[CorpusDocument] = []
        for document, failure in page_results:
            if document is not None:
                documents.append(document)
            if failure is not None:
                failures.append(failure)

        return CorpusSyncResult(
            documents=tuple(documents),
            failures=tuple(failures),
        )

    async def _ingest_page_with_limit(
        self,
        source: TrustedSource,
        discovered: DiscoveredURL,
        limit: asyncio.Semaphore,
    ) -> tuple[CorpusDocument | None, CorpusSyncFailure | None]:
        async with limit:
            return await self._ingest_page(source, discovered)

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
        discovered: DiscoveredURL,
    ) -> tuple[CorpusDocument | None, CorpusSyncFailure | None]:
        url = discovered.url
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
                    published_at=_select_published_at(
                        page_published_at=result.document.published_at,
                        discovery_published_at=discovered.published_at,
                        existing_published_at=(
                            existing.published_at if existing is not None else None
                        ),
                        fetched_at=result.fetched_at,
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


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme, parsed.hostname or "", parsed.port


def _failure(url: str, error: Exception) -> CorpusSyncFailure:
    return CorpusSyncFailure(
        url=url,
        reason=str(error) or type(error).__name__,
    )


def _remember_discovered_url(
    urls: dict[str, DiscoveredURL],
    discovered: DiscoveredURL,
) -> None:
    existing = urls.get(discovered.url)
    if existing is None or (
        existing.published_at is None and discovered.published_at is not None
    ):
        urls[discovered.url] = discovered


def _select_published_at(
    *,
    page_published_at: datetime | None,
    discovery_published_at: datetime | None,
    existing_published_at: datetime | None,
    fetched_at: datetime,
) -> datetime | None:
    latest_allowed = fetched_at + timedelta(hours=24)
    for candidate in (
        page_published_at,
        discovery_published_at,
        existing_published_at,
    ):
        if candidate is None or candidate.tzinfo is None:
            continue
        candidate = candidate.astimezone(timezone.utc)
        if candidate <= latest_allowed:
            return candidate
    return None

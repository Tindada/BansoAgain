"""End-to-end tests for trusted-source corpus synchronization."""

import asyncio
from pathlib import Path

import httpx
import pytest

from banso.corpus.ingestion.discovery_fetcher import DiscoveryEndpointFetcher
from banso.corpus.ingestion.page_fetcher import CorpusPageFetcher
from banso.corpus.ingestion.registry import TrustedSource
from banso.corpus.ingestion.robots import RobotsChecker
from banso.corpus.ingestion.sync import CorpusSyncService
from banso.corpus.models import (
    CorpusDocumentStatus,
    CorpusDocumentWrite,
)
from banso.corpus.sqlite_store import SQLiteCorpusStore


def _source(**overrides: object) -> TrustedSource:
    values: dict[str, object] = {
        "id": "example-official",
        "name": "Example Official",
        "source_type": "official",
        "allowed_domains": ("example.org",),
        "allowed_path_prefixes": ("/reports",),
        "feeds": ("https://example.org/feed.xml",),
        "sitemaps": ("https://example.org/sitemap.xml",),
    }
    values.update(overrides)
    return TrustedSource.model_validate(values)


def _service(
    store: SQLiteCorpusStore,
    client: httpx.AsyncClient,
    *,
    max_page_concurrency: int = 4,
) -> CorpusSyncService:
    return CorpusSyncService(
        store,
        discovery_fetcher=DiscoveryEndpointFetcher(client=client),
        robots_checker=RobotsChecker(client=client),
        page_fetcher=CorpusPageFetcher(client=client),
        max_page_concurrency=max_page_concurrency,
    )


@pytest.mark.anyio
async def test_syncs_feed_and_nested_sitemap_into_corpus(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers.get("if-none-match"):
            return httpx.Response(304, request=request)

        responses = {
            "/feed.xml": (
                '<rss><channel><item><link>https://example.org/reports/1'
                "</link></item><item><link>https://outside.example/report"
                "</link></item></channel></rss>"
            ),
            "/sitemap.xml": (
                "<sitemapindex><sitemap><loc>https://example.org/maps/reports.xml"
                "</loc></sitemap></sitemapindex>"
            ),
            "/maps/reports.xml": (
                "<urlset><url><loc>https://example.org/reports/2</loc></url>"
                "<url><loc>https://example.org/reports/blocked</loc></url></urlset>"
            ),
        }
        if request.url.path in responses:
            return httpx.Response(
                200,
                text=responses[request.url.path],
                headers={"ETag": f'"{request.url.path}"'},
                request=request,
            )
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /reports/blocked\n",
                request=request,
            )
        if request.url.path in {"/reports/1", "/reports/2"}:
            number = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                text=f"<main><h1>Report {number}</h1><p>Body {number}.</p></main>",
                headers={
                    "Content-Type": "text/html",
                    "ETag": f'"page-{number}"',
                },
                request=request,
            )
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
    ) as client:
        with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
            service = _service(store, client)
            first = await service.sync_source(_source())
            second = await service.sync_source(_source())

            documents = store.list_documents()
            assert [document.url for document in documents] == [
                "https://example.org/reports/1",
                "https://example.org/reports/2",
                "https://example.org/reports/blocked",
            ]
            assert [document.status for document in documents] == [
                CorpusDocumentStatus.ACTIVE,
                CorpusDocumentStatus.ACTIVE,
                CorpusDocumentStatus.INACTIVE,
            ]
            assert documents[2].failure_reason == "robots_disallowed"
            assert store.get_by_url("https://outside.example/report") is None
            assert store.get_discovery_endpoint(
                "https://example.org/maps/reports.xml"
            ) is not None

    assert len(first.documents) == 3
    assert first.failures == ()
    assert len(second.documents) == 3
    assert second.failures == ()
    assert sum(request.url.path == "/robots.txt" for request in requests) == 1
    assert sum(request.url.path.startswith("/reports/") for request in requests) == 4


@pytest.mark.anyio
async def test_bounds_page_fetch_concurrency_per_origin(tmp_path: Path) -> None:
    active_pages = 0
    peak_active_pages = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal active_pages, peak_active_pages
        if request.url.path == "/feed.xml":
            links = "".join(
                f"<item><link>https://example.org/reports/{number}</link></item>"
                for number in range(4)
            )
            return httpx.Response(
                200,
                text=f"<rss><channel>{links}</channel></rss>",
                request=request,
            )
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)

        active_pages += 1
        peak_active_pages = max(peak_active_pages, active_pages)
        await asyncio.sleep(0.01)
        active_pages -= 1
        return httpx.Response(
            200,
            text="<main><p>Official report.</p></main>",
            headers={"Content-Type": "text/html"},
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
    ) as client:
        with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
            result = await _service(
                store,
                client,
                max_page_concurrency=2,
            ).sync_source(_source(sitemaps=()))

    assert len(result.documents) == 4
    assert result.failures == ()
    assert peak_active_pages == 2


@pytest.mark.anyio
async def test_does_not_save_validators_for_invalid_discovery_content(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text="<rss>",
                headers={"ETag": '"invalid"'},
                request=request,
            )
        )
    ) as client:
        with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
            result = await _service(store, client).sync_source(
                _source(sitemaps=())
            )

            assert len(result.failures) == 1
            assert store.get_discovery_endpoint(
                "https://example.org/feed.xml"
            ) is None


@pytest.mark.anyio
async def test_page_failure_preserves_existing_active_document(
    tmp_path: Path,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/feed.xml":
            return httpx.Response(
                200,
                text="<rss><channel><item><link>"
                "https://example.org/reports/1"
                "</link></item></channel></rss>",
                request=request,
            )
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        return httpx.Response(503, request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
    ) as client:
        with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
            existing = store.upsert(
                CorpusDocumentWrite(
                    source_id="example-official",
                    url="https://example.org/reports/1",
                    status=CorpusDocumentStatus.ACTIVE,
                    text="Last known good body.",
                    etag='"old"',
                )
            )
            result = await _service(store, client).sync_source(
                _source(sitemaps=())
            )

            assert result.documents == (existing,)
            assert len(result.failures) == 1
            assert store.get(existing.id) == existing

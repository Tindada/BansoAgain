"""Tests for conditional discovery endpoint fetching."""

import asyncio

import httpx
import pytest

from banso.corpus import DiscoveryEndpointFetcher, DiscoveryEndpointState


async def _fetch_modified_then_not_modified() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                content=b"<rss />",
                headers={
                    "etag": '"feed-v1"',
                    "last-modified": "Wed, 29 Jul 2026 08:00:00 GMT",
                },
                request=request,
            )
        return httpx.Response(304, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = DiscoveryEndpointFetcher(client=client)
        first = await fetcher.fetch(
            DiscoveryEndpointState(url="https://example.org/feed.xml")
        )
        second = await fetcher.fetch(first.state)

    assert first.content == b"<rss />"
    assert first.final_url == "https://example.org/feed.xml"
    assert first.state.etag == '"feed-v1"'
    assert first.state.last_modified == "Wed, 29 Jul 2026 08:00:00 GMT"
    assert second.content is None
    assert second.state == first.state
    assert "if-none-match" not in requests[0].headers
    assert requests[1].headers["if-none-match"] == '"feed-v1"'
    assert (
        requests[1].headers["if-modified-since"]
        == "Wed, 29 Jul 2026 08:00:00 GMT"
    )


def test_fetcher_uses_saved_validators_and_handles_not_modified() -> None:
    asyncio.run(_fetch_modified_then_not_modified())


def test_fetcher_clears_stale_validators_after_unconditional_response() -> None:
    async def run() -> DiscoveryEndpointState:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<rss />", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            result = await DiscoveryEndpointFetcher(client=client).fetch(
                DiscoveryEndpointState(
                    url="https://example.org/feed.xml",
                    etag='"stale"',
                    last_modified="Wed, 29 Jul 2026 08:00:00 GMT",
                )
            )
            return result.state

    assert asyncio.run(run()) == DiscoveryEndpointState(
        url="https://example.org/feed.xml"
    )


def test_fetcher_exposes_http_errors() -> None:
    async def run() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await DiscoveryEndpointFetcher(client=client).fetch(
                DiscoveryEndpointState(url="https://example.org/feed.xml")
            )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())

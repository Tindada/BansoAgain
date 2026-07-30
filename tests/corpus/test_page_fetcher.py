"""Tests for conditional corpus page fetching and parsing."""

import httpx
import pytest

from banso.corpus import CorpusPageFetcher


@pytest.mark.anyio
async def test_fetches_parsed_page_then_uses_http_validators() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "text/html; charset=utf-8",
                    "ETag": '"v1"',
                    "Last-Modified": "Wed, 29 Jul 2026 08:00:00 GMT",
                },
                text="<html><head><title>Report</title></head>"
                "<body><main><p>Official update.</p></main></body></html>",
                request=request,
            )
        return httpx.Response(304, request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
    ) as client:
        fetcher = CorpusPageFetcher(client=client)
        result = await fetcher.fetch("https://example.org/report")
        unchanged = await fetcher.fetch(
            "https://example.org/report",
            etag='"v1"',
            last_modified="Wed, 29 Jul 2026 08:00:00 GMT",
        )

    assert result is not None
    assert result.final_url == "https://example.org/report"
    assert result.document.title == "Report"
    assert result.document.text == "Official update."
    assert result.media_type == "text/html"
    assert result.etag == '"v1"'
    assert result.last_modified == "Wed, 29 Jul 2026 08:00:00 GMT"
    assert unchanged is None
    assert requests[1].headers["If-None-Match"] == '"v1"'
    assert (
        requests[1].headers["If-Modified-Since"]
        == "Wed, 29 Jul 2026 08:00:00 GMT"
    )


@pytest.mark.anyio
async def test_rejects_unsupported_page_media_type() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=b"image",
                request=request,
            )
        )
    ) as client:
        fetcher = CorpusPageFetcher(client=client)
        with pytest.raises(ValueError, match="unsupported document media type"):
            await fetcher.fetch("https://example.org/image.png")

"""Tests for robots.txt checks and per-origin caching."""

import asyncio

import httpx
import pytest

from banso.corpus.ingestion.robots import RobotsChecker, RobotsDecision


async def _check_rules_and_cache() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text="""
            User-agent: *
            Disallow: /private/
            Allow: /private/public/

            User-agent: banso-corpus
            Disallow: /drafts/
            Allow: /drafts/published/
            """,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        checker = RobotsChecker(client=client)
        assert (
            await checker.check("https://example.org/drafts/internal")
            == RobotsDecision.DISALLOWED
        )
        assert (
            await checker.check("https://example.org/drafts/published/report")
            == RobotsDecision.ALLOWED
        )
        assert (
            await checker.check("https://example.org/private/report")
            == RobotsDecision.ALLOWED
        )

    assert len(requests) == 1
    assert str(requests[0].url) == "https://example.org/robots.txt"
    assert requests[0].headers["user-agent"] == "banso-corpus/0.1"


def test_checker_applies_specific_group_and_caches_by_origin() -> None:
    asyncio.run(_check_rules_and_cache())


async def _check_longest_rule_and_allow_tie() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="""
            User-agent: *
            Disallow: /reports/*
            Allow: /reports/public$
            Disallow: /same
            Allow: /same
            """,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        checker = RobotsChecker(client=client)
        assert (
            await checker.check("https://example.org/reports/private")
            == RobotsDecision.DISALLOWED
        )
        assert (
            await checker.check("https://example.org/reports/public")
            == RobotsDecision.ALLOWED
        )
        assert (
            await checker.check("https://example.org/reports/public/more")
            == RobotsDecision.DISALLOWED
        )
        assert (
            await checker.check("https://example.org/same")
            == RobotsDecision.ALLOWED
        )


def test_checker_uses_longest_rule_with_allow_winning_a_tie() -> None:
    asyncio.run(_check_longest_rule_and_allow_tie())


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (404, RobotsDecision.ALLOWED),
        (410, RobotsDecision.ALLOWED),
        (401, RobotsDecision.DISALLOWED),
        (403, RobotsDecision.DISALLOWED),
        (429, RobotsDecision.DEFERRED),
        (500, RobotsDecision.DEFERRED),
    ],
)
def test_checker_maps_http_failures(
    status_code: int,
    expected: RobotsDecision,
) -> None:
    async def run() -> RobotsDecision:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await RobotsChecker(client=client).check(
                "https://example.org/report"
            )

    assert asyncio.run(run()) == expected


def test_checker_defers_transport_failures() -> None:
    async def run() -> RobotsDecision:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection failed", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await RobotsChecker(client=client).check(
                "https://example.org/report"
            )

    assert asyncio.run(run()) == RobotsDecision.DEFERRED


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "ftp://example.org/report",
        "https://user:secret@example.org/report",
    ],
)
def test_checker_rejects_invalid_content_urls(url: str) -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        asyncio.run(RobotsChecker().check(url))

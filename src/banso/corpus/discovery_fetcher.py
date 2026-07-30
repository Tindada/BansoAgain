"""Conditional HTTP fetching for RSS, Atom, and Sitemap endpoints."""

from dataclasses import dataclass

import httpx

from banso.corpus.models import DiscoveryEndpointState


@dataclass(frozen=True)
class DiscoveryFetchResult:
    """A fetched discovery document, or a not-modified response."""

    state: DiscoveryEndpointState
    content: bytes | None


class DiscoveryEndpointFetcher:
    """Fetch discovery endpoints with saved HTTP validators."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
        user_agent: str = "banso-corpus/0.1",
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._headers = {
            "User-Agent": user_agent,
            "Accept": (
                "application/atom+xml,application/rss+xml,application/xml,"
                "text/xml;q=0.9,*/*;q=0.1"
            ),
        }

    async def fetch(
        self,
        state: DiscoveryEndpointState,
    ) -> DiscoveryFetchResult:
        """Fetch an endpoint, returning no content for HTTP 304."""

        headers = dict(self._headers)
        if state.etag is not None:
            headers["If-None-Match"] = state.etag
        if state.last_modified is not None:
            headers["If-Modified-Since"] = state.last_modified

        if self._client is not None:
            response = await self._client.get(state.url, headers=headers)
        else:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(state.url, headers=headers)

        not_modified = response.status_code == 304
        if not not_modified:
            response.raise_for_status()
        return DiscoveryFetchResult(
            state=_response_state(response, previous=state),
            content=None if not_modified else response.content,
        )


def _response_state(
    response: httpx.Response,
    *,
    previous: DiscoveryEndpointState,
) -> DiscoveryEndpointState:
    preserve = response.status_code == 304
    return DiscoveryEndpointState(
        url=previous.url,
        etag=response.headers.get("etag", previous.etag if preserve else None),
        last_modified=response.headers.get(
            "last-modified",
            previous.last_modified if preserve else None,
        ),
    )

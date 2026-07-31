"""robots.txt access checks for background corpus ingestion."""

from enum import StrEnum

import httpx
from protego import Protego


class RobotsDecision(StrEnum):
    """The result of checking whether a URL may be fetched."""

    ALLOWED = "allowed"
    DISALLOWED = "disallowed"
    DEFERRED = "deferred"


class RobotsChecker:
    """Fetch and cache robots.txt decisions by origin."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "banso-corpus/0.1",
        timeout: float = 20.0,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout
        self._cache: dict[
            tuple[str, str, int | None],
            Protego | RobotsDecision,
        ] = {}

    async def check(self, url: str) -> RobotsDecision:
        """Return whether robots policy allows fetching a content URL."""

        target = _http_url(url)
        origin = (target.scheme, target.host, target.port)
        policy = self._cache.get(origin)
        if policy is None:
            policy = await self._load_policy(
                target.copy_with(path="/robots.txt", query=None, fragment=None)
            )
            self._cache[origin] = policy

        if isinstance(policy, RobotsDecision):
            return policy
        return (
            RobotsDecision.ALLOWED
            if policy.can_fetch(str(target), self._user_agent)
            else RobotsDecision.DISALLOWED
        )

    async def _load_policy(
        self,
        robots_url: httpx.URL,
    ) -> Protego | RobotsDecision:
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/plain,*/*;q=0.1",
        }
        try:
            if self._client is not None:
                response = await self._client.get(robots_url, headers=headers)
            else:
                async with httpx.AsyncClient(
                    timeout=self._timeout,
                    follow_redirects=True,
                ) as client:
                    response = await client.get(robots_url, headers=headers)
        except httpx.TransportError:
            return RobotsDecision.DEFERRED

        if 200 <= response.status_code < 300:
            return Protego.parse(response.text)
        if response.status_code in {404, 410}:
            return RobotsDecision.ALLOWED
        if response.status_code in {401, 403}:
            return RobotsDecision.DISALLOWED
        return RobotsDecision.DEFERRED


def _http_url(url: str) -> httpx.URL:
    try:
        parsed = httpx.URL(url.strip())
    except (httpx.InvalidURL, ValueError) as error:
        raise ValueError("expected an absolute HTTP(S) URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.host
        or parsed.userinfo
    ):
        raise ValueError("expected an absolute HTTP(S) URL")
    return parsed

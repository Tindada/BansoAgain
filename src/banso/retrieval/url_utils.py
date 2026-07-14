"""Utilities for identifying content publishers from result URLs."""

from urllib.parse import urlsplit


def publisher_domain(url: str) -> str:
    """Return a normalized publisher hostname for a result URL."""

    hostname = urlsplit(url).hostname or ""
    hostname = hostname.lower().rstrip(".")
    return hostname.removeprefix("www.")


def publisher_home_url(url: str) -> str | None:
    """Return the origin URL for a publisher, when the URL is valid."""

    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"

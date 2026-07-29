"""Shared URL identity and publisher utilities."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def normalize_url(url: str, *, ignored_query_params: set[str] | None = None) -> str:
    """Normalize a URL for identity and duplicate detection."""

    ignored_query_params = ignored_query_params or set()
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    query_pairs: list[tuple[str, str]] = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in ignored_query_params
    ]
    query = urlencode(sorted(query_pairs))

    return urlunsplit((scheme, netloc, path, query, ""))


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

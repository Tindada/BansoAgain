"""Pure RSS, Atom, and Sitemap URL discovery."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

from banso.datetime_utils import parse_external_datetime
from banso.retrieval.url_utils import normalize_url


class DiscoveryParseError(ValueError):
    """Raised when a discovery document cannot be parsed."""


@dataclass(frozen=True)
class DiscoveredURL:
    """A content URL and optional metadata supplied by discovery."""

    url: str
    published_at: datetime | None = None


@dataclass(frozen=True)
class SitemapDiscovery:
    """URLs declared by either a Sitemap urlset or sitemap index."""

    content_urls: tuple[DiscoveredURL, ...] = ()
    sitemap_urls: tuple[str, ...] = ()


def parse_feed_urls(content: bytes | str, *, discovery_url: str) -> tuple[DiscoveredURL, ...]:
    """Return normalized content URLs and publication dates from a feed."""

    root = _parse_xml(content)
    root_name = _local_name(root.tag)

    if root_name == "feed":
        entries = (
            _atom_entry(entry)
            for entry in root
            if _local_name(entry.tag) == "entry"
        )
    elif root_name in {"rss", "RDF"}:
        entries = (
            _rss_item(item)
            for item in root.iter()
            if _local_name(item.tag) == "item"
        )
    else:
        raise DiscoveryParseError(f"unsupported feed root element: {root_name}")

    return _normalize_discovered_urls(entries, discovery_url=discovery_url)


def parse_sitemap_urls(
    content: bytes | str,
    *,
    discovery_url: str,
) -> SitemapDiscovery:
    """Return normalized page or nested Sitemap URLs from a Sitemap document."""

    root = _parse_xml(content)
    root_name = _local_name(root.tag)

    if root_name == "urlset":
        urls = (
            _child_text(entry, "loc")
            for entry in root
            if _local_name(entry.tag) == "url"
        )
        return SitemapDiscovery(
            content_urls=_normalize_discovered_urls(
                (DiscoveredURL(url=url) for url in urls if url is not None),
                discovery_url=discovery_url,
            )
        )
    if root_name == "sitemapindex":
        urls = (
            _child_text(entry, "loc")
            for entry in root
            if _local_name(entry.tag) == "sitemap"
        )
        return SitemapDiscovery(
            sitemap_urls=_normalize_urls(urls, discovery_url=discovery_url)
        )
    raise DiscoveryParseError(f"unsupported Sitemap root element: {root_name}")


def _parse_xml(content: bytes | str) -> ElementTree.Element:
    raw = content.encode() if isinstance(content, str) else content
    upper_content = raw.upper()
    if b"<!DOCTYPE" in upper_content or b"<!ENTITY" in upper_content:
        raise DiscoveryParseError("XML declarations for DTDs or entities are forbidden")
    try:
        return ElementTree.fromstring(content)
    except (ElementTree.ParseError, ValueError) as error:
        raise DiscoveryParseError("invalid XML discovery document") from error


def _atom_entry(entry: ElementTree.Element) -> DiscoveredURL | None:
    for child in entry:
        if _local_name(child.tag) != "link":
            continue
        href = child.get("href")
        if child.get("rel", "alternate").strip().lower() == "alternate" and href:
            return DiscoveredURL(
                url=href,
                published_at=parse_external_datetime(_child_text(entry, "published")),
            )
    return None


def _rss_item(item: ElementTree.Element) -> DiscoveredURL | None:
    url = _child_text(item, "link")
    if url is None:
        return None
    return DiscoveredURL(
        url=url,
        published_at=parse_external_datetime(
            _child_text(item, "pubDate") or _child_text(item, "date")
        ),
    )


def _child_text(element: ElementTree.Element, name: str) -> str | None:
    return next(
        (
            child.text
            for child in element
            if _local_name(child.tag) == name and child.text
        ),
        None,
    )


def _normalize_urls(
    urls: Iterable[str | None],
    *,
    discovery_url: str,
) -> tuple[str, ...]:
    _validate_http_url(discovery_url)
    discovered: dict[str, None] = {}
    for value in urls:
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = urljoin(discovery_url, value.strip())
        try:
            _validate_http_url(candidate)
        except ValueError:
            continue
        discovered.setdefault(normalize_url(candidate), None)
    return tuple(discovered)


def _normalize_discovered_urls(
    entries: Iterable[DiscoveredURL | None],
    *,
    discovery_url: str,
) -> tuple[DiscoveredURL, ...]:
    _validate_http_url(discovery_url)
    discovered: dict[str, datetime | None] = {}
    for entry in entries:
        if entry is None or not entry.url.strip():
            continue
        candidate = urljoin(discovery_url, entry.url.strip())
        try:
            _validate_http_url(candidate)
        except ValueError:
            continue
        normalized = normalize_url(candidate)
        if normalized not in discovered or discovered[normalized] is None:
            discovered[normalized] = entry.published_at
    return tuple(
        DiscoveredURL(url=url, published_at=published_at)
        for url, published_at in discovered.items()
    )


def _validate_http_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
    except ValueError as error:
        raise ValueError("expected an absolute HTTP(S) URL") from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.hostname.endswith(".")
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("expected an absolute HTTP(S) URL")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

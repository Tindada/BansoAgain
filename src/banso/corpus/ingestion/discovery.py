"""Pure RSS, Atom, and Sitemap URL discovery."""

from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

from banso.retrieval.url_utils import normalize_url


class DiscoveryParseError(ValueError):
    """Raised when a discovery document cannot be parsed."""


@dataclass(frozen=True)
class SitemapDiscovery:
    """URLs declared by either a Sitemap urlset or sitemap index."""

    content_urls: tuple[str, ...] = ()
    sitemap_urls: tuple[str, ...] = ()


def parse_feed_urls(content: bytes | str, *, document_url: str) -> tuple[str, ...]:
    """Return normalized content URLs from an RSS or Atom document."""

    root = _parse_xml(content)
    root_name = _local_name(root.tag)

    if root_name == "feed":
        links = (
            _atom_entry_url(entry)
            for entry in root
            if _local_name(entry.tag) == "entry"
        )
    elif root_name in {"rss", "RDF"}:
        links = (
            _child_text(item, "link")
            for item in root.iter()
            if _local_name(item.tag) == "item"
        )
    else:
        raise DiscoveryParseError(f"unsupported feed root element: {root_name}")

    return _normalize_urls(links, document_url=document_url)


def parse_sitemap_urls(
    content: bytes | str,
    *,
    document_url: str,
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
            content_urls=_normalize_urls(urls, document_url=document_url)
        )
    if root_name == "sitemapindex":
        urls = (
            _child_text(entry, "loc")
            for entry in root
            if _local_name(entry.tag) == "sitemap"
        )
        return SitemapDiscovery(
            sitemap_urls=_normalize_urls(urls, document_url=document_url)
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


def _atom_entry_url(entry: ElementTree.Element) -> str | None:
    for child in entry:
        if _local_name(child.tag) != "link":
            continue
        href = child.get("href")
        if child.get("rel", "alternate").strip().lower() == "alternate" and href:
            return href
    return None


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
    document_url: str,
) -> tuple[str, ...]:
    _validate_http_url(document_url)
    discovered: dict[str, None] = {}
    for value in urls:
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = urljoin(document_url, value.strip())
        try:
            _validate_http_url(candidate)
        except ValueError:
            continue
        discovered.setdefault(normalize_url(candidate), None)
    return tuple(discovered)


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

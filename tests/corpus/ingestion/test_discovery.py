"""Tests for pure RSS, Atom, and Sitemap discovery."""

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from banso.corpus.ingestion.discovery import (
    DiscoveredURL,
    DiscoveryParseError,
    SitemapDiscovery,
    parse_feed_urls,
    parse_sitemap_urls,
)


def test_parse_rss_urls_normalizes_and_deduplicates_links() -> None:
    content = """
    <rss version="2.0">
      <channel>
        <item>
          <link>https://example.org/reports/latest/?b=2&amp;a=1#summary</link>
          <pubDate>Mon, 10 Aug 2026 08:00:00 GMT</pubDate>
        </item>
        <item><link>https://example.org/reports/latest?a=1&amp;b=2</link></item>
        <item><link>mailto:press@example.org</link></item>
      </channel>
    </rss>
    """

    assert parse_feed_urls(
        content,
        discovery_url="https://example.org/feed.xml",
    ) == (
        DiscoveredURL(
            url="https://example.org/reports/latest?a=1&b=2",
            published_at=datetime(2026, 8, 10, 8, tzinfo=timezone.utc),
        ),
    )


def test_parse_atom_urls_uses_alternate_links_and_resolves_relative_urls() -> None:
    content = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <link rel="self" href="/api/entries/1" />
        <link rel="alternate" href="/reports/1" />
        <published>2026-08-09T12:30:00+02:00</published>
      </entry>
      <entry>
        <link href="https://example.org/reports/2/" />
        <updated>2026-08-10T09:30:00Z</updated>
      </entry>
    </feed>
    """

    assert parse_feed_urls(
        content,
        discovery_url="https://example.org/feeds/news.xml",
    ) == (
        DiscoveredURL(
            url="https://example.org/reports/1",
            published_at=datetime(2026, 8, 9, 10, 30, tzinfo=timezone.utc),
        ),
        DiscoveredURL(url="https://example.org/reports/2"),
    )


def test_parse_rss_one_urls() -> None:
    content = """
    <rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns="http://purl.org/rss/1.0/">
      <item>
        <link>https://example.org/reports/1</link>
        <dc:date xmlns:dc="http://purl.org/dc/elements/1.1/">2026-08-08</dc:date>
      </item>
    </rdf:RDF>
    """

    assert parse_feed_urls(
        content,
        discovery_url="https://example.org/feed.rdf",
    ) == (
        DiscoveredURL(
            url="https://example.org/reports/1",
            published_at=datetime(2026, 8, 8, tzinfo=timezone.utc),
        ),
    )


def test_parse_sitemap_urlset_returns_only_content_urls() -> None:
    content = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://example.org/reports/1</loc>
        <lastmod>2026-08-10T08:00:00Z</lastmod>
      </url>
      <url><loc>/reports/2/</loc></url>
    </urlset>
    """

    assert parse_sitemap_urls(
        content,
        discovery_url="https://example.org/sitemap.xml",
    ) == SitemapDiscovery(
        content_urls=(
            DiscoveredURL(url="https://example.org/reports/1"),
            DiscoveredURL(url="https://example.org/reports/2"),
        )
    )


def test_parse_sitemap_index_returns_only_nested_sitemaps() -> None:
    content = """
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.org/sitemaps/reports.xml</loc></sitemap>
      <sitemap><loc>/sitemaps/archive.xml</loc></sitemap>
    </sitemapindex>
    """

    assert parse_sitemap_urls(
        content,
        discovery_url="https://example.org/sitemap.xml",
    ) == SitemapDiscovery(
        sitemap_urls=(
            "https://example.org/sitemaps/reports.xml",
            "https://example.org/sitemaps/archive.xml",
        )
    )


@pytest.mark.parametrize(
    ("parser", "content", "message"),
    [
        (parse_feed_urls, "<html />", "unsupported feed"),
        (parse_sitemap_urls, "<html />", "unsupported Sitemap"),
        (parse_feed_urls, "<rss>", "invalid XML"),
        (
            parse_sitemap_urls,
            '<!DOCTYPE x [<!ENTITY y "value">]><urlset />',
            "forbidden",
        ),
    ],
)
def test_discovery_rejects_invalid_or_unsafe_documents(
    parser: Callable[..., object],
    content: str,
    message: str,
) -> None:
    with pytest.raises(DiscoveryParseError, match=message):
        parser(content, discovery_url="https://example.org/discovery.xml")

"""Tests for retrieval result filtering."""

from banso.retrieval.filter import (
    RetrievalFilter,
    RetrievalFilterConfig,
)
from banso.retrieval.models import SearchResult
from banso.retrieval.url_utils import normalize_url


def test_retrieval_filter_drops_empty_and_duplicate_results() -> None:
    results = [
        SearchResult(title="First", url="https://Example.com/news?a=1&utm_source=x"),
        SearchResult(title="Duplicate", url="https://example.com/news?utm_medium=y&a=1"),
        SearchResult(title="", url="https://example.com/empty-title"),
        SearchResult(title="Empty URL", url=" "),
        SearchResult(title="Second", url="https://example.com/second"),
    ]

    output = RetrievalFilter().apply(results)

    assert [result.title for result in output.results] == ["First", "Second"]
    assert output.report.input_count == 5
    assert output.report.output_count == 2
    assert output.report.dropped_duplicate_url == 1
    assert output.report.dropped_empty_title == 1
    assert output.report.dropped_empty_url == 1
    assert output.report.dropped_invalid_url == 0
    assert output.report.truncated_count == 0


def test_retrieval_filter_preserves_order_and_limits_results() -> None:
    results = [
        SearchResult(title="First", url="https://example.com/first"),
        SearchResult(title="Second", url="https://example.com/second"),
        SearchResult(title="Third", url="https://example.com/third"),
    ]

    output = RetrievalFilter(RetrievalFilterConfig(max_results=2)).apply(results)

    assert [result.title for result in output.results] == ["First", "Second"]
    assert output.report.input_count == 3
    assert output.report.output_count == 2
    assert output.report.truncated_count == 1


def test_retrieval_filter_drops_urls_that_cannot_be_fetched_over_http() -> None:
    results = [
        SearchResult(title="Relative redirect", url="/goto?url=opaque-token"),
        SearchResult(title="Protocol relative", url="//example.com/article"),
        SearchResult(title="Unsupported scheme", url="ftp://example.com/article"),
        SearchResult(title="Missing host", url="https:///article"),
        SearchResult(title="HTTP", url="http://example.com/first"),
        SearchResult(title="HTTPS", url="https://example.com/second"),
    ]

    output = RetrievalFilter().apply(results)

    assert [result.title for result in output.results] == ["HTTP", "HTTPS"]
    assert output.report.input_count == 6
    assert output.report.output_count == 2
    assert output.report.dropped_invalid_url == 4


def test_invalid_urls_do_not_consume_the_result_limit() -> None:
    results = [
        SearchResult(title="Invalid", url="/goto?url=opaque-token"),
        SearchResult(title="First valid", url="https://example.com/first"),
        SearchResult(title="Second valid", url="https://example.com/second"),
    ]

    output = RetrievalFilter(RetrievalFilterConfig(max_results=1)).apply(results)

    assert [result.title for result in output.results] == ["First valid"]
    assert output.report.dropped_invalid_url == 1
    assert output.report.truncated_count == 1


def test_normalize_url_removes_tracking_params_and_fragments() -> None:
    assert (
        normalize_url(
            "HTTPS://Example.COM/news/?b=2&utm_source=x&a=1#section",
            ignored_query_params={"utm_source"},
        )
        == "https://example.com/news?a=1&b=2"
    )

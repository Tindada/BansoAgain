"""Search result filtering utilities."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field

from banso.retrieval.models import SearchResult


class RetrievalFilterConfig(BaseModel):
    """Configuration for basic search result filtering."""

    max_results: int = 10
    require_title: bool = True
    require_url: bool = True
    deduplicate_urls: bool = True
    ignored_query_params: set[str] = Field(
        default_factory=lambda: {
            "fbclid",
            "gclid",
            "mc_cid",
            "mc_eid",
            "utm_campaign",
            "utm_content",
            "utm_medium",
            "utm_source",
            "utm_term",
        }
    )


class RetrievalFilterReport(BaseModel):
    """Summary of filtering decisions."""

    input_count: int
    output_count: int
    dropped_empty_title: int = 0
    dropped_empty_url: int = 0
    dropped_invalid_url: int = 0
    dropped_duplicate_url: int = 0
    truncated_count: int = 0


class RetrievalFilterResult(BaseModel):
    """Filtered search results plus decision metadata."""

    results: list[SearchResult]
    report: RetrievalFilterReport


class RetrievalFilter:
    """Applies basic quality filters to search results."""

    def __init__(self, config: RetrievalFilterConfig | None = None) -> None:
        self.config = config or RetrievalFilterConfig()

    def apply(self, results: list[SearchResult]) -> RetrievalFilterResult:
        seen_urls: set[str] = set()
        filtered: list[SearchResult] = []
        report = RetrievalFilterReport(input_count=len(results), output_count=0)

        for result in results:
            title = result.title.strip()
            url = result.url.strip()

            if self.config.require_title and not title:
                report.dropped_empty_title += 1
                continue

            if self.config.require_url and not url:
                report.dropped_empty_url += 1
                continue

            if not _is_readable_http_url(url):
                report.dropped_invalid_url += 1
                continue

            if self.config.deduplicate_urls:
                normalized_url = normalize_url(
                    url,
                    ignored_query_params=self.config.ignored_query_params,
                )
                if normalized_url in seen_urls:
                    report.dropped_duplicate_url += 1
                    continue
                seen_urls.add(normalized_url)

            if len(filtered) >= self.config.max_results:
                report.truncated_count += 1
                continue

            filtered.append(result)

        report.output_count = len(filtered)
        return RetrievalFilterResult(results=filtered, report=report)


def _is_readable_http_url(url: str) -> bool:
    """Return whether a URL can be passed directly to an HTTP document reader."""

    try:
        parsed = urlsplit(url)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def normalize_url(url: str, *, ignored_query_params: set[str] | None = None) -> str:
    """Normalize URLs for duplicate detection."""

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

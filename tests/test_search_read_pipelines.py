"""Direct tests for the executor's search and read helpers."""

import asyncio

from banso.artifacts.store import InMemoryArtifactStore
from banso.agent.action import RetrievalRoute
from banso.agent.executors.read import execute_read
from banso.agent.executors.retry import RetryPolicy
from banso.agent.executors.search import (
    SearchFailure,
    SearchSuccess,
    execute_search,
)
from banso.agent.observation import ExtractionFailure
from banso.agent.state import AgentState, UserQuery
from banso.documents.extractor import (
    EvidenceExtractionError,
    EvidenceExtractionRequest,
)
from banso.documents.fetcher import DocumentFetchError, DocumentFetchRequest
from banso.documents.models import Document, DocumentEvidence
from banso.retrieval.filter import RetrievalFilter
from banso.retrieval.models import SearchResult
from banso.retrieval.provider import RetrievalError, SearchRequest
from banso.retrieval.source_classifier import SourceClassifier


class StaticProvider:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        self.requests.append(request)
        return self.results


class FailingProvider:
    def __init__(self, *, failures: int, retryable: bool) -> None:
        self.failures = failures
        self.retryable = retryable
        self.attempt_count = 0

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        self.attempt_count += 1
        if self.attempt_count <= self.failures:
            raise RetrievalError(
                provider="test",
                reason="transport" if self.retryable else "http_status",
                status_code=None if self.retryable else 400,
                message="retrieval failed",
                source_error_type="TestError",
            )
        return [
            SearchResult(
                id="recovered-result",
                title="Recovered",
                url="https://example.com/recovered",
            )
        ]


class RecordingFetcher:
    def __init__(self, *, redirect_url: str | None = None) -> None:
        self.redirect_url = redirect_url
        self.requests: list[DocumentFetchRequest] = []

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        self.requests.append(request)
        suffix = request.url.rsplit("/", 1)[-1]
        return Document(
            id=f"document-{suffix}",
            title=request.title or suffix,
            url=self.redirect_url or request.url,
            text=f"body for {suffix}",
        )


class FlakyFetcher:
    def __init__(self) -> None:
        self.attempt_count = 0

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        self.attempt_count += 1
        if self.attempt_count == 1:
            raise DocumentFetchError(
                url=request.url,
                reason="timeout",
                message="timed out",
                source_error_type="TimeoutError",
            )
        return Document(
            id="document-after-retry",
            title=request.title,
            url=request.url,
            text="body after retry",
        )


class FirstFetchFailingFetcher(RecordingFetcher):
    async def fetch(self, request: DocumentFetchRequest) -> Document:
        if not self.requests:
            self.requests.append(request)
            raise DocumentFetchError(
                url=request.url,
                status_code=403,
                reason="http_status",
                message="forbidden",
                source_error_type="HTTPStatusError",
            )
        return await super().fetch(request)


class TrackingFetcher:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def fetch(self, request: DocumentFetchRequest) -> Document:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        index = int(request.url.rsplit("/", 1)[-1])
        await asyncio.sleep((12 - index) * 0.001)
        self.active -= 1
        return Document(
            id=f"document-{index}",
            title=request.title,
            url=request.url,
            text=f"body for {index}",
        )


class RecordingExtractor:
    def __init__(self) -> None:
        self.requests: list[EvidenceExtractionRequest] = []

    async def extract(self, request: EvidenceExtractionRequest) -> str:
        self.requests.append(request)
        return f"evidence from {request.document.id}"


class TrackingExtractor(RecordingExtractor):
    def __init__(self, fail_document: str | None = None) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.fail_document = fail_document
        self.call_counts: dict[str, int] = {}

    async def extract(self, request: EvidenceExtractionRequest) -> str:
        self.requests.append(request)
        document_id = request.document.id
        self.call_counts[document_id] = self.call_counts.get(document_id, 0) + 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if document_id == self.fail_document:
            raise EvidenceExtractionError("failed", reason="llm_error")
        return document_id


class EmptyExtractor:
    async def extract(self, request: EvidenceExtractionRequest) -> None:
        return None


def _candidates(count: int) -> list[SearchResult]:
    return [
        SearchResult(
            id=f"result-{index}",
            title=f"Result {index}",
            url=f"https://example.com/{index}",
        )
        for index in range(count)
    ]


def _run_search(
    request: SearchRequest,
    state: AgentState,
    provider,
    *,
    store: InMemoryArtifactStore | None = None,
) -> SearchSuccess | SearchFailure:
    return asyncio.run(
        execute_search(
            request,
            state,
            store=store or InMemoryArtifactStore(),
            retrieval_provider=provider,
            retrieval_filter=RetrievalFilter(),
            source_classifier=SourceClassifier(),
            retry_policy=RetryPolicy(delay_seconds=0),
            route=RetrievalRoute.WEB,
        )
    )


def _run_read(
    candidates: list[SearchResult],
    state: AgentState,
    fetcher,
    extractor,
    *,
    store: InMemoryArtifactStore | None = None,
    limit: int,
    max_extraction_concurrency: int = 4,
):
    return asyncio.run(
        execute_read(
            candidates,
            state,
            store=store or InMemoryArtifactStore(),
            document_fetcher=fetcher,
            evidence_extractor=extractor,
            ignored_query_params=set(),
            limit=limit,
            max_extraction_concurrency=max_extraction_concurrency,
            retry_policy=RetryPolicy(delay_seconds=0),
            route=RetrievalRoute.WEB,
        )
    )


def test_search_runs_provider_and_processes_results_without_reading() -> None:
    store = InMemoryArtifactStore()
    existing = SearchResult(
        id="existing-result",
        title="Existing",
        url="https://example.com/existing",
    )
    store.put(existing)
    state = AgentState(
        query=UserQuery(text="user question"),
        search_result_index={existing.url: existing.id},
    )
    provider = StaticProvider(
        [
            SearchResult(
                id="duplicate-result",
                title="Existing again",
                url="https://example.com/existing",
            ),
            SearchResult(
                id="new-result",
                title="New",
                url="https://example.com/new",
            ),
        ]
    )
    request = SearchRequest(query="specific need", source_domains=["example.com"])

    outcome = _run_search(request, state, provider, store=store)

    assert isinstance(outcome, SearchSuccess)
    assert provider.requests == [request]
    assert outcome.search_result_ids == ["existing-result", "new-result"]
    assert outcome.search_result_merge_report.new_result_count == 1
    assert outcome.search_result_merge_report.reused_result_count == 1
    assert outcome.search_result_index_updates == {
        "https://example.com/new": "new-result"
    }
    assert store.list(Document) == []
    assert store.list(DocumentEvidence) == []


def test_search_retries_a_transient_provider_failure() -> None:
    provider = FailingProvider(failures=1, retryable=True)
    outcome = _run_search(
        SearchRequest(query="query"),
        AgentState(query=UserQuery(text="question")),
        provider,
    )

    assert isinstance(outcome, SearchSuccess)
    assert provider.attempt_count == 2
    assert outcome.search_result_ids == ["recovered-result"]


def test_search_does_not_retry_a_terminal_provider_failure() -> None:
    provider = FailingProvider(failures=2, retryable=False)
    outcome = _run_search(
        SearchRequest(query="query"),
        AgentState(query=UserQuery(text="question")),
        provider,
    )

    assert isinstance(outcome, SearchFailure)
    assert provider.attempt_count == 1
    assert outcome.status_code == 400


def test_read_fetches_and_extracts_only_the_supplied_candidates() -> None:
    store = InMemoryArtifactStore()
    candidates = _candidates(2)
    fetcher = RecordingFetcher()
    extractor = RecordingExtractor()
    state = AgentState(query=UserQuery(text="user question"))

    result = _run_read(
        candidates,
        state,
        fetcher,
        extractor,
        store=store,
        limit=1,
    )

    assert [request.url for request in fetcher.requests] == [
        "https://example.com/0"
    ]
    assert [request.query for request in extractor.requests] == ["user question"]
    assert len(result.fetch_outcomes) == 1
    assert len(result.extraction_outcomes) == 1
    assert result.document_index_updates == {
        "https://example.com/0": "document-0"
    }
    assert len(store.list(Document)) == 1
    assert len(store.list(DocumentEvidence)) == 1


def test_read_retries_fetch_within_the_read() -> None:
    fetcher = FlakyFetcher()
    result = _run_read(
        _candidates(1),
        AgentState(query=UserQuery(text="question")),
        fetcher,
        RecordingExtractor(),
        limit=1,
    )

    assert fetcher.attempt_count == 2
    assert result.fetch_outcomes[0].attempt_count == 2
    assert len(result.extraction_outcomes) == 1


def test_read_fetches_fallbacks_until_reaching_the_document_limit() -> None:
    fetcher = FirstFetchFailingFetcher()
    result = _run_read(
        _candidates(3),
        AgentState(query=UserQuery(text="question")),
        fetcher,
        RecordingExtractor(),
        limit=2,
    )

    assert len(fetcher.requests) == 3
    assert [outcome.status for outcome in result.fetch_outcomes] == [
        "failure",
        "success",
        "success",
    ]
    assert len(result.extraction_outcomes) == 2


def test_read_deduplicates_documents_redirected_within_a_fetch_batch() -> None:
    store = InMemoryArtifactStore()
    fetcher = RecordingFetcher(redirect_url="https://example.com/canonical")
    result = _run_read(
        _candidates(2),
        AgentState(query=UserQuery(text="question")),
        fetcher,
        RecordingExtractor(),
        store=store,
        limit=2,
    )

    assert len(fetcher.requests) == 2
    assert len(store.list(Document)) == 1
    assert len(result.document_index_updates) == 1
    assert {outcome.document_id for outcome in result.fetch_outcomes} == {
        "document-0"
    }


def test_read_limits_fetch_concurrency_and_preserves_order() -> None:
    fetcher = TrackingFetcher()
    result = _run_read(
        _candidates(12),
        AgentState(query=UserQuery(text="question")),
        fetcher,
        RecordingExtractor(),
        limit=12,
    )

    assert fetcher.max_active == 10
    assert [outcome.document_id for outcome in result.fetch_outcomes] == [
        f"document-{index}" for index in range(12)
    ]


def test_read_limits_extraction_concurrency_and_preserves_order() -> None:
    extractor = TrackingExtractor()
    result = _run_read(
        _candidates(4),
        AgentState(query=UserQuery(text="question")),
        RecordingFetcher(),
        extractor,
        limit=4,
        max_extraction_concurrency=2,
    )

    assert extractor.max_active == 2
    assert [outcome.document_id for outcome in result.extraction_outcomes] == [
        f"document-{index}" for index in range(4)
    ]


def test_read_isolates_a_known_extraction_failure() -> None:
    extractor = TrackingExtractor("document-1")
    result = _run_read(
        _candidates(2),
        AgentState(query=UserQuery(text="question")),
        RecordingFetcher(),
        extractor,
        limit=2,
        max_extraction_concurrency=2,
    )

    failures = [
        outcome
        for outcome in result.extraction_outcomes
        if isinstance(outcome, ExtractionFailure)
    ]
    assert len(failures) == 1
    assert failures[0].document_id == "document-1"
    assert failures[0].attempt_count == 2
    assert extractor.call_counts["document-1"] == 2


def test_read_does_not_store_empty_evidence() -> None:
    store = InMemoryArtifactStore()
    result = _run_read(
        _candidates(2),
        AgentState(query=UserQuery(text="question")),
        RecordingFetcher(),
        EmptyExtractor(),
        store=store,
        limit=2,
    )

    assert all(
        outcome.evidence_id is None for outcome in result.extraction_outcomes
    )
    assert store.list(DocumentEvidence) == []

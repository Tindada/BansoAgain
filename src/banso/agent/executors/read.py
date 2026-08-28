"""Document reading helpers used by the news action executor."""

import asyncio
from collections import deque
from dataclasses import dataclass

from banso.artifacts.store import ArtifactStore
from banso.agent.action import RetrievalRoute
from banso.agent.executors.retry import AttemptResult, RetryPolicy, run_with_retry
from banso.agent.observation import (
    DocumentFetchFailure,
    EvidenceExtractionFailure,
    ExtractionFailure,
    ExtractionOutcome,
    ExtractionSuccess,
    FetchFailure,
    FetchOutcome,
    FetchSuccess,
)
from banso.agent.state import AgentState
from banso.documents.extractor import (
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceExtractor,
)
from banso.documents.fetcher import (
    DocumentFetchError,
    DocumentFetchRequest,
    DocumentFetcher,
)
from banso.documents.models import Document, DocumentEvidence
from banso.retrieval.models import SearchResult
from banso.retrieval.url_utils import normalize_url
from banso.tracing.trace import start_span


_MAX_FETCH_CONCURRENCY = 10


@dataclass(frozen=True)
class ReadResult:
    """Fetch and extraction outcomes produced from selected search results."""

    fetch_outcomes: list[FetchOutcome]
    extraction_outcomes: list[ExtractionOutcome]
    document_index_updates: dict[str, str]


async def execute_read(
    candidates: list[SearchResult],
    state: AgentState,
    *,
    store: ArtifactStore,
    document_fetcher: DocumentFetcher,
    evidence_extractor: EvidenceExtractor,
    ignored_query_params: set[str],
    limit: int,
    max_extraction_concurrency: int,
    retry_policy: RetryPolicy,
    route: RetrievalRoute,
) -> ReadResult:
    """Fetch selected result pages and extract user-query-relevant evidence."""
    with start_span(
        "news.research.fetch",
        attributes={"route": route.value},
    ) as span:
        fetch_outcomes, document_index_updates = await _fetch_from_queue(
            candidates,
            limit,
            state,
            store=store,
            document_fetcher=document_fetcher,
            ignored_query_params=ignored_query_params,
            retry_policy=retry_policy,
        )
        span.set_output({"fetch_outcomes": fetch_outcomes})

    document_ids = _documents_to_extract(fetch_outcomes, state)
    with start_span(
        "news.research.extract",
        attributes={"route": route.value},
    ) as span:
        extraction_outcomes = await _extract_documents(
            document_ids,
            state,
            store=store,
            evidence_extractor=evidence_extractor,
            max_extraction_concurrency=max_extraction_concurrency,
            retry_policy=retry_policy,
        )
        span.set_output({"extraction_outcomes": extraction_outcomes})

    return ReadResult(
        fetch_outcomes=fetch_outcomes,
        extraction_outcomes=extraction_outcomes,
        document_index_updates=document_index_updates,
    )


def _documents_to_extract(
    fetch_outcomes: list[FetchOutcome],
    state: AgentState,
) -> list[str]:
    """Return newly fetched unique documents that require extraction."""
    return [
        document_id
        for document_id in dict.fromkeys(
            outcome.document_id
            for outcome in fetch_outcomes
            if isinstance(outcome, FetchSuccess)
        )
        if document_id not in state.documents
    ]


async def _fetch_from_queue(
    candidates: list[SearchResult],
    limit: int,
    state: AgentState,
    *,
    store: ArtifactStore,
    document_fetcher: DocumentFetcher,
    ignored_query_params: set[str],
    retry_policy: RetryPolicy,
) -> tuple[list[FetchOutcome], dict[str, str]]:
    outcomes: list[FetchOutcome] = []
    document_index = dict(state.document_index)
    document_index_updates: dict[str, str] = {}
    new_document_count = 0
    queue = deque(candidates)

    while queue and new_document_count < limit:
        batch_size = min(
            _MAX_FETCH_CONCURRENCY,
            limit - new_document_count,
            len(queue),
        )
        batch = [queue.popleft() for _ in range(batch_size)]

        async def fetch(
            result: SearchResult,
        ) -> tuple[
            SearchResult,
            str | None,
            AttemptResult[Document, DocumentFetchError] | None,
        ]:
            document_id = document_index.get(
                normalize_url(
                    result.url,
                    ignored_query_params=ignored_query_params,
                )
            )
            if document_id is not None:
                return result, document_id, None

            request = DocumentFetchRequest(
                url=result.url,
                title=result.title,
                source=result.source,
                metadata={"search_result_id": result.id},
            )
            attempt = await run_with_retry(
                lambda: document_fetcher.fetch(request),
                error_type=DocumentFetchError,
                is_retryable=lambda error: error.retryable,
                policy=retry_policy,
            )
            return result, None, attempt

        batch_entries = await asyncio.gather(
            *(fetch(result) for result in batch)
        )
        for result, document_id, attempt in batch_entries:
            if document_id is not None:
                outcomes.append(
                    FetchSuccess(
                        search_result_id=result.id,
                        document_id=document_id,
                        attempt_count=0,
                    )
                )
                continue

            if attempt is None:
                raise AssertionError("uncached fetch entry has no attempt")
            if attempt.error is not None:
                error = attempt.error
                outcomes.append(
                    FetchFailure(
                        search_result_id=result.id,
                        failure=DocumentFetchFailure(
                            url=error.url,
                            status_code=error.status_code,
                            reason=error.reason,
                            message=error.message,
                            source_error_type=error.source_error_type,
                        ),
                        attempt_count=attempt.attempt_count,
                    )
                )
                continue
            document = attempt.value
            if document is None:
                raise AssertionError("successful fetch attempt returned no document")

            normalized_document_url = normalize_url(
                document.url,
                ignored_query_params=ignored_query_params,
            )
            document_id = document_index.get(normalized_document_url)
            if document_id is None:
                document_id = store.put(document)
                document_index[normalized_document_url] = document_id
                document_index_updates[normalized_document_url] = document_id
                new_document_count += 1

            outcomes.append(
                FetchSuccess(
                    search_result_id=result.id,
                    document_id=document_id,
                    attempt_count=attempt.attempt_count,
                )
            )

    return outcomes, document_index_updates


async def _extract_documents(
    document_ids: list[str],
    state: AgentState,
    *,
    store: ArtifactStore,
    evidence_extractor: EvidenceExtractor,
    max_extraction_concurrency: int,
    retry_policy: RetryPolicy,
) -> list[ExtractionOutcome]:
    documents: list[Document] = []
    for document_id in document_ids:
        document = store.get(document_id, Document)
        if document is None:
            raise ValueError(
                "Document artifact is missing or has the wrong type: "
                f"{document_id}"
            )
        documents.append(document)

    semaphore = asyncio.Semaphore(max_extraction_concurrency)

    async def extract(
        document: Document,
    ) -> tuple[
        Document,
        str | None,
        EvidenceExtractionError | None,
        int,
    ]:
        async with semaphore:
            request = EvidenceExtractionRequest(
                query=state.query.text,
                document=document,
            )
            attempt = await run_with_retry(
                lambda: evidence_extractor.extract(request),
                error_type=EvidenceExtractionError,
                is_retryable=lambda error: error.retryable,
                policy=retry_policy,
            )
            return (
                document,
                attempt.value,
                attempt.error,
                attempt.attempt_count,
            )

    results = await asyncio.gather(*(extract(document) for document in documents))
    outcomes: list[ExtractionOutcome] = []
    for document, evidence, error, attempt_count in results:
        if error is not None:
            outcomes.append(
                ExtractionFailure(
                    document_id=document.id,
                    failure=EvidenceExtractionFailure(
                        url=document.url,
                        reason=error.reason,
                        message=str(error),
                    ),
                    attempt_count=attempt_count,
                )
            )
            continue
        evidence_id = (
            store.put(DocumentEvidence(document_id=document.id, text=evidence))
            if evidence is not None
            else None
        )
        outcomes.append(
            ExtractionSuccess(
                document_id=document.id,
                evidence_id=evidence_id,
                attempt_count=attempt_count,
            )
        )
    return outcomes

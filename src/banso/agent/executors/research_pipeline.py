"""Internal retrieval-to-evidence pipeline for one research action."""

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from banso.artifacts.store import ArtifactStore
from banso.agent.action import ResearchActionParams, RetrievalRoute
from banso.agent.observation import (
    CompletedResearchObservation,
    DocumentFetchFailure,
    EvidenceExtractionFailure,
    ExtractionFailure,
    ExtractionOutcome,
    ExtractionSuccess,
    FetchFailure,
    FetchOutcome,
    FetchSuccess,
    ResearchObservation,
    RetrievalFailedResearchObservation,
)
from banso.agent.selection.passthrough_selector import (
    PassthroughSearchResultSelector,
)
from banso.agent.selection.selector import (
    SearchResultSelection,
    SearchResultSelectionRequest,
    SearchResultSelector,
)
from banso.agent.state import AgentState, SearchResultState
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
from banso.documents.models import Document, EvidenceItem
from banso.agent.executors.retry import RetryPolicy, run_with_retry
from banso.retrieval.filter import RetrievalFilter
from banso.retrieval.models import (
    RetrievalFilterReport,
    SearchResult,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)
from banso.retrieval.provider import RetrievalError, RetrievalProvider, SearchRequest
from banso.retrieval.source_classifier import SourceClassifier
from banso.retrieval.url_utils import normalize_url
from banso.tracing.trace import start_span


@dataclass(frozen=True)
class ResearchRouteComponents:
    """Retrieval and document-fetch components for one semantic route."""

    retrieval_provider: RetrievalProvider
    document_fetcher: DocumentFetcher


class ResearchPipeline:
    """Run retrieval, selection, fetching, and extraction as one operation."""

    def __init__(
        self,
        store: ArtifactStore,
        research_routes: Mapping[RetrievalRoute, ResearchRouteComponents],
        evidence_extractor: EvidenceExtractor,
        *,
        retrieval_filter: RetrievalFilter | None = None,
        source_classifier: SourceClassifier | None = None,
        search_result_selector: SearchResultSelector | None = None,
        max_extraction_concurrency: int = 4,
        fetch_retry_policy: RetryPolicy | None = None,
        extraction_retry_policy: RetryPolicy | None = None,
        retrieval_retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not research_routes:
            raise ValueError("research_routes must contain at least one route")
        if max_extraction_concurrency < 1:
            raise ValueError("max_extraction_concurrency must be at least 1")

        self.store = store
        self.research_routes = dict(research_routes)
        self.evidence_extractor = evidence_extractor
        self.retrieval_filter = retrieval_filter or RetrievalFilter()
        self.source_classifier = source_classifier or SourceClassifier()
        self.search_result_selector = (
            search_result_selector or PassthroughSearchResultSelector()
        )
        self.max_extraction_concurrency = max_extraction_concurrency
        self.fetch_retry_policy = fetch_retry_policy or RetryPolicy()
        self.extraction_retry_policy = extraction_retry_policy or RetryPolicy()
        self.retrieval_retry_policy = retrieval_retry_policy or RetryPolicy()

    async def run(
        self,
        params: ResearchActionParams,
        state: AgentState,
    ) -> ResearchObservation:
        """Execute one route-specific research pipeline."""
        components = self.research_routes.get(params.route)
        if components is None:
            raise ValueError(f"research route is not enabled: {params.route.value}")

        with start_span(
            "news.research.retrieve",
            input=params.model_dump(
                mode="json",
                include={"query", "source_domains"},
                exclude_none=True,
            ),
            attributes={"route": params.route.value},
        ) as span:
            request = SearchRequest(
                query=params.query,
                language=state.query.language,
                region=state.query.region,
                time_range=state.query.time_range,
                source_domains=params.source_domains,
            )
            attempt = await run_with_retry(
                lambda: components.retrieval_provider.search(request),
                error_type=RetrievalError,
                is_retryable=lambda error: error.retryable,
                policy=self.retrieval_retry_policy,
            )
            if attempt.error is not None:
                failure = RetrievalFailedResearchObservation(
                    query=params.query,
                    route=params.route,
                    source_domains=params.source_domains,
                    provider=attempt.error.provider,
                    reason=attempt.error.reason,
                    status_code=attempt.error.status_code,
                    message=attempt.error.message,
                    source_error_type=attempt.error.source_error_type,
                    retryable=attempt.error.retryable,
                    attempt_count=attempt.attempt_count,
                )
                span.set_attribute("outcome", "failure")
                span.set_output({"retrieval_failure": failure})
                return failure
            if attempt.value is None:
                raise AssertionError("successful retrieval returned no result")
            (
                search_result_ids,
                search_result_index_updates,
                merge_report,
                filter_report,
                classification_report,
            ) = self._process_retrieval_results(attempt.value, state)
            span.set_attribute("outcome", "success")
            span.set_output(
                {
                    "search_result_ids": search_result_ids,
                    "filter_report": filter_report,
                    "classification_report": classification_report,
                    "merge_report": merge_report,
                }
            )

        with start_span(
            "news.research.select",
            attributes={"route": params.route.value},
        ) as span:
            candidates = self._candidate_results(search_result_ids, state)
            if candidates:
                selection = await self.search_result_selector.select(
                    SearchResultSelectionRequest(
                        research_query=params.query,
                        candidates=candidates,
                        state=state,
                    )
                )
                selected_results = self._resolve_selection(selection, candidates)
            else:
                selected_results = []
            selection_report = SearchResultSelectionReport(
                candidate_ids=[result.id for result in candidates],
                selected_ids=[result.id for result in selected_results],
            )
            span.set_output({"selection_report": selection_report})

        with start_span(
            "news.research.fetch",
            attributes={"route": params.route.value},
        ) as span:
            fetch_outcomes, document_index_updates = await self._fetch_from_queue(
                selected_results,
                state.budget.max_results_per_research,
                state,
                components.document_fetcher,
            )
            span.set_output({"fetch_outcomes": fetch_outcomes})

        document_ids = self._documents_to_extract(fetch_outcomes, state)
        with start_span(
            "news.research.extract",
            attributes={"route": params.route.value},
        ) as span:
            extraction_outcomes = await self._extract_selected(
                document_ids,
                state,
            )
            span.set_output({"extraction_outcomes": extraction_outcomes})

        return CompletedResearchObservation(
            query=params.query,
            route=params.route,
            source_domains=params.source_domains,
            search_result_ids=search_result_ids,
            retrieval_filter_report=filter_report,
            source_classification_report=classification_report,
            search_result_merge_report=merge_report,
            selection_report=selection_report,
            fetch_outcomes=fetch_outcomes,
            extraction_outcomes=extraction_outcomes,
            search_result_index_updates=search_result_index_updates,
            document_index_updates=document_index_updates,
        )

    def _process_retrieval_results(
        self,
        raw_results: list[SearchResult],
        state: AgentState,
    ) -> tuple[
        list[str],
        dict[str, str],
        SearchResultMergeReport,
        RetrievalFilterReport,
        SourceClassificationReport,
    ]:
        filtered = self.retrieval_filter.apply(raw_results)
        classified = self.source_classifier.apply(filtered.results)
        index_updates: dict[str, str] = {}
        result_ids: list[str] = []
        new_result_count = 0
        reused_result_count = 0

        for result in classified.results:
            normalized_url = self._normalize_url(result.url)
            existing_result_id = state.search_result_index.get(normalized_url)
            if existing_result_id is not None:
                result_ids.append(existing_result_id)
                reused_result_count += 1
                continue

            result_id = self.store.put(result)
            index_updates[normalized_url] = result_id
            result_ids.append(result_id)
            new_result_count += 1

        return (
            result_ids,
            index_updates,
            SearchResultMergeReport(
                candidate_count=len(classified.results),
                new_result_count=new_result_count,
                reused_result_count=reused_result_count,
            ),
            filtered.report,
            classified.report,
        )

    def _candidate_results(
        self,
        retrieved_ids: list[str],
        state: AgentState,
    ) -> list[SearchResult]:
        candidates: list[SearchResult] = []
        for result_id in dict.fromkeys([*retrieved_ids, *state.search_results]):
            result_state = state.search_results.get(result_id, SearchResultState())
            if result_state.document_id is None and result_state.failure is None:
                result = self.store.get(result_id, SearchResult)
                if result is None:
                    raise ValueError(
                        "SearchResult artifact is missing or has the wrong type: "
                        f"{result_id}"
                    )
                candidates.append(result)
        return candidates

    @staticmethod
    def _resolve_selection(
        selection: SearchResultSelection,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        selected_id_set = set(selection.selected_ids)
        unknown_ids = selected_id_set - {candidate.id for candidate in candidates}
        if unknown_ids:
            raise ValueError(
                "search result selector returned unknown IDs: "
                + ", ".join(sorted(unknown_ids))
            )
        return [
            candidate for candidate in candidates if candidate.id in selected_id_set
        ]

    @staticmethod
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
        self,
        candidates: list[SearchResult],
        limit: int,
        state: AgentState,
        document_fetcher: DocumentFetcher,
    ) -> tuple[list[FetchOutcome], dict[str, str]]:
        outcomes: list[FetchOutcome] = []
        document_index = dict(state.document_index)
        document_index_updates: dict[str, str] = {}
        new_document_count = 0
        queue = deque(candidates)

        while queue and new_document_count < limit:
            result = queue.popleft()
            result_id = result.id

            document_id = document_index.get(self._normalize_url(result.url))
            if document_id is not None:
                outcomes.append(
                    FetchSuccess(
                        search_result_id=result_id,
                        document_id=document_id,
                        attempt_count=0,
                    )
                )
                continue

            request = DocumentFetchRequest(
                url=result.url,
                title=result.title,
                source=result.source,
                metadata={"search_result_id": result_id},
            )
            attempt = await run_with_retry(
                lambda: document_fetcher.fetch(request),
                error_type=DocumentFetchError,
                is_retryable=lambda error: error.retryable,
                policy=self.fetch_retry_policy,
            )
            if attempt.error is not None:
                error = attempt.error
                outcomes.append(
                    FetchFailure(
                        search_result_id=result_id,
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

            normalized_document_url = self._normalize_url(document.url)
            document_id = document_index.get(normalized_document_url)
            if document_id is None:
                document_id = self.store.put(document)
                document_index[normalized_document_url] = document_id
                document_index_updates[normalized_document_url] = document_id
                new_document_count += 1

            outcomes.append(
                FetchSuccess(
                    search_result_id=result_id,
                    document_id=document_id,
                    attempt_count=attempt.attempt_count,
                )
            )

        return outcomes, document_index_updates

    async def _extract_selected(
        self,
        document_ids: list[str],
        state: AgentState,
    ) -> list[ExtractionOutcome]:
        documents: list[Document] = []
        for document_id in document_ids:
            document = self.store.get(document_id, Document)
            if document is None:
                raise ValueError(
                    f"Document artifact is missing or has the wrong type: {document_id}"
                )
            documents.append(document)

        semaphore = asyncio.Semaphore(self.max_extraction_concurrency)

        async def extract(
            document: Document,
        ) -> tuple[
            Document,
            list[EvidenceItem] | None,
            EvidenceExtractionError | None,
            int,
        ]:
            async with semaphore:
                request = EvidenceExtractionRequest(
                    query=state.query.text,
                    document=document,
                )
                attempt = await run_with_retry(
                    lambda: self.evidence_extractor.extract(request),
                    error_type=EvidenceExtractionError,
                    is_retryable=lambda error: error.retryable,
                    policy=self.extraction_retry_policy,
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
            if evidence is None:
                raise AssertionError("successful extraction returned no evidence list")
            for item in evidence:
                if item.document_id != document.id:
                    raise ValueError(
                        f"EvidenceItem {item.id} references document "
                        f"{item.document_id}, expected {document.id}"
                    )
            outcomes.append(
                ExtractionSuccess(
                    document_id=document.id,
                    evidence_ids=[self.store.put(item) for item in evidence],
                    attempt_count=attempt_count,
                )
            )
        return outcomes

    def _normalize_url(self, url: str) -> str:
        return normalize_url(
            url,
            ignored_query_params=self.retrieval_filter.config.ignored_query_params,
        )

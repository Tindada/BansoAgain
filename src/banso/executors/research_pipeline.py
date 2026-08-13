"""Internal retrieval-to-evidence pipeline for one research action."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from banso.artifacts import ArtifactStore
from banso.core.action import ResearchActionParams, RetrievalRoute
from banso.core.observation import (
    DocumentFetchFailure,
    EvidenceExtractionFailure,
    ExtractionFailure,
    ExtractionOutcome,
    ExtractionSuccess,
    FetchFailure,
    FetchOutcome,
    FetchSuccess,
    ResearchObservation,
    RetrievalFilterReport,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)
from banso.core.state import AgentState, SearchResultState
from banso.documents import (
    Document,
    DocumentFetchError,
    DocumentFetchRequest,
    DocumentFetcher,
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceExtractor,
    EvidenceItem,
)
from banso.retrieval import (
    RetrievalProvider,
    SearchRequest,
    SearchResult,
    SourceClassifier,
    normalize_url,
)
from banso.retrieval.filter import RetrievalFilter
from banso.executors.retry import RetryPolicy, run_with_retry
from banso.tracing import start_span


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
        max_extraction_concurrency: int = 4,
        fetch_retry_policy: RetryPolicy | None = None,
        extraction_retry_policy: RetryPolicy | None = None,
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
        self.max_extraction_concurrency = max_extraction_concurrency
        self.fetch_retry_policy = fetch_retry_policy or RetryPolicy()
        self.extraction_retry_policy = extraction_retry_policy or RetryPolicy()

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
            input={"query": params.query},
            attributes={"route": params.route.value},
        ) as span:
            (
                search_result_ids,
                search_result_index_updates,
                merge_report,
                filter_report,
                classification_report,
            ) = await self._retrieve(params, state, components.retrieval_provider)
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
            selection_report = self._select_results(search_result_ids, state)
            span.set_output(selection_report)

        with start_span(
            "news.research.fetch",
            attributes={"route": params.route.value},
        ) as span:
            fetch_outcomes, document_index_updates = await self._fetch_selected(
                selection_report.selected_ids,
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

        return ResearchObservation(
            query=params.query,
            route=params.route,
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

    async def _retrieve(
        self,
        params: ResearchActionParams,
        state: AgentState,
        retrieval_provider: RetrievalProvider,
    ) -> tuple[
        list[str],
        dict[str, str],
        SearchResultMergeReport,
        RetrievalFilterReport,
        SourceClassificationReport,
    ]:
        raw_results = await retrieval_provider.search(
            SearchRequest(
                query=params.query,
                language=state.query.language,
                region=state.query.region,
                time_range=state.query.time_range,
            )
        )
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

    @staticmethod
    def _select_results(
        retrieved_ids: list[str],
        state: AgentState,
    ) -> SearchResultSelectionReport:
        candidate_ids: list[str] = []
        for result_id in dict.fromkeys([*retrieved_ids, *state.search_results]):
            result_state = state.search_results.get(result_id, SearchResultState())
            if result_state.document_id is None and result_state.failure is None:
                candidate_ids.append(result_id)
        limit = min(
            state.budget.max_results_per_research,
            state.remaining_document_capacity,
        )
        return SearchResultSelectionReport(
            candidate_ids=candidate_ids,
            selected_ids=candidate_ids[:limit],
            deferred_ids=candidate_ids[limit:],
        )

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

    async def _fetch_selected(
        self,
        selected_ids: list[str],
        state: AgentState,
        document_fetcher: DocumentFetcher,
    ) -> tuple[list[FetchOutcome], dict[str, str]]:
        outcomes: list[FetchOutcome] = []
        document_index = dict(state.document_index)
        document_index_updates: dict[str, str] = {}

        for result_id in selected_ids:
            result = self.store.get(result_id, SearchResult)
            if result is None:
                raise ValueError(
                    "SearchResult artifact is missing or has the wrong type: "
                    f"{result_id}"
                )

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
                    query=state.query,
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

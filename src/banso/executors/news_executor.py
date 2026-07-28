"""News action executor."""

import asyncio

from banso.artifacts import ArtifactStore
from banso.core.action import AgentAction, AgentActionType
from banso.core.lifecycle import (
    active_document_count,
    eligible_extraction_document_ids,
    eligible_fetch_result_ids,
    remaining_document_fetches,
)
from banso.core.observation import (
    CurateEvidenceObservation,
    DocumentFetchFailure,
    EvidenceExtractionFailure,
    ExtractEvidenceObservation,
    ExtractionFailure,
    ExtractionSuccess,
    FetchDocumentsObservation,
    FetchFailure,
    FetchSuccess,
    FinishObservation,
    Observation,
    PlanSearchObservation,
    SearchObservation,
    SearchResultMergeReport,
    StopObservation,
)
from banso.core.state import AgentState
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
    OriginalQueryPlanner,
    RetrievalProvider,
    SearchRequest,
    SearchResult,
    SearchPlanningRequest,
    SearchQueryPlanner,
    SourceClassifier,
    normalize_url,
)
from banso.retrieval.filter import RetrievalFilter
from banso.synthesis import Synthesizer, SynthesisRequest


class NewsActionExecutor:
    """Executes news-domain actions using pluggable components."""

    def __init__(
        self,
        store: ArtifactStore,
        retrieval_provider: RetrievalProvider,
        document_fetcher: DocumentFetcher,
        evidence_extractor: EvidenceExtractor,
        synthesizer: Synthesizer,
        retrieval_filter: RetrievalFilter | None = None,
        source_classifier: SourceClassifier | None = None,
        search_query_planner: SearchQueryPlanner | None = None,
        max_extraction_concurrency: int = 3,
    ) -> None:
        if max_extraction_concurrency < 1:
            raise ValueError("max_extraction_concurrency must be at least 1")

        self.store = store
        self.retrieval_provider = retrieval_provider
        self.document_fetcher = document_fetcher
        self.evidence_extractor = evidence_extractor
        self.synthesizer = synthesizer
        self.retrieval_filter = retrieval_filter or RetrievalFilter()
        self.source_classifier = source_classifier or SourceClassifier()
        self.search_query_planner = search_query_planner or OriginalQueryPlanner()
        self.max_extraction_concurrency = max_extraction_concurrency

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        """Execute a news-domain action."""
        if action.type == AgentActionType.PLAN_SEARCH:
            plan = await self.search_query_planner.plan(
                SearchPlanningRequest(
                    query=state.query,
                    reference_time=state.reference_time,
                    max_searches=state.budget.max_searches,
                )
            )
            return PlanSearchObservation(search_plan=plan)

        if action.type == AgentActionType.SEARCH:
            return await self._search(action, state)

        if action.type == AgentActionType.FETCH_DOCUMENTS:
            return await self._fetch_documents(state)

        if action.type == AgentActionType.EXTRACT_EVIDENCE:
            return await self._extract_evidence(state)

        if action.type == AgentActionType.CURATE_EVIDENCE:
            return self._curate_evidence(action, state)

        if action.type == AgentActionType.FINISH:
            return await self._synthesize(state)

        return StopObservation()

    async def _search(self, action: AgentAction, state: AgentState) -> Observation:
        query = action.params.get("query", state.query.text)
        if not isinstance(query, str):
            query = state.query.text

        raw_results = await self.retrieval_provider.search(
            SearchRequest(
                query=query,
                language=state.query.language,
                region=state.query.region,
                time_range=state.query.time_range,
            )
        )
        filtered = self.retrieval_filter.apply(raw_results)
        classified = self.source_classifier.apply(filtered.results)
        results = classified.results
        index_updates: dict[str, str] = {}
        result_ids: list[str] = []
        new_result_count = 0
        reused_result_count = 0

        for result in results:
            normalized_url = normalize_url(
                result.url,
                ignored_query_params=self.retrieval_filter.config.ignored_query_params,
            )
            existing_result_id = state.search_result_index.get(normalized_url)
            if existing_result_id is not None:
                result_ids.append(existing_result_id)
                reused_result_count += 1
                continue

            result_id = self.store.put(result)
            index_updates[normalized_url] = result_id
            result_ids.append(result_id)
            new_result_count += 1

        return SearchObservation(
            search_queries=[query],
            search_result_ids=result_ids,
            search_result_index_updates=index_updates,
            search_result_merge_report=SearchResultMergeReport(
                candidate_count=len(results),
                new_result_count=new_result_count,
                reused_result_count=reused_result_count,
            ),
            retrieval_filter_report=filtered.report,
            source_classification_report=classified.report,
        )

    async def _fetch_documents(self, state: AgentState) -> Observation:
        fetch_outcomes: list[FetchSuccess | FetchFailure] = []
        document_index = dict(state.document_index)
        document_index_updates: dict[str, str] = {}
        result_ids = eligible_fetch_result_ids(state)[:remaining_document_fetches(state)]
        for result_id in result_ids:
            result = self.store.get(result_id, SearchResult)
            if result is None:
                raise ValueError(f"SearchResult artifact is missing or has the wrong type: {result_id}")

            normalized_result_url = normalize_url(
                result.url,
                ignored_query_params=self.retrieval_filter.config.ignored_query_params,
            )
            document_id = document_index.get(normalized_result_url)
            if document_id is not None:
                fetch_outcomes.append(
                    FetchSuccess(
                        search_result_id=result_id,
                        document_id=document_id,
                    )
                )
                continue

            try:
                document = await self.document_fetcher.fetch(
                    DocumentFetchRequest(
                        url=result.url,
                        title=result.title,
                        source=result.source,
                        metadata={"search_result_id": result.id},
                    )
                )
            except DocumentFetchError as error:
                fetch_outcomes.append(
                    FetchFailure(
                        search_result_id=result_id,
                        failure=DocumentFetchFailure(
                            url=error.url,
                            status_code=error.status_code,
                            reason=error.reason,
                            retryable=error.retryable,
                            message=error.message,
                            source_error_type=error.source_error_type,
                        ),
                    )
                )
                continue

            normalized_document_url = normalize_url(
                document.url,
                ignored_query_params=self.retrieval_filter.config.ignored_query_params,
            )
            document_id = document_index.get(normalized_document_url)
            if document_id is None:
                document_id = self.store.put(document)
                document_index[normalized_document_url] = document_id
                document_index_updates[normalized_document_url] = document_id

            fetch_outcomes.append(
                FetchSuccess(
                    search_result_id=result_id,
                    document_id=document_id,
                )
            )

        return FetchDocumentsObservation(
            fetch_outcomes=fetch_outcomes,
            document_index_updates=document_index_updates,
        )

    async def _extract_evidence(self, state: AgentState) -> Observation:
        documents: list[Document] = []
        for document_id in eligible_extraction_document_ids(state):
            document = self.store.get(document_id, Document)
            if document is None:
                raise ValueError(f"Document artifact is missing or has the wrong type: {document_id}")
            documents.append(document)

        semaphore = asyncio.Semaphore(self.max_extraction_concurrency)

        async def extract(
            document: Document,
        ) -> tuple[Document, list[EvidenceItem], EvidenceExtractionError | None]:
            async with semaphore:
                try:
                    evidence = await self.evidence_extractor.extract(
                        EvidenceExtractionRequest(
                            query=state.query,
                            document=document,
                        )
                    )
                except EvidenceExtractionError as error:
                    return document, [], error
                return document, evidence, None

        extraction_results = await asyncio.gather(*(extract(document) for document in documents))
        extraction_outcomes: list[ExtractionSuccess | ExtractionFailure] = []
        for document, evidence, error in extraction_results:
            if error is not None:
                extraction_outcomes.append(
                    ExtractionFailure(
                        document_id=document.id,
                        failure=EvidenceExtractionFailure(
                            url=document.url,
                            reason=error.reason,
                            retryable=error.retryable,
                            message=str(error),
                        ),
                    )
                )
                continue
            for item in evidence:
                if item.document_id != document.id:
                    raise ValueError(
                        f"EvidenceItem {item.id} references document "
                        f"{item.document_id}, expected {document.id}"
                    )
            extraction_outcomes.append(
                ExtractionSuccess(
                    document_id=document.id,
                    evidence_ids=[self.store.put(item) for item in evidence],
                )
            )

        return ExtractEvidenceObservation(extraction_outcomes=extraction_outcomes)

    def _curate_evidence(self, action: AgentAction, state: AgentState) -> Observation:
        shelve_ids = action.params["shelve_document_ids"]
        reactivate_ids = action.params["reactivate_document_ids"]

        invalid_ids = [
            document_id
            for document_id in shelve_ids
            if (document := state.documents.get(document_id)) is None
            or document.lifecycle_status != "active"
        ]
        invalid_ids.extend(
            document_id
            for document_id in reactivate_ids
            if (document := state.documents.get(document_id)) is None
            or document.lifecycle_status != "shelved"
        )
        if invalid_ids:
            raise ValueError(
                "curate_evidence contains invalid lifecycle transitions: "
                + ", ".join(invalid_ids)
            )

        projected_active_count = (
            active_document_count(state) - len(shelve_ids) + len(reactivate_ids)
        )
        if projected_active_count > state.budget.max_active_documents:
            raise ValueError("curate_evidence would exceed the active document limit")

        return CurateEvidenceObservation()

    async def _synthesize(self, state: AgentState) -> Observation:
        if active_document_count(state) > state.budget.max_active_documents:
            raise ValueError("finish requires curation within the active document limit")

        documents: list[Document] = []
        evidence: list[EvidenceItem] = []
        for document_id, document_state in state.documents.items():
            if document_state.lifecycle_status != "active":
                continue
            document = self.store.get(document_id, Document)
            if document is not None:
                documents.append(document)
            for evidence_id in document_state.evidence_ids:
                item = self.store.get(evidence_id, EvidenceItem)
                if item is None:
                    continue
                evidence.append(item)

        result = await self.synthesizer.synthesize(
            SynthesisRequest(
                query=state.query,
                evidence=evidence,
                documents=documents,
            )
        )

        return FinishObservation(
            final_answer=result.answer,
            citations=result.citations,
        )

"""News action executor."""

import asyncio

from banso.artifacts import ArtifactStore
from banso.core.action import AgentAction, AgentActionType
from banso.core.observation import Observation
from banso.core.state import AgentState
from banso.documents import (
    Document,
    DocumentReadError,
    DocumentReadRequest,
    DocumentReader,
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
        document_reader: DocumentReader,
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
        self.document_reader = document_reader
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
            return Observation(
                data={"search_plan": plan.model_dump(mode="json")},
            )

        if action.type == AgentActionType.SEARCH:
            return await self._search(action, state)

        if action.type == AgentActionType.READ_DOCUMENT:
            return await self._read_document(state)

        if action.type == AgentActionType.EXTRACT_EVIDENCE:
            return await self._extract_evidence(state)

        if action.type == AgentActionType.FINISH:
            return await self._synthesize(state)

        return Observation()

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

        return Observation(
            data={
                "search_queries": [query],
                "search_result_ids": result_ids,
                "search_result_index_updates": index_updates,
                "search_result_merge_report": {
                    "candidate_count": len(results),
                    "new_result_count": new_result_count,
                    "reused_result_count": reused_result_count,
                },
                "retrieval_filter_report": filtered.report.model_dump(),
                "source_classification_report": classified.report(),
            },
        )

    async def _read_document(self, state: AgentState) -> Observation:
        document_ids: list[str] = []
        failures: list[dict[str, str | int | None]] = []

        result_ids = state.search_result_ids[: state.budget.max_documents_to_read]
        for result_id in result_ids:
            result = self.store.get(result_id, SearchResult)
            if result is None:
                continue
            try:
                document = await self.document_reader.read(
                    DocumentReadRequest(
                        url=result.url,
                        title=result.title,
                        source=result.source,
                        metadata={"search_result_id": result.id},
                    )
                )
            except DocumentReadError as error:
                failures.append(
                    {
                        "search_result_id": result.id,
                        "url": error.url,
                        "status_code": error.status_code,
                        "reason": error.reason,
                        "message": error.message,
                        "source_error_type": error.source_error_type,
                    }
                )
                continue
            document_ids.append(self.store.put(document))

        return Observation(
            data={
                "successfully_read_document_count": len(document_ids),
                "failed_document_count": len(failures),
                "document_ids": document_ids,
                "document_read_failures": failures,
            },
        )

    async def _extract_evidence(self, state: AgentState) -> Observation:
        documents = [
            document
            for document_id in state.document_ids
            if (document := self.store.get(document_id, Document)) is not None
        ]
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

        extraction_results = await asyncio.gather(
            *(extract(document) for document in documents)
        )
        evidence_ids = [
            self.store.put(item)
            for _, evidence_items, _ in extraction_results
            for item in evidence_items
        ]
        failures = [
            {
                "document_id": document.id,
                "url": document.url,
                "reason": error.reason,
                "message": str(error),
            }
            for document, _, error in extraction_results
            if error is not None
        ]
        documents_without_evidence = [
            document.id
            for document, evidence_items, error in extraction_results
            if error is None and not evidence_items
        ]

        successful_extractions = sum(
            error is None for _, _, error in extraction_results
        )
        return Observation(
            data={
                "successful_document_count": successful_extractions,
                "failed_document_count": len(failures),
                "evidence_count": len(evidence_ids),
                "evidence_ids": evidence_ids,
                "evidence_extraction_failures": failures,
                "documents_without_evidence": documents_without_evidence,
            },
        )

    async def _synthesize(self, state: AgentState) -> Observation:
        evidence = [
            item
            for evidence_id in state.evidence_ids
            if (item := self.store.get(evidence_id, EvidenceItem)) is not None
        ]
        documents = [
            document
            for document_id in state.document_ids
            if (document := self.store.get(document_id, Document)) is not None
        ]

        result = await self.synthesizer.synthesize(
            SynthesisRequest(
                query=state.query,
                evidence=evidence,
                documents=documents,
            )
        )

        return Observation(
            data={
                "final_answer": result.answer,
                "citations": result.citations,
            },
        )

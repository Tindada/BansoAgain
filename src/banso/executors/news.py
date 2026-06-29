"""News action executor."""

from banso.artifacts import ArtifactStore
from banso.core.action import AgentAction, AgentActionType
from banso.core.result import Observation
from banso.core.state import AgentState
from banso.documents import (
    Document,
    DocumentReadRequest,
    DocumentReader,
    EvidenceExtractionRequest,
    EvidenceExtractor,
    EvidenceItem,
)
from banso.retrieval import RetrievalProvider, SearchRequest, SearchResult
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
    ) -> None:
        self.store = store
        self.retrieval_provider = retrieval_provider
        self.document_reader = document_reader
        self.evidence_extractor = evidence_extractor
        self.synthesizer = synthesizer

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        """Execute a news-domain action."""
        if action.type == AgentActionType.SEARCH:
            return await self._search(action, state)

        if action.type == AgentActionType.READ_DOCUMENT:
            return await self._read_document(state)

        if action.type == AgentActionType.EXTRACT_EVIDENCE:
            return await self._extract_evidence(state)

        if action.type == AgentActionType.SYNTHESIZE:
            return await self._synthesize(state)

        return Observation(action_type=action.type)

    async def _search(self, action: AgentAction, state: AgentState) -> Observation:
        query = action.params.get("query", state.query.text)
        if not isinstance(query, str):
            query = state.query.text

        results = await self.retrieval_provider.search(
            SearchRequest(
                query=query,
                language=state.query.language,
                region=state.query.region,
                time_range=state.query.time_range,
            )
        )
        result_ids = [self.store.put(result) for result in results]

        return Observation(
            action_type=action.type,
            data={
                "search_queries": [query],
                "search_result_ids": result_ids,
            },
        )

    async def _read_document(self, state: AgentState) -> Observation:
        document_ids: list[str] = []

        for result_id in state.search_result_ids:
            result = self.store.get(result_id, SearchResult)
            if result is None:
                continue

            document = await self.document_reader.read(
                DocumentReadRequest(
                    url=result.url,
                    title=result.title,
                    source=result.source,
                    metadata={"search_result_id": result.id},
                )
            )
            document_ids.append(self.store.put(document))

        return Observation(
            action_type=AgentActionType.READ_DOCUMENT,
            data={"document_ids": document_ids},
        )

    async def _extract_evidence(self, state: AgentState) -> Observation:
        evidence_ids: list[str] = []

        for document_id in state.document_ids:
            document = self.store.get(document_id, Document)
            if document is None:
                continue

            evidence_items = await self.evidence_extractor.extract(
                EvidenceExtractionRequest(
                    query=state.query,
                    document=document,
                )
            )
            evidence_ids.extend(self.store.put(item) for item in evidence_items)

        return Observation(
            action_type=AgentActionType.EXTRACT_EVIDENCE,
            data={"evidence_ids": evidence_ids},
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
            action_type=AgentActionType.SYNTHESIZE,
            data={
                "final_answer": result.answer,
                "citations": result.citations,
            },
        )

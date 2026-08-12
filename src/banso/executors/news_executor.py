"""News action executor."""

from collections.abc import Mapping

from banso.artifacts import ArtifactStore
from banso.core.action import (
    AgentAction,
    AgentActionType,
    ResearchActionParams,
    RetrievalRoute,
)
from banso.core.observation import (
    CurateEvidenceObservation,
    FinishObservation,
    Observation,
    StopObservation,
)
from banso.core.state import AgentState
from banso.documents import Document, EvidenceExtractor, EvidenceItem
from banso.executors.research_pipeline import (
    ResearchPipeline,
    ResearchRouteComponents,
)
from banso.executors.retry import RetryPolicy
from banso.retrieval import SourceClassifier
from banso.retrieval.filter import RetrievalFilter
from banso.synthesis import Synthesizer, SynthesisRequest


class NewsActionExecutor:
    """Dispatch news actions and own the composite research pipeline."""

    def __init__(
        self,
        store: ArtifactStore,
        research_routes: Mapping[RetrievalRoute, ResearchRouteComponents],
        evidence_extractor: EvidenceExtractor,
        synthesizer: Synthesizer,
        retrieval_filter: RetrievalFilter | None = None,
        source_classifier: SourceClassifier | None = None,
        max_extraction_concurrency: int = 3,
        fetch_retry_policy: RetryPolicy | None = None,
        extraction_retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.store = store
        self.synthesizer = synthesizer
        self.research_pipeline = ResearchPipeline(
            store=store,
            research_routes=research_routes,
            evidence_extractor=evidence_extractor,
            retrieval_filter=retrieval_filter,
            source_classifier=source_classifier,
            max_extraction_concurrency=max_extraction_concurrency,
            fetch_retry_policy=fetch_retry_policy,
            extraction_retry_policy=extraction_retry_policy,
        )

    @property
    def research_routes(self) -> dict[RetrievalRoute, ResearchRouteComponents]:
        """Return the configured research routes."""
        return self.research_pipeline.research_routes

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        """Execute a news-domain action."""
        if action.type == AgentActionType.RESEARCH:
            params = ResearchActionParams.model_validate(action.params)
            return await self.research_pipeline.run(params, state)

        if action.type == AgentActionType.CURATE_EVIDENCE:
            return self._curate_evidence(action, state)

        if action.type == AgentActionType.FINISH:
            return await self._synthesize(state)

        if action.type == AgentActionType.STOP:
            return StopObservation()

        raise ValueError(f"unsupported action type: {action.type.value}")

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
            state.active_document_count - len(shelve_ids) + len(reactivate_ids)
        )
        if projected_active_count > state.budget.max_active_documents:
            raise ValueError("curate_evidence would exceed the active document limit")

        return CurateEvidenceObservation()

    async def _synthesize(self, state: AgentState) -> Observation:
        if state.active_document_count > state.budget.max_active_documents:
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
                if item is not None:
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

"""News action executor."""

from collections.abc import Mapping

from banso.artifacts.store import ArtifactStore
from banso.agent.action import (
    AgentAction,
    AgentActionType,
    ResearchActionParams,
    RetrievalRoute,
)
from banso.agent.executors.research_pipeline import (
    ResearchPipeline,
    ResearchRouteComponents,
)
from banso.agent.executors.retry import RetryPolicy
from banso.agent.observation import (
    FinishObservation,
    Observation,
    RewriteScratchObservation,
    StopObservation,
)
from banso.agent.research_context import (
    build_research_history,
    document_reference_maps,
)
from banso.agent.selection.selector import SearchResultSelector
from banso.agent.state import AgentState
from banso.documents.extractor import EvidenceExtractor
from banso.documents.models import Document, DocumentEvidence
from banso.retrieval.filter import RetrievalFilter
from banso.retrieval.source_classifier import SourceClassifier
from banso.scratch.rewriter import (
    ScratchEvidenceGroup,
    ScratchRewriter,
    ScratchRewriteRequest,
)
from banso.synthesis.synthesizer import (
    SynthesisEvidenceGroup,
    SynthesisRequest,
    Synthesizer,
)


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
        search_result_selector: SearchResultSelector | None = None,
        max_extraction_concurrency: int = 4,
        retry_policy: RetryPolicy | None = None,
        scratch_rewriter: ScratchRewriter | None = None,
    ) -> None:
        self.store = store
        self.synthesizer = synthesizer
        self.scratch_rewriter = scratch_rewriter
        self.research_pipeline = ResearchPipeline(
            store=store,
            research_routes=research_routes,
            evidence_extractor=evidence_extractor,
            retrieval_filter=retrieval_filter,
            source_classifier=source_classifier,
            search_result_selector=search_result_selector,
            max_extraction_concurrency=max_extraction_concurrency,
            retry_policy=retry_policy,
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

        if action.type == AgentActionType.REWRITE_SCRATCH:
            return await self._rewrite_scratch(state)

        if action.type == AgentActionType.FINISH:
            return await self._synthesize(state)

        if action.type == AgentActionType.STOP:
            return StopObservation()

        raise ValueError(f"unsupported action type: {action.type.value}")

    async def _rewrite_scratch(
        self,
        state: AgentState,
    ) -> Observation:
        id_to_ref, _ = document_reference_maps(state)
        evidence_groups: list[ScratchEvidenceGroup] = []
        for document_id, document_state in state.documents.items():
            evidence_id = document_state.evidence_id
            if evidence_id is None:
                continue
            document = self.store.get(document_id, Document)
            evidence = self.store.get(evidence_id, DocumentEvidence)
            if (
                document is None
                or evidence is None
                or evidence.document_id != document_id
            ):
                raise ValueError(
                    f"Document has missing or invalid evidence: {document_id}"
                )
            evidence_groups.append(
                ScratchEvidenceGroup(
                    document_ref=id_to_ref[document_id],
                    title=document.title,
                    source_url=document.url,
                    source=document.source,
                    published_at=document.published_at,
                    evidence_text=evidence.text,
                )
            )

        result = await self.scratch_rewriter.rewrite(
            ScratchRewriteRequest(
                query=state.query.text,
                language=state.query.language,
                time_range=state.query.time_range,
                reference_time=state.reference_time,
                current_scratch=state.scratch,
                research_history=[
                    item.model_dump(mode="json", exclude_none=True)
                    for item in build_research_history(state)
                ],
                evidence_groups=evidence_groups,
            )
        )
        return RewriteScratchObservation(content=result.content)

    async def _synthesize(self, state: AgentState) -> Observation:
        evidence_groups: list[SynthesisEvidenceGroup] = []
        for document_id, document_state in state.documents.items():
            if document_state.evidence_id is None:
                continue
            document = self.store.get(document_id, Document)
            if document is None:
                continue
            evidence_id = document_state.evidence_id
            evidence = (
                self.store.get(evidence_id, DocumentEvidence)
                if evidence_id is not None
                else None
            )
            if evidence is None or evidence.document_id != document_id:
                raise ValueError(
                    f"Document has missing or invalid evidence: {document_id}"
                )
            evidence_groups.append(
                SynthesisEvidenceGroup(
                    document_id=document.id,
                    title=document.title,
                    source_url=document.url,
                    source=document.source,
                    published_at=document.published_at,
                    evidence_text=evidence.text,
                )
            )

        result = await self.synthesizer.synthesize(
            SynthesisRequest(
                query=state.query.text,
                language=state.query.language,
                time_range=state.query.time_range,
                reference_time=state.reference_time,
                scratch=state.scratch,
                evidence_groups=evidence_groups,
                metadata=state.synthesis_metadata,
            )
        )
        return FinishObservation(
            final_answer=result.answer,
            citations=result.citations,
        )

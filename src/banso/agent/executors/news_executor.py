"""News action executor."""

from collections.abc import Mapping

from banso.artifacts.store import ArtifactStore
from banso.agent.action import (
    AgentAction,
    AgentActionType,
    ReadActionParams,
    ResearchActionParams,
    RetrievalRoute,
    SearchActionParams,
)
from banso.agent.executors.read import execute_read
from banso.agent.executors.research_pipeline import (
    ResearchRouteComponents,
    execute_research,
)
from banso.agent.executors.retry import RetryPolicy
from banso.agent.executors.search import execute_search
from banso.agent.observation import (
    FetchSuccess,
    FinishObservation,
    Observation,
    ReadObservation,
    RewriteNotesObservation,
    StopObservation,
)
from banso.agent.research_context import (
    build_research_history,
    document_reference_maps,
    search_result_reference_maps,
)
from banso.agent.selection.passthrough_selector import (
    PassthroughSearchResultSelector,
)
from banso.agent.selection.selector import SearchResultSelector
from banso.agent.state import AgentState
from banso.documents.extractor import EvidenceExtractor
from banso.documents.models import Document, DocumentEvidence
from banso.retrieval.filter import RetrievalFilter
from banso.retrieval.models import SearchResult
from banso.retrieval.provider import SearchRequest
from banso.retrieval.source_classifier import SourceClassifier
from banso.notes.rewriter import NotesEvidenceGroup, NotesRewriter, NotesRewriteRequest
from banso.synthesis.synthesizer import (
    SynthesisEvidenceGroup,
    SynthesisRequest,
    Synthesizer,
)


class NewsActionExecutor:
    """Dispatch news actions and own their execution dependencies."""

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
        notes_rewriter: NotesRewriter | None = None,
    ) -> None:
        if not research_routes:
            raise ValueError("research_routes must contain at least one route")
        if max_extraction_concurrency < 1:
            raise ValueError("max_extraction_concurrency must be at least 1")

        self.store = store
        self.synthesizer = synthesizer
        self.notes_rewriter = notes_rewriter
        self.research_routes = dict(research_routes)
        self.evidence_extractor = evidence_extractor
        self.retrieval_filter = retrieval_filter or RetrievalFilter()
        self.source_classifier = source_classifier or SourceClassifier()
        self.search_result_selector = (
            search_result_selector or PassthroughSearchResultSelector()
        )
        self.max_extraction_concurrency = max_extraction_concurrency
        self.retry_policy = retry_policy or RetryPolicy()

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        """Execute a news-domain action."""
        if action.type == AgentActionType.RESEARCH:
            params = ResearchActionParams.model_validate(action.params)
            return await execute_research(
                params,
                state,
                store=self.store,
                research_routes=self.research_routes,
                evidence_extractor=self.evidence_extractor,
                retrieval_filter=self.retrieval_filter,
                source_classifier=self.source_classifier,
                search_result_selector=self.search_result_selector,
                max_extraction_concurrency=self.max_extraction_concurrency,
                retry_policy=self.retry_policy,
            )

        if action.type == AgentActionType.SEARCH:
            params = SearchActionParams.model_validate(action.params)
            components = self._route_components(params.route)
            return await execute_search(
                SearchRequest(
                    query=params.query,
                    max_results=10,
                    language=state.query.language,
                    region=state.query.region,
                    time_range=state.query.time_range,
                    source_domains=params.source_domains,
                ),
                state,
                store=self.store,
                retrieval_provider=components.retrieval_provider,
                retrieval_filter=self.retrieval_filter,
                source_classifier=self.source_classifier,
                retry_policy=self.retry_policy,
                route=params.route,
            )

        if action.type == AgentActionType.READ:
            params = ReadActionParams.model_validate(action.params)
            groups = self._load_read_candidates(params.search_result_refs, state)
            document_index = dict(state.document_index)
            known_document_ids = set(state.documents)
            fetch_outcomes = []
            extraction_outcomes = []
            document_index_updates: dict[str, str] = {}
            for route, candidates in groups.items():
                components = self._route_components(route)
                batch = await execute_read(
                    candidates,
                    evidence_query=state.query.text,
                    document_index=document_index,
                    known_document_ids=known_document_ids,
                    store=self.store,
                    document_fetcher=components.document_fetcher,
                    evidence_extractor=self.evidence_extractor,
                    ignored_query_params=(
                        self.retrieval_filter.config.ignored_query_params
                    ),
                    limit=state.budget.max_results_per_research,
                    max_extraction_concurrency=self.max_extraction_concurrency,
                    retry_policy=self.retry_policy,
                    route=route,
                )
                fetch_outcomes.extend(batch.fetch_outcomes)
                extraction_outcomes.extend(batch.extraction_outcomes)
                document_index_updates.update(batch.document_index_updates)
                document_index.update(batch.document_index_updates)
                for outcome in batch.fetch_outcomes:
                    if isinstance(outcome, FetchSuccess):
                        known_document_ids.add(outcome.document_id)
            return ReadObservation(
                fetch_outcomes=fetch_outcomes,
                extraction_outcomes=extraction_outcomes,
                document_index_updates=document_index_updates,
            )

        if action.type == AgentActionType.REWRITE_NOTES:
            return await self._rewrite_notes(state)

        if action.type == AgentActionType.FINISH:
            return await self._synthesize(state)

        if action.type == AgentActionType.STOP:
            return StopObservation()

        raise ValueError(f"unsupported action type: {action.type.value}")

    def _route_components(self, route: RetrievalRoute) -> ResearchRouteComponents:
        components = self.research_routes.get(route)
        if components is None:
            raise ValueError(f"retrieval route is not enabled: {route.value}")
        return components

    def _load_read_candidates(
        self,
        result_refs: list[str],
        state: AgentState,
    ) -> dict[RetrievalRoute, list[SearchResult]]:
        if len(result_refs) > state.budget.max_results_per_research:
            raise ValueError("read contains too many search results")

        _, ref_to_id = search_result_reference_maps(state)
        groups: dict[RetrievalRoute, list[SearchResult]] = {}
        for result_ref in result_refs:
            result_id = ref_to_id.get(result_ref)
            if result_id is None:
                raise ValueError(f"read contains an unknown search result: {result_ref}")
            result_state = state.search_results[result_id]
            route = result_state.retrieval_route
            if result_state.document_id is not None or result_state.failure is not None:
                raise ValueError(f"search result has already been read: {result_ref}")
            result = self.store.get(result_id, SearchResult)
            if result is None:
                raise ValueError(f"invalid SearchResult artifact: {result_id}")
            groups.setdefault(route, []).append(result)
        return groups

    async def _rewrite_notes(
        self,
        state: AgentState,
    ) -> Observation:
        id_to_ref, _ = document_reference_maps(state)
        evidence_groups: list[NotesEvidenceGroup] = []
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
                NotesEvidenceGroup(
                    document_ref=id_to_ref[document_id],
                    title=document.title,
                    source_url=document.url,
                    source=document.source,
                    published_at=document.published_at,
                    evidence_text=evidence.text,
                )
            )

        result = await self.notes_rewriter.rewrite(
            NotesRewriteRequest(
                query=state.query.text,
                language=state.query.language,
                time_range=state.query.time_range,
                reference_time=state.reference_time,
                current_notes=state.notes,
                research_history=[
                    item.model_dump(mode="json", exclude_none=True)
                    for item in build_research_history(state)
                ],
                evidence_groups=evidence_groups,
            )
        )
        return RewriteNotesObservation(content=result.content)

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
                notes=state.notes,
                evidence_groups=evidence_groups,
                metadata=state.synthesis_metadata,
            )
        )
        return FinishObservation(
            final_answer=result.answer,
            citations=result.citations,
        )

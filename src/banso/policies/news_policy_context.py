"""Decision context exposed to the LLM-backed news policy."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from banso.artifacts import ArtifactStore
from banso.core.action import RetrievalRoute
from banso.core.observation import (
    ExtractionFailure,
    FetchFailure,
    ResearchObservation,
    ResearchObservationBase,
    RetrievalFailedResearchObservation,
)
from banso.core.state import AgentState, DocumentLifecycleStatus
from banso.documents import Document, EvidenceItem
from banso.retrieval import Source, SourceType
from banso.retrieval.url_utils import publisher_domain


class PolicySourceView(BaseModel):
    """Compact source identity visible to the policy."""

    name: str | None = None
    domain: str
    type: SourceType


class EvidenceGroup(BaseModel):
    """Representative evidence claims grouped by source document."""

    document_ref: str
    research_refs: list[str]
    lifecycle_status: DocumentLifecycleStatus
    lifecycle_reason: str | None = None
    lifecycle_updated_at_step: int | None = None
    document_title: str
    source: PolicySourceView
    published_at: datetime | None = None
    evidence_count: int
    claim_previews: list[str]


class ResearchHistoryItemBase(BaseModel):
    """Facts shared by successful and failed research actions."""

    research_ref: str
    query: str
    route: RetrievalRoute
    source_domains: list[str] | None = None


class CompletedResearchHistoryItem(ResearchHistoryItemBase):
    """Compact outcomes from a completed research action."""

    status: Literal["completed"] = "completed"
    retrieved_results: int
    new_results: int
    reused_results: int
    selected_results: int
    deferred_results: int
    fetch_successes: int
    fetch_failures: int
    extraction_successes: int
    extraction_failures: int


class RetrievalFailedResearchHistoryItem(ResearchHistoryItemBase):
    """Trace-safe diagnostics from a failed retrieval."""

    status: Literal["retrieval_failed"] = "retrieval_failed"
    reason: str
    status_code: int | None = None
    retryable: bool
    attempt_count: int


ResearchHistoryItem = Annotated[
    CompletedResearchHistoryItem | RetrievalFailedResearchHistoryItem,
    Field(discriminator="status"),
]


class BudgetSummary(BaseModel):
    """Remaining execution capacity relevant to the next action."""

    remaining_steps: int
    remaining_researches: int
    max_results_per_research: int
    max_active_documents: int
    active_document_overflow: int


class ArtifactSummary(BaseModel):
    """Collected artifact totals independent of context display limits."""

    search_result_count: int
    document_count: int
    active_document_count: int
    shelved_document_count: int
    unusable_document_count: int
    evidence_count: int
    active_evidence_count: int
    shelved_evidence_count: int
    distinct_evidence_source_count: int


class WorkingSetSummary(BaseModel):
    """Document references grouped by their current curation status."""

    active_document_refs: list[str]
    shelved_document_refs: list[str]


class PolicyUserQuery(BaseModel):
    """User query fields relevant to policy decisions."""

    text: str
    region: str | None = None
    time_range: str | None = None


class NewsPolicyContext(BaseModel):
    """Compact facts visible to the LLM news policy."""

    user_query: PolicyUserQuery
    reference_time: datetime
    enabled_routes: list[RetrievalRoute]
    budget: BudgetSummary
    research_history: list[ResearchHistoryItem]
    artifacts: ArtifactSummary
    working_set: WorkingSetSummary
    evidence_groups: list[EvidenceGroup]


def document_reference_maps(state: AgentState) -> tuple[dict[str, str], dict[str, str]]:
    """Return stable rollout-local mappings between document IDs and LLM refs."""
    id_to_ref = {
        document_id: f"D{index}"
        for index, document_id in enumerate(state.documents, start=1)
    }
    return id_to_ref, {
        document_ref: document_id
        for document_id, document_ref in id_to_ref.items()
    }


class NewsPolicyContextBuilder:
    """Build deterministic LLM decision context from state and artifacts."""

    def __init__(
        self,
        store: ArtifactStore,
        enabled_routes: list[RetrievalRoute],
        *,
        max_evidence_per_document: int = 10,
        max_claim_chars: int = 300,
    ) -> None:
        if not enabled_routes:
            raise ValueError("enabled_routes must contain at least one route")
        if len(set(enabled_routes)) != len(enabled_routes):
            raise ValueError("enabled_routes must be unique")
        if max_evidence_per_document < 0:
            raise ValueError("max_evidence_per_document must be non-negative")
        if max_claim_chars < 0:
            raise ValueError("max_claim_chars must be non-negative")

        self.store = store
        self.enabled_routes = list(enabled_routes)
        self.max_evidence_per_document = max_evidence_per_document
        self.max_claim_chars = max_claim_chars

    def build(self, state: AgentState) -> NewsPolicyContext:
        """Resolve state facts and artifacts into bounded decision context."""
        documents = {
            document_id: self._load(document_id, Document)
            for document_id in state.documents
        }
        evidence_by_document: dict[str, list[EvidenceItem]] = {}
        for document_id, document_state in state.documents.items():
            evidence_by_document[document_id] = [
                self._load(evidence_id, EvidenceItem)
                for evidence_id in document_state.evidence_ids
            ]

        id_to_ref, _ = document_reference_maps(state)
        active_count = state.active_document_count
        evidence_count = sum(map(len, evidence_by_document.values()))
        active_evidence_count = sum(
            len(evidence_by_document[document_id])
            for document_id, document in state.documents.items()
            if document.lifecycle_status == "active"
        )
        shelved_evidence_count = sum(
            len(evidence_by_document[document_id])
            for document_id, document in state.documents.items()
            if document.lifecycle_status == "shelved"
        )
        evidence_domains = {
            domain
            for items in evidence_by_document.values()
            for item in items
            if (domain := publisher_domain(item.source_url))
        }
        research_entries = [
            entry
            for entry in state.action_history
            if isinstance(entry.observation, ResearchObservationBase)
        ]
        referenced_research = [
            (f"R{index}", entry.observation)
            for index, entry in enumerate(research_entries, start=1)
        ]
        document_research_refs: dict[str, list[str]] = {}
        for research_ref, observation in referenced_research:
            if isinstance(observation, RetrievalFailedResearchObservation):
                continue
            document_ids = {
                state.search_results[result_id].document_id
                for result_id in observation.search_result_ids
            }
            for document_id in document_ids:
                if document_id is not None:
                    document_research_refs.setdefault(document_id, []).append(research_ref)

        return NewsPolicyContext(
            user_query=PolicyUserQuery(
                text=state.query.text,
                region=state.query.region,
                time_range=state.query.time_range,
            ),
            reference_time=state.reference_time,
            enabled_routes=list(self.enabled_routes),
            budget=BudgetSummary(
                remaining_steps=state.remaining_steps,
                remaining_researches=state.remaining_research_capacity,
                max_results_per_research=state.budget.max_results_per_research,
                max_active_documents=state.budget.max_active_documents,
                active_document_overflow=max(
                    active_count - state.budget.max_active_documents,
                    0,
                ),
            ),
            research_history=[
                self._build_research_history(research_ref, observation)
                for research_ref, observation in referenced_research
            ],
            artifacts=ArtifactSummary(
                search_result_count=len(state.search_results),
                document_count=len(state.documents),
                active_document_count=active_count,
                shelved_document_count=sum(
                    document.lifecycle_status == "shelved"
                    for document in state.documents.values()
                ),
                unusable_document_count=sum(
                    document.lifecycle_status == "unusable"
                    for document in state.documents.values()
                ),
                evidence_count=evidence_count,
                active_evidence_count=active_evidence_count,
                shelved_evidence_count=shelved_evidence_count,
                distinct_evidence_source_count=len(evidence_domains),
            ),
            working_set=WorkingSetSummary(
                active_document_refs=[
                    id_to_ref[document_id]
                    for document_id, document in state.documents.items()
                    if document.lifecycle_status == "active"
                ],
                shelved_document_refs=[
                    id_to_ref[document_id]
                    for document_id, document in state.documents.items()
                    if document.lifecycle_status == "shelved"
                ],
            ),
            evidence_groups=[
                self._build_evidence_group(
                    id_to_ref[document_id],
                    document_research_refs.get(document_id, []),
                    documents[document_id],
                    document_state.lifecycle_status,
                    document_state.lifecycle_reason,
                    document_state.lifecycle_updated_at_step,
                    evidence_by_document[document_id],
                )
                for document_id, document_state in state.documents.items()
                if document_state.lifecycle_status is not None
            ],
        )

    def _build_research_history(
        self,
        research_ref: str,
        observation: ResearchObservation,
    ) -> ResearchHistoryItem:
        if isinstance(observation, RetrievalFailedResearchObservation):
            return RetrievalFailedResearchHistoryItem(
                research_ref=research_ref,
                query=observation.query,
                route=observation.route,
                source_domains=observation.source_domains,
                reason=observation.reason,
                status_code=observation.status_code,
                retryable=observation.retryable,
                attempt_count=observation.attempt_count,
            )

        fetch_failures = [
            outcome
            for outcome in observation.fetch_outcomes
            if isinstance(outcome, FetchFailure)
        ]
        extraction_failures = [
            outcome
            for outcome in observation.extraction_outcomes
            if isinstance(outcome, ExtractionFailure)
        ]
        return CompletedResearchHistoryItem(
            research_ref=research_ref,
            query=observation.query,
            route=observation.route,
            source_domains=observation.source_domains,
            retrieved_results=len(observation.search_result_ids),
            new_results=observation.search_result_merge_report.new_result_count,
            reused_results=observation.search_result_merge_report.reused_result_count,
            selected_results=len(observation.selection_report.selected_ids),
            deferred_results=len(observation.selection_report.deferred_ids),
            fetch_successes=len(observation.fetch_outcomes) - len(fetch_failures),
            fetch_failures=len(fetch_failures),
            extraction_successes=(
                len(observation.extraction_outcomes) - len(extraction_failures)
            ),
            extraction_failures=len(extraction_failures),
        )

    def _build_evidence_group(
        self,
        document_ref: str,
        research_refs: list[str],
        document: Document,
        lifecycle_status: DocumentLifecycleStatus,
        lifecycle_reason: str | None,
        lifecycle_updated_at_step: int | None,
        evidence: list[EvidenceItem],
    ) -> EvidenceGroup:
        visible_evidence = evidence[: self.max_evidence_per_document]
        return EvidenceGroup(
            document_ref=document_ref,
            research_refs=research_refs,
            lifecycle_status=lifecycle_status,
            lifecycle_reason=lifecycle_reason,
            lifecycle_updated_at_step=lifecycle_updated_at_step,
            document_title=document.title,
            source=self._build_source(document.source, document.url),
            published_at=document.published_at,
            evidence_count=len(evidence),
            claim_previews=[
                item.claim[: self.max_claim_chars]
                for item in visible_evidence
            ],
        )

    @staticmethod
    def _build_source(source: Source | None, item_url: str) -> PolicySourceView:
        domain = publisher_domain(item_url)
        return PolicySourceView(
            name=source.name if source is not None else domain or None,
            domain=domain,
            type=source.type if source is not None else SourceType.UNKNOWN,
        )

    def _load(self, artifact_id: str, artifact_type: type[Document] | type[EvidenceItem]):
        artifact = self.store.get(artifact_id, artifact_type)
        if artifact is None:
            raise ValueError(
                f"{artifact_type.__name__} artifact is missing or has the wrong type: "
                f"{artifact_id}"
            )
        return artifact

"""Decision context exposed to the LLM-backed news policy."""

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from banso.artifacts import ArtifactStore
from banso.core.action import AgentActionType
from banso.core.lifecycle import (
    LifecycleStatus,
    active_document_count,
    eligible_extraction_document_ids,
    eligible_fetch_result_ids,
    progress_status,
    remaining_document_fetches,
)
from banso.core.state import (
    ActionHistoryEntry,
    AgentState,
    DocumentLifecycleStatus,
    ExtractProgress,
    SearchResultState,
    UserQuery,
)
from banso.documents import Document, EvidenceItem
from banso.retrieval import SearchResult, Source, SourceType
from banso.retrieval.url_utils import publisher_domain

TArtifact = TypeVar("TArtifact", SearchResult, Document, EvidenceItem)


class PolicySourceView(BaseModel):
    """Compact source identity visible to the policy."""

    name: str | None = None
    domain: str
    type: SourceType


class SearchResultCandidate(BaseModel):
    """One search result actionable by the next fetch operation."""

    title: str
    snippet: str | None = None
    source: PolicySourceView
    published_at: datetime | None = None
    fetch_status: LifecycleStatus


class DocumentCandidate(BaseModel):
    """One document actionable by the next extraction operation."""

    document_ref: str
    title: str
    text_preview: str
    source: PolicySourceView
    published_at: datetime | None = None
    extraction_status: LifecycleStatus


class EvidenceGroup(BaseModel):
    """Representative evidence claims grouped by source document."""

    document_ref: str
    lifecycle_status: DocumentLifecycleStatus
    lifecycle_reason: str | None = None
    lifecycle_updated_at_step: int | None = None
    document_title: str
    source: PolicySourceView
    published_at: datetime | None = None
    evidence_count: int
    claim_previews: list[str]


class SearchHistoryItem(BaseModel):
    """One completed search relevant to future decisions."""

    query: str
    intent: str | None = None
    new_results: int
    reused_results: int


class BudgetSummary(BaseModel):
    """Remaining execution capacity relevant to the next action."""

    remaining_steps: int
    remaining_searches: int
    remaining_document_fetches: int
    max_active_documents: int
    active_document_overflow: int


class ResourceWorkSummary(BaseModel):
    """Current lifecycle and actionable work for one resource stage."""

    pending: int
    retryable: int
    failed: int
    actionable: int
    failure_reasons: dict[str, int]


class WorkSummary(BaseModel):
    """Actionable fetch and extraction work."""

    fetch: ResourceWorkSummary
    extraction: ResourceWorkSummary
    extracted_without_evidence: int


class ArtifactSummary(BaseModel):
    """Full collected-artifact totals independent of visible limits."""

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


class NewsPolicyContext(BaseModel):
    """Compact, action-oriented facts visible to the LLM news policy."""

    user_query: UserQuery
    reference_time: datetime
    budget: BudgetSummary
    search_history: list[SearchHistoryItem]
    work: WorkSummary
    artifacts: ArtifactSummary
    working_set: WorkingSetSummary
    candidate_results: list[SearchResultCandidate]
    candidate_documents: list[DocumentCandidate]
    evidence_groups: list[EvidenceGroup]


def document_reference_maps(state: AgentState) -> tuple[dict[str, str], dict[str, str]]:
    """Return stable rollout-local mappings between document IDs and LLM refs."""
    id_to_ref = {
        document_id: f"D{index}"
        for index, document_id in enumerate(state.documents, start=1)
    }
    return id_to_ref, {document_ref: document_id for document_id, document_ref in id_to_ref.items()}


class NewsPolicyContextBuilder:
    """Build deterministic LLM decision context from state and artifacts."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        max_search_results: int = 30,
        max_documents: int = 8,
        max_evidence_per_document: int = 10,
        max_snippet_chars: int = 300,
        max_document_preview_chars: int = 750,
        max_claim_chars: int = 300,
    ) -> None:
        limits = {
            "max_search_results": max_search_results,
            "max_documents": max_documents,
            "max_evidence_per_document": max_evidence_per_document,
            "max_snippet_chars": max_snippet_chars,
            "max_document_preview_chars": max_document_preview_chars,
            "max_claim_chars": max_claim_chars,
        }
        for name, value in limits.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        self.store = store
        self.max_search_results = max_search_results
        self.max_documents = max_documents
        self.max_evidence_per_document = max_evidence_per_document
        self.max_snippet_chars = max_snippet_chars
        self.max_document_preview_chars = max_document_preview_chars
        self.max_claim_chars = max_claim_chars

    def build(self, state: AgentState) -> NewsPolicyContext:
        """Resolve state facts and artifacts into bounded decision context."""
        search_results = self._load_all(list(state.search_results), SearchResult)
        documents = self._load_all(list(state.documents), Document)
        evidence_items_by_document_id: dict[str, list[EvidenceItem]] = {}
        for document_id, document_state in state.documents.items():
            evidence_items = self._load_all(document_state.evidence_ids, EvidenceItem)
            evidence_items_by_document_id[document_id] = evidence_items
        search_result_by_id = {search_result.id: search_result for search_result in search_results}
        document_by_id = {document.id: document for document in documents}
        search_history = [
            self._build_search(entry)
            for entry in state.action_history
            if entry.action_type == AgentActionType.SEARCH
        ]
        fetch_statuses = {
            search_result_id: progress_status(
                result,
                state.budget.max_fetch_attempts,
            )
            for search_result_id, result in state.search_results.items()
        }
        extraction_statuses = {
            document_id: progress_status(
                document.extraction,
                state.budget.max_extraction_attempts,
            )
            for document_id, document in state.documents.items()
        }
        fetch_counts = Counter(fetch_statuses.values())
        extraction_counts = Counter(extraction_statuses.values())
        evidence_count_by_document_id = {
            document_id: len(evidence_items)
            for document_id, evidence_items in evidence_items_by_document_id.items()
        }
        id_to_document_ref, _ = document_reference_maps(state)
        remaining_fetches = remaining_document_fetches(state)
        actionable_fetch_ids = eligible_fetch_result_ids(state)[:remaining_fetches]
        actionable_extraction_ids = eligible_extraction_document_ids(state)
        candidate_search_results = [
            search_result_by_id[search_result_id]
            for search_result_id in actionable_fetch_ids
        ]
        candidate_documents = [
            document_by_id[document_id]
            for document_id in actionable_extraction_ids
        ]
        evidence_domains = {
            domain
            for evidence_items in evidence_items_by_document_id.values()
            for item in evidence_items
            if (domain := publisher_domain(item.source_url))
        }
        active_count = active_document_count(state)
        active_evidence_count = sum(
            evidence_count_by_document_id[document_id]
            for document_id, document in state.documents.items()
            if document.lifecycle_status == "active"
        )
        shelved_evidence_count = sum(
            evidence_count_by_document_id[document_id]
            for document_id, document in state.documents.items()
            if document.lifecycle_status == "shelved"
        )
        evidence_count = sum(evidence_count_by_document_id.values())

        return NewsPolicyContext(
            user_query=state.query.model_copy(deep=True),
            reference_time=state.reference_time,
            budget=BudgetSummary(
                remaining_steps=max(state.budget.max_steps - state.current_step, 0),
                remaining_searches=max(state.budget.max_searches - len(search_history), 0),
                remaining_document_fetches=remaining_fetches,
                max_active_documents=state.budget.max_active_documents,
                active_document_overflow=max(
                    active_count - state.budget.max_active_documents,
                    0,
                ),
            ),
            search_history=search_history,
            work=WorkSummary(
                fetch=ResourceWorkSummary(
                    pending=fetch_counts["pending"],
                    retryable=fetch_counts["retryable"],
                    failed=fetch_counts["failed"],
                    actionable=len(actionable_fetch_ids),
                    failure_reasons=self._failure_reasons(
                        result
                        for result in state.search_results.values()
                        if result.attempt_count > 0
                    ),
                ),
                extraction=ResourceWorkSummary(
                    pending=extraction_counts["pending"],
                    retryable=extraction_counts["retryable"],
                    failed=extraction_counts["failed"],
                    actionable=len(actionable_extraction_ids),
                    failure_reasons=self._failure_reasons(
                        document.extraction
                        for document in state.documents.values()
                        if document.extraction is not None
                    ),
                ),
                extracted_without_evidence=sum(
                    status == "succeeded"
                    and evidence_count_by_document_id[document_id] == 0
                    for document_id, status in extraction_statuses.items()
                ),
            ),
            artifacts=ArtifactSummary(
                search_result_count=len(search_results),
                document_count=len(documents),
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
                    id_to_document_ref[document_id]
                    for document_id, document in state.documents.items()
                    if document.lifecycle_status == "active"
                ],
                shelved_document_refs=[
                    id_to_document_ref[document_id]
                    for document_id, document in state.documents.items()
                    if document.lifecycle_status == "shelved"
                ],
            ),
            candidate_results=self._build_search_results(candidate_search_results, fetch_statuses),
            candidate_documents=self._build_documents(
                candidate_documents,
                extraction_statuses,
                id_to_document_ref,
            ),
            evidence_groups=self._build_evidence_groups(
                state,
                evidence_items_by_document_id,
                document_by_id,
                id_to_document_ref,
            ),
        )

    def _build_search_results(
        self,
        search_results: list[SearchResult],
        statuses: dict[str, LifecycleStatus],
    ) -> list[SearchResultCandidate]:
        return [
            SearchResultCandidate(
                title=search_result.title,
                snippet=(
                    search_result.snippet[: self.max_snippet_chars]
                    if search_result.snippet is not None
                    else None
                ),
                source=self._build_source(search_result.source, search_result.url),
                published_at=search_result.published_at,
                fetch_status=statuses[search_result.id],
            )
            for search_result in search_results[: self.max_search_results]
        ]

    def _build_documents(
        self,
        documents: list[Document],
        statuses: dict[str, LifecycleStatus],
        id_to_document_ref: dict[str, str],
    ) -> list[DocumentCandidate]:
        return [
            DocumentCandidate(
                document_ref=id_to_document_ref[document.id],
                title=document.title,
                text_preview=document.text[: self.max_document_preview_chars],
                source=self._build_source(document.source, document.url),
                published_at=document.published_at,
                extraction_status=statuses[document.id],
            )
            for document in documents[: self.max_documents]
        ]

    def _build_evidence_groups(
        self,
        state: AgentState,
        evidence_items_by_document_id: dict[str, list[EvidenceItem]],
        document_by_id: dict[str, Document],
        id_to_document_ref: dict[str, str],
    ) -> list[EvidenceGroup]:
        groups: list[EvidenceGroup] = []
        for document_id, document_state in state.documents.items():
            if document_state.lifecycle_status is None:
                continue
            evidence_items = evidence_items_by_document_id[document_id]
            visible_evidence = evidence_items[: self.max_evidence_per_document]
            document = document_by_id[document_id]
            groups.append(
                EvidenceGroup(
                    document_ref=id_to_document_ref[document_id],
                    lifecycle_status=document_state.lifecycle_status,
                    lifecycle_reason=document_state.lifecycle_reason,
                    lifecycle_updated_at_step=document_state.lifecycle_updated_at_step,
                    document_title=document.title,
                    source=self._build_source(document.source, document.url),
                    published_at=document.published_at,
                    evidence_count=len(evidence_items),
                    claim_previews=[item.claim[: self.max_claim_chars] for item in visible_evidence],
                )
            )
        return groups

    @staticmethod
    def _build_source(source: Source | None, item_url: str) -> PolicySourceView:
        domain = publisher_domain(item_url)
        return PolicySourceView(
            name=source.name if source is not None else domain or None,
            domain=domain,
            type=source.type if source is not None else SourceType.UNKNOWN,
        )

    @staticmethod
    def _failure_reasons(progress_values: Iterable[SearchResultState | ExtractProgress]) -> dict[str, int]:
        reasons = Counter(progress.failure.reason for progress in progress_values if progress.failure is not None)
        return dict(sorted(reasons.items()))

    @staticmethod
    def _build_search(entry: ActionHistoryEntry) -> SearchHistoryItem:
        merge_report = entry.observation.data.get("search_result_merge_report", {})
        return SearchHistoryItem(
            query=entry.params["query"],
            intent=entry.params.get("intent"),
            new_results=merge_report.get("new_result_count", 0),
            reused_results=merge_report.get("reused_result_count", 0),
        )

    def _load_all(self, artifact_ids: list[str], artifact_type: type[TArtifact]) -> list[TArtifact]:
        artifacts: list[TArtifact] = []
        for artifact_id in artifact_ids:
            artifact = self.store.get(artifact_id, artifact_type)
            if artifact is None:
                raise ValueError(
                    f"{artifact_type.__name__} artifact is missing or has the wrong type: "
                    f"{artifact_id}"
                )
            artifacts.append(artifact)
        return artifacts

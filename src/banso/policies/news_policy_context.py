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
    eligible_extraction_document_ids,
    eligible_read_result_ids,
    progress_status,
    remaining_document_count,
)
from banso.core.state import (
    ActionHistoryEntry,
    AgentState,
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
    """One search result actionable by the next read operation."""

    title: str
    snippet: str | None = None
    source: PolicySourceView
    published_at: datetime | None = None
    read_status: LifecycleStatus


class DocumentCandidate(BaseModel):
    """One document actionable by the next extraction operation."""

    title: str
    text_preview: str
    source: PolicySourceView
    published_at: datetime | None = None
    extraction_status: LifecycleStatus


class EvidenceGroup(BaseModel):
    """Representative evidence claims grouped by source document."""

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
    remaining_document_slots: int


class ResourceWorkSummary(BaseModel):
    """Current lifecycle and actionable work for one resource stage."""

    pending: int
    retryable: int
    failed: int
    actionable: int
    failure_reasons: dict[str, int]


class WorkSummary(BaseModel):
    """Actionable read and extraction work."""

    read: ResourceWorkSummary
    extraction: ResourceWorkSummary
    extracted_without_evidence: int


class ArtifactSummary(BaseModel):
    """Full collected-artifact totals independent of visible limits."""

    search_results: int
    documents: int
    evidence: int
    distinct_evidence_sources: int


class NewsPolicyContext(BaseModel):
    """Compact, action-oriented facts visible to the LLM news policy."""

    user_query: UserQuery
    reference_time: datetime
    budget: BudgetSummary
    search_history: list[SearchHistoryItem]
    work: WorkSummary
    artifacts: ArtifactSummary
    candidate_results: list[SearchResultCandidate]
    candidate_documents: list[DocumentCandidate]
    evidence_groups: list[EvidenceGroup]


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
        evidence_by_document_id: dict[str, list[EvidenceItem]] = {}
        for document_id, document_state in state.documents.items():
            document_evidence = self._load_all(document_state.evidence_ids, EvidenceItem)
            evidence_by_document_id[document_id] = document_evidence
        search_result_by_id = {search_result.id: search_result for search_result in search_results}
        document_by_id = {document.id: document for document in documents}
        search_history = [
            self._build_search(entry)
            for entry in state.action_history
            if entry.action_type == AgentActionType.SEARCH
        ]
        read_statuses = {
            search_result_id: progress_status(
                result,
                state.budget.max_read_attempts,
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
        read_counts = Counter(read_statuses.values())
        extraction_counts = Counter(extraction_statuses.values())
        evidence_counts = {
            document_id: len(document_evidence)
            for document_id, document_evidence in evidence_by_document_id.items()
        }
        remaining_document_slots = remaining_document_count(state)
        actionable_read_ids = eligible_read_result_ids(state)[:remaining_document_slots]
        actionable_extraction_ids = eligible_extraction_document_ids(state)
        candidate_search_results = [
            search_result_by_id[search_result_id]
            for search_result_id in actionable_read_ids
        ]
        candidate_documents = [
            document_by_id[document_id]
            for document_id in actionable_extraction_ids
        ]
        evidence_domains = {
            domain
            for document_evidence in evidence_by_document_id.values()
            for item in document_evidence
            if (domain := publisher_domain(item.source_url))
        }

        return NewsPolicyContext(
            user_query=state.query.model_copy(deep=True),
            reference_time=state.reference_time,
            budget=BudgetSummary(
                remaining_steps=max(state.budget.max_steps - state.current_step, 0),
                remaining_searches=max(state.budget.max_searches - len(search_history), 0),
                remaining_document_slots=remaining_document_slots,
            ),
            search_history=search_history,
            work=WorkSummary(
                read=ResourceWorkSummary(
                    pending=read_counts["pending"],
                    retryable=read_counts["retryable"],
                    failed=read_counts["failed"],
                    actionable=len(actionable_read_ids),
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
                    status == "succeeded" and evidence_counts[document_id] == 0
                    for document_id, status in extraction_statuses.items()
                ),
            ),
            artifacts=ArtifactSummary(
                search_results=len(search_results),
                documents=len(documents),
                evidence=sum(evidence_counts.values()),
                distinct_evidence_sources=len(evidence_domains),
            ),
            candidate_results=self._build_search_results(candidate_search_results, read_statuses),
            candidate_documents=self._build_documents(candidate_documents, extraction_statuses),
            evidence_groups=self._build_evidence_groups(
                evidence_by_document_id,
                document_by_id,
            ),
        )

    def _build_search_results(self, search_results: list[SearchResult], statuses: dict[str, LifecycleStatus]) -> list[SearchResultCandidate]:
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
                read_status=statuses[search_result.id],
            )
            for search_result in search_results[: self.max_search_results]
        ]

    def _build_documents(self, documents: list[Document], statuses: dict[str, LifecycleStatus]) -> list[DocumentCandidate]:
        return [
            DocumentCandidate(
                title=document.title,
                text_preview=document.text[
                    : self.max_document_preview_chars
                ],
                source=self._build_source(document.source, document.url),
                published_at=document.published_at,
                extraction_status=statuses[document.id],
            )
            for document in documents[: self.max_documents]
        ]

    def _build_evidence_groups(
        self,
        evidence_by_document_id: dict[str, list[EvidenceItem]],
        document_by_id: dict[str, Document],
    ) -> list[EvidenceGroup]:
        groups: list[EvidenceGroup] = []
        for document_id, document_evidence in evidence_by_document_id.items():
            visible_evidence = document_evidence[: self.max_evidence_per_document]
            if not visible_evidence:
                continue
            document = document_by_id[document_id]
            groups.append(
                EvidenceGroup(
                    document_title=document.title,
                    source=self._build_source(document.source, document.url),
                    published_at=document.published_at,
                    evidence_count=len(document_evidence),
                    claim_previews=[
                        item.claim[: self.max_claim_chars]
                        for item in visible_evidence
                    ],
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
        reasons = Counter(
            progress.failure.reason
            for progress in progress_values
            if progress.failure is not None
        )
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

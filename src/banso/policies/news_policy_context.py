"""Decision context exposed to the LLM-backed news policy."""

from collections import Counter
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from banso.artifacts import ArtifactStore
from banso.core.action import AgentActionType
from banso.core.lifecycle import (
    LifecycleStatus,
    extraction_status,
    read_status,
    remaining_document_count,
)
from banso.core.state import ActionHistoryEntry, AgentState, UserQuery
from banso.documents import Document, EvidenceItem
from banso.retrieval import SearchResult, Source

TArtifact = TypeVar("TArtifact", SearchResult, Document, EvidenceItem)
_STATUS_PRIORITY: dict[LifecycleStatus, int] = {
    "pending": 0,
    "retryable": 1,
    "succeeded": 2,
    "failed": 3,
}


class SearchResultView(BaseModel):
    """Policy-visible fields from a search result."""

    id: str
    title: str
    url: str
    snippet: str | None = None
    source: Source | None = None
    published_at: datetime | None = None
    read_status: LifecycleStatus
    document_id: str | None = None
    read_failure_reason: str | None = None


class DocumentView(BaseModel):
    """Policy-visible fields from a fetched document."""

    id: str
    title: str
    url: str
    text_preview: str
    source: Source | None = None
    published_at: datetime | None = None
    author: str | None = None
    extraction_status: LifecycleStatus
    evidence_count: int
    extraction_failure_reason: str | None = None


class EvidenceView(BaseModel):
    """Policy-visible fields from an extracted evidence item."""

    id: str
    document_id: str
    claim_preview: str
    source_url: str
    published_at: datetime | None = None
    confidence: float | None = None


class SearchAttempt(BaseModel):
    """One completed search relevant to future decisions."""

    step_index: int
    query: str
    intent: str | None = None
    result_count: int
    new_result_count: int
    reused_result_count: int


class NewsPolicyContext(BaseModel):
    """Bounded decision facts visible to the LLM-backed news policy."""

    user_query: UserQuery
    reference_time: datetime
    current_step: int
    max_steps: int
    remaining_step_count: int
    executed_search_count: int
    max_searches: int
    remaining_search_count: int
    max_documents_to_read: int
    remaining_document_count: int
    searches: list[SearchAttempt]
    search_result_count: int
    omitted_search_result_count: int
    pending_read_count: int
    retryable_read_count: int
    failed_read_count: int
    search_results: list[SearchResultView]
    document_count: int
    omitted_document_count: int
    pending_extraction_count: int
    retryable_extraction_count: int
    failed_extraction_count: int
    documents_without_evidence_count: int
    documents: list[DocumentView]
    evidence_count: int
    omitted_evidence_count: int
    evidence: list[EvidenceView]


class NewsPolicyContextBuilder:
    """Build deterministic LLM decision context from state and artifacts."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        max_search_results: int = 30,
        max_documents: int = 8,
        max_evidence: int = 50,
        max_snippet_chars: int = 500,
        max_document_preview_chars: int = 1000,
        max_claim_chars: int = 1000,
    ) -> None:
        limits = {
            "max_search_results": max_search_results,
            "max_documents": max_documents,
            "max_evidence": max_evidence,
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
        self.max_evidence = max_evidence
        self.max_snippet_chars = max_snippet_chars
        self.max_document_preview_chars = max_document_preview_chars
        self.max_claim_chars = max_claim_chars

    def build(self, state: AgentState) -> NewsPolicyContext:
        """Resolve state facts and artifacts into bounded decision context."""
        search_results = self._load_all(state.search_result_ids, SearchResult)
        documents = self._load_all(state.document_ids, Document)
        evidence = self._load_all(state.evidence_ids, EvidenceItem)
        searches = [
            self._build_search(entry)
            for entry in state.action_history
            if entry.action_type == AgentActionType.SEARCH
        ]
        read_statuses = {
            result_id: read_status(state, result_id)
            for result_id in state.search_result_ids
        }
        extraction_statuses = {
            document_id: extraction_status(state, document_id)
            for document_id in state.document_ids
        }
        read_counts = Counter(read_statuses.values())
        extraction_counts = Counter(extraction_statuses.values())
        evidence_counts = Counter(item.document_id for item in evidence)
        executed_search_count = len(searches)

        return NewsPolicyContext(
            user_query=state.query.model_copy(deep=True),
            reference_time=state.reference_time,
            current_step=state.current_step,
            max_steps=state.budget.max_steps,
            remaining_step_count=max(
                state.budget.max_steps - state.current_step,
                0,
            ),
            executed_search_count=executed_search_count,
            max_searches=state.budget.max_searches,
            remaining_search_count=max(
                state.budget.max_searches - executed_search_count,
                0,
            ),
            max_documents_to_read=state.budget.max_documents_to_read,
            remaining_document_count=remaining_document_count(state),
            searches=searches,
            search_result_count=len(search_results),
            omitted_search_result_count=max(len(search_results) - self.max_search_results, 0),
            pending_read_count=read_counts["pending"],
            retryable_read_count=read_counts["retryable"],
            failed_read_count=read_counts["failed"],
            search_results=self._build_search_results(
                search_results,
                state,
                read_statuses,
            ),
            document_count=len(documents),
            omitted_document_count=max(len(documents) - self.max_documents, 0),
            pending_extraction_count=extraction_counts["pending"],
            retryable_extraction_count=extraction_counts["retryable"],
            failed_extraction_count=extraction_counts["failed"],
            documents_without_evidence_count=sum(
                status == "succeeded" and evidence_counts[document_id] == 0
                for document_id, status in extraction_statuses.items()
            ),
            documents=self._build_documents(
                documents,
                state,
                extraction_statuses,
                evidence_counts,
            ),
            evidence_count=len(evidence),
            omitted_evidence_count=max(len(evidence) - self.max_evidence, 0),
            evidence=self._build_evidence(evidence),
        )

    def _build_search_results(
        self,
        results: list[SearchResult],
        state: AgentState,
        statuses: dict[str, LifecycleStatus],
    ) -> list[SearchResultView]:
        visible_results = sorted(
            results,
            key=lambda result: _STATUS_PRIORITY[statuses[result.id]],
        )[: self.max_search_results]
        return [
            SearchResultView(
                id=result.id,
                title=result.title,
                url=result.url,
                snippet=(
                    result.snippet[: self.max_snippet_chars]
                    if result.snippet is not None
                    else None
                ),
                source=result.source,
                published_at=result.published_at,
                read_status=statuses[result.id],
                document_id=(
                    progress.document_id
                    if (progress := state.read_progress.get(result.id)) is not None
                    else None
                ),
                read_failure_reason=(
                    progress.failure.reason
                    if progress is not None and progress.failure is not None
                    else None
                ),
            )
            for result in visible_results
        ]

    def _build_documents(
        self,
        documents: list[Document],
        state: AgentState,
        statuses: dict[str, LifecycleStatus],
        evidence_counts: Counter[str],
    ) -> list[DocumentView]:
        visible_documents = sorted(
            documents,
            key=lambda document: _STATUS_PRIORITY[statuses[document.id]],
        )[: self.max_documents]
        return [
            DocumentView(
                id=document.id,
                title=document.title,
                url=document.url,
                text_preview=document.text[: self.max_document_preview_chars],
                source=document.source,
                published_at=document.published_at,
                author=document.author,
                extraction_status=statuses[document.id],
                evidence_count=evidence_counts[document.id],
                extraction_failure_reason=(
                    progress.failure.reason
                    if (
                        progress := state.extract_progress.get(document.id)
                    ) is not None
                    and progress.failure is not None
                    else None
                ),
            )
            for document in visible_documents
        ]

    def _build_evidence(
        self,
        evidence: list[EvidenceItem],
    ) -> list[EvidenceView]:
        return [
            EvidenceView(
                id=item.id,
                document_id=item.document_id,
                claim_preview=item.claim[: self.max_claim_chars],
                source_url=item.source_url,
                published_at=item.published_at,
                confidence=item.confidence,
            )
            for item in evidence[: self.max_evidence]
        ]

    @staticmethod
    def _build_search(
        entry: ActionHistoryEntry,
    ) -> SearchAttempt:
        data = entry.observation.data
        merge_report = data.get("search_result_merge_report", {})
        return SearchAttempt(
            step_index=entry.step_index,
            query=entry.params["query"],
            intent=entry.params.get("intent"),
            result_count=len(data.get("search_result_ids", [])),
            new_result_count=merge_report.get("new_result_count", 0),
            reused_result_count=merge_report.get("reused_result_count", 0),
        )

    def _load_all(
        self,
        artifact_ids: list[str],
        artifact_type: type[TArtifact],
    ) -> list[TArtifact]:
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

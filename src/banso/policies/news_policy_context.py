"""Decision context exposed to the LLM-backed news policy."""

from datetime import datetime
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, Field

from banso.artifacts import ArtifactStore
from banso.core.action import AgentActionType
from banso.core.state import ActionHistoryEntry, AgentState, UserQuery
from banso.documents import Document, EvidenceItem
from banso.retrieval import SearchResult, Source

TArtifact = TypeVar("TArtifact", SearchResult, Document, EvidenceItem)


class SearchResultView(BaseModel):
    """Policy-visible fields from a search result."""

    id: str
    title: str
    url: str
    snippet: str | None = None
    source: Source | None = None
    published_at: datetime | None = None


class DocumentView(BaseModel):
    """Policy-visible fields from a fetched document."""

    id: str
    title: str
    url: str
    text_preview: str
    source: Source | None = None
    published_at: datetime | None = None
    author: str | None = None


class EvidenceView(BaseModel):
    """Policy-visible fields from an extracted evidence item."""

    id: str
    document_id: str
    claim_preview: str
    source_url: str
    published_at: datetime | None = None
    confidence: float | None = None


class ResourceFailure(BaseModel):
    """A recoverable failure while processing one resource."""

    resource_id: str
    url: str
    reason: str
    status_code: int | None = None


class SearchAttempt(BaseModel):
    """One completed search and its result association."""

    type: Literal[AgentActionType.SEARCH] = AgentActionType.SEARCH
    step_index: int
    query: str
    intent: str | None = None
    result_ids: list[str]
    new_result_count: int
    reused_result_count: int


class ReadAttempt(BaseModel):
    """One completed document reading attempt."""

    type: Literal[AgentActionType.READ_DOCUMENT] = AgentActionType.READ_DOCUMENT
    step_index: int
    document_ids: list[str]
    failures: list[ResourceFailure]


class ExtractAttempt(BaseModel):
    """One completed evidence extraction attempt."""

    type: Literal[AgentActionType.EXTRACT_EVIDENCE] = AgentActionType.EXTRACT_EVIDENCE
    step_index: int
    evidence_ids: list[str]
    successful_document_count: int
    documents_without_evidence: list[str]
    failures: list[ResourceFailure]


Attempt = Annotated[
    SearchAttempt | ReadAttempt | ExtractAttempt,
    Field(discriminator="type"),
]


class NewsPolicyContext(BaseModel):
    """Bounded decision facts visible to the LLM-backed news policy."""

    user_query: UserQuery
    current_step: int
    max_steps: int
    remaining_step_count: int
    executed_search_count: int
    max_searches: int
    remaining_search_count: int
    max_documents_to_read: int
    attempts: list[Attempt]
    search_result_count: int
    omitted_search_result_count: int
    search_results: list[SearchResultView]
    document_count: int
    omitted_document_count: int
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
        executed_search_count = sum(
            entry.action_type == AgentActionType.SEARCH
            for entry in state.action_history
        )

        return NewsPolicyContext(
            user_query=state.query.model_copy(deep=True),
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
            attempts=[
                attempt
                for entry in state.action_history
                if (attempt := self._build_attempt(entry)) is not None
            ],
            search_result_count=len(search_results),
            omitted_search_result_count=max(len(search_results) - self.max_search_results, 0),
            search_results=self._build_search_results(search_results),
            document_count=len(documents),
            omitted_document_count=max(len(documents) - self.max_documents, 0),
            documents=self._build_documents(documents),
            evidence_count=len(evidence),
            omitted_evidence_count=max(len(evidence) - self.max_evidence, 0),
            evidence=self._build_evidence(evidence),
        )

    def _build_search_results(
        self,
        results: list[SearchResult],
    ) -> list[SearchResultView]:
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
            )
            for result in results[: self.max_search_results]
        ]

    def _build_documents(
        self,
        documents: list[Document],
    ) -> list[DocumentView]:
        return [
            DocumentView(
                id=document.id,
                title=document.title,
                url=document.url,
                text_preview=document.text[: self.max_document_preview_chars],
                source=document.source,
                published_at=document.published_at,
                author=document.author,
            )
            for document in documents[: self.max_documents]
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
    def _build_attempt(
        entry: ActionHistoryEntry,
    ) -> Attempt | None:
        data = entry.observation.data
        if entry.action_type == AgentActionType.SEARCH:
            merge_report = data.get("search_result_merge_report", {})
            return SearchAttempt(
                step_index=entry.step_index,
                query=entry.params["query"],
                intent=entry.params.get("intent"),
                result_ids=data.get("search_result_ids", []),
                new_result_count=merge_report.get("new_result_count", 0),
                reused_result_count=merge_report.get("reused_result_count", 0),
            )

        if entry.action_type == AgentActionType.READ_DOCUMENT:
            failures = data.get("document_read_failures", [])
            return ReadAttempt(
                step_index=entry.step_index,
                document_ids=data.get("document_ids", []),
                failures=[
                    ResourceFailure(
                        resource_id=failure["search_result_id"],
                        url=failure["url"],
                        reason=failure["reason"],
                        status_code=failure.get("status_code"),
                    )
                    for failure in failures
                ],
            )

        if entry.action_type == AgentActionType.EXTRACT_EVIDENCE:
            failures = data.get("evidence_extraction_failures", [])
            return ExtractAttempt(
                step_index=entry.step_index,
                evidence_ids=data.get("evidence_ids", []),
                successful_document_count=data.get("successful_document_count", 0),
                documents_without_evidence=data.get("documents_without_evidence", []),
                failures=[
                    ResourceFailure(
                        resource_id=failure["document_id"],
                        url=failure["url"],
                        reason=failure["reason"],
                    )
                    for failure in failures
                ],
            )

        return None

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

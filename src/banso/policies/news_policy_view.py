"""Bounded policy-visible views for the news agent."""

from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from banso.artifacts import ArtifactStore
from banso.core.state import AgentState
from banso.documents import Document, EvidenceItem
from banso.retrieval import SearchResult, Source

TArtifact = TypeVar("TArtifact", SearchResult, Document, EvidenceItem)


class SearchResultPolicyView(BaseModel):
    """Policy-visible fields from a search result."""

    id: str
    title: str
    url: str
    snippet: str | None = None
    source: Source | None = None
    published_at: datetime | None = None


class DocumentPolicyView(BaseModel):
    """Policy-visible fields from a fetched document."""

    id: str
    title: str
    url: str
    text_preview: str
    source: Source | None = None
    published_at: datetime | None = None
    author: str | None = None


class EvidencePolicyView(BaseModel):
    """Policy-visible fields from an extracted evidence item."""

    id: str
    document_id: str
    claim: str
    source_url: str
    published_at: datetime | None = None
    confidence: float | None = None


class NewsPolicyStateView(BaseModel):
    """A bounded snapshot of state and artifacts visible to a news policy."""

    state: AgentState
    search_results: list[SearchResultPolicyView]
    documents: list[DocumentPolicyView]
    evidence: list[EvidencePolicyView]


class NewsPolicyStateViewBuilder:
    """Build a deterministic policy view from state and stored artifacts."""

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

    def build(self, state: AgentState) -> NewsPolicyStateView:
        """Resolve all state artifact ids and return their bounded policy view."""
        search_results = self._load_all(state.search_result_ids, SearchResult, "search result")
        documents = self._load_all(state.document_ids, Document, "document")
        evidence = self._load_all(state.evidence_ids, EvidenceItem, "evidence")

        return NewsPolicyStateView(
            state=state.model_copy(deep=True),
            search_results=[
                SearchResultPolicyView(
                    id=result.id,
                    title=result.title,
                    url=result.url,
                    snippet=self._truncate(result.snippet, self.max_snippet_chars),
                    source=result.source,
                    published_at=result.published_at,
                )
                for result in search_results[: self.max_search_results]
            ],
            documents=[
                DocumentPolicyView(
                    id=document.id,
                    title=document.title,
                    url=document.url,
                    text_preview=document.text[: self.max_document_preview_chars],
                    source=document.source,
                    published_at=document.published_at,
                    author=document.author,
                )
                for document in documents[: self.max_documents]
            ],
            evidence=[
                EvidencePolicyView(
                    id=item.id,
                    document_id=item.document_id,
                    claim=item.claim[: self.max_claim_chars],
                    source_url=item.source_url,
                    published_at=item.published_at,
                    confidence=item.confidence,
                )
                for item in evidence[: self.max_evidence]
            ],
        )

    def _load_all(
        self,
        artifact_ids: list[str],
        artifact_type: type[TArtifact],
        artifact_label: str,
    ) -> list[TArtifact]:
        artifacts: list[TArtifact] = []
        for artifact_id in artifact_ids:
            artifact = self.store.get(artifact_id, artifact_type)
            if artifact is None:
                raise ValueError(
                    f"{artifact_label} artifact is missing or has the wrong type: "
                    f"{artifact_id}"
                )
            artifacts.append(artifact)
        return artifacts

    @staticmethod
    def _truncate(value: str | None, max_chars: int) -> str | None:
        if value is None:
            return None
        return value[:max_chars]

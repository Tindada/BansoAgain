"""Concurrency and extraction-failure tests for atomic research."""

import asyncio

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentAction, AgentActionType, AgentState, RetrievalRoute, UserQuery
from banso.core.observation import ExtractionFailure
from banso.documents import (
    Document,
    DocumentFetchRequest,
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceItem,
)
from banso.executors import NewsActionExecutor, ResearchRouteComponents
from banso.retrieval import SearchRequest, SearchResult
from banso.synthesis import SynthesisRequest, SynthesisResult


class Provider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        return [
            SearchResult(
                id=f"result-{index}",
                title=f"Result {index}",
                url=f"https://example.com/{index}",
            )
            for index in range(4)
        ]


class Fetcher:
    async def fetch(self, request: DocumentFetchRequest) -> Document:
        suffix = request.url.rsplit("/", 1)[-1]
        return Document(
            id=f"document-{suffix}",
            title=request.title or suffix,
            url=request.url,
            text="body",
        )


class Synthesizer:
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        return SynthesisResult(answer="answer")


class TrackingExtractor:
    def __init__(self, fail_document: str | None = None) -> None:
        self.active = 0
        self.max_active = 0
        self.fail_document = fail_document
        self.call_counts: dict[str, int] = {}

    async def extract(self, request: EvidenceExtractionRequest) -> list[EvidenceItem]:
        self.call_counts[request.document.id] = (
            self.call_counts.get(request.document.id, 0) + 1
        )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if request.document.id == self.fail_document:
            raise EvidenceExtractionError("failed", reason="llm_error")
        return [
            EvidenceItem(
                document_id=request.document.id,
                claim=request.document.id,
                source_url=request.document.url,
            )
        ]


def _action() -> AgentAction:
    return AgentAction(
        type=AgentActionType.RESEARCH,
        params={"query": "query", "route": "web"},
    )


def test_extraction_respects_concurrency_and_preserves_order() -> None:
    extractor = TrackingExtractor()
    executor = NewsActionExecutor(
        store=InMemoryArtifactStore(),
        research_routes={
            RetrievalRoute.WEB: ResearchRouteComponents(Provider(), Fetcher())
        },
        evidence_extractor=extractor,
        synthesizer=Synthesizer(),
        max_extraction_concurrency=2,
    )

    observation = asyncio.run(
        executor.execute(_action(), AgentState(query=UserQuery(text="query")))
    )

    assert extractor.max_active == 2
    assert [outcome.document_id for outcome in observation.extraction_outcomes] == [
        f"document-{index}" for index in range(4)
    ]


def test_known_extraction_failure_is_isolated() -> None:
    extractor = TrackingExtractor("document-1")
    executor = NewsActionExecutor(
        store=InMemoryArtifactStore(),
        research_routes={
            RetrievalRoute.WEB: ResearchRouteComponents(Provider(), Fetcher())
        },
        evidence_extractor=extractor,
        synthesizer=Synthesizer(),
        max_extraction_concurrency=2,
    )

    observation = asyncio.run(
        executor.execute(_action(), AgentState(query=UserQuery(text="query")))
    )

    failures = [
        outcome
        for outcome in observation.extraction_outcomes
        if isinstance(outcome, ExtractionFailure)
    ]
    assert len(failures) == 1
    assert failures[0].document_id == "document-1"
    assert failures[0].attempt_count == 2
    assert extractor.call_counts["document-1"] == 2


def test_extraction_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        NewsActionExecutor(
            store=InMemoryArtifactStore(),
            research_routes={
                RetrievalRoute.WEB: ResearchRouteComponents(Provider(), Fetcher())
            },
            evidence_extractor=TrackingExtractor(),
            synthesizer=Synthesizer(),
            max_extraction_concurrency=0,
        )

"""Tests for synthesis over collected evidence."""

import asyncio
from datetime import datetime, timezone

import pytest

from banso.artifacts.store import InMemoryArtifactStore
from banso.agent.action import AgentAction, AgentActionType, RetrievalRoute
from banso.agent.executors.news_executor import NewsActionExecutor
from banso.agent.executors.research_pipeline import ResearchRouteComponents
from banso.agent.state import AgentState, DocumentState, UserQuery
from banso.documents.models import Document, DocumentEvidence
from banso.retrieval.fake import FakeRetrievalProvider
from banso.scratch.rewriter import ScratchRewriteRequest, ScratchRewriteResult
from banso.source import Source, SourceType
from banso.synthesis.synthesizer import (
    Citation,
    SynthesisRequest,
    SynthesisResult,
)


class UnusedFetcher:
    async def fetch(self, request):
        raise AssertionError("fetch should not run")


class UnusedExtractor:
    async def extract(self, request):
        raise AssertionError("extraction should not run")


class RecordingSynthesizer:
    def __init__(self) -> None:
        self.request: SynthesisRequest | None = None

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.request = request
        return SynthesisResult(
            answer="answer",
            citations=[
                Citation(
                    reference="S1",
                    document_id="document-0",
                    source_url="https://example.com/0",
                )
            ],
        )


class StaticScratchRewriter:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[ScratchRewriteRequest] = []

    async def rewrite(self, request: ScratchRewriteRequest) -> ScratchRewriteResult:
        self.requests.append(request)
        return ScratchRewriteResult(content=self.content)


def _executor(
    store: InMemoryArtifactStore,
    synthesizer: RecordingSynthesizer,
    scratch_rewriter: StaticScratchRewriter | None = None,
) -> NewsActionExecutor:
    return NewsActionExecutor(
        store=store,
        research_routes={
            RetrievalRoute.WEB: ResearchRouteComponents(
                FakeRetrievalProvider(),
                UnusedFetcher(),
            )
        },
        evidence_extractor=UnusedExtractor(),
        synthesizer=synthesizer,
        scratch_rewriter=scratch_rewriter or StaticScratchRewriter("unused"),
    )


def _state_and_store(count: int) -> tuple[AgentState, InMemoryArtifactStore]:
    store = InMemoryArtifactStore()
    documents: dict[str, DocumentState] = {}
    for index in range(count):
        document = Document(
            id=f"document-{index}",
            url=f"https://example.com/{index}",
            title=f"Document {index}",
            text="body",
            source=Source(name="Example", type=SourceType.NEWS),
            published_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        evidence = DocumentEvidence(
            id=f"evidence-{index}",
            document_id=document.id,
            text=f"evidence {index}",
        )
        store.put(document)
        store.put(evidence)
        documents[document.id] = DocumentState(evidence_id=evidence.id)
    return (
        AgentState(
            query=UserQuery(text="query", language="en", time_range="week"),
            reference_time=datetime(2026, 8, 2, tzinfo=timezone.utc),
            scratch="Coverage ledger",
            documents=documents,
        ),
        store,
    )


def test_finish_synthesizes_all_collected_evidence() -> None:
    state, store = _state_and_store(5)
    synthesizer = RecordingSynthesizer()

    observation = asyncio.run(
        _executor(store, synthesizer).execute(
            AgentAction(type=AgentActionType.FINISH),
            state,
        )
    )

    assert observation.final_answer == "answer"
    assert observation.citations[0].document_id == "document-0"
    assert synthesizer.request is not None
    assert synthesizer.request.query == "query"
    assert synthesizer.request.language == "en"
    assert synthesizer.request.time_range == "week"
    assert synthesizer.request.reference_time == datetime(
        2026, 8, 2, tzinfo=timezone.utc
    )
    assert synthesizer.request.scratch == "Coverage ledger"
    assert [group.document_id for group in synthesizer.request.evidence_groups] == [
        "document-0",
        "document-1",
        "document-2",
        "document-3",
        "document-4",
    ]
    first = synthesizer.request.evidence_groups[0]
    assert first.title == "Document 0"
    assert first.source_url == "https://example.com/0"
    assert first.source == Source(name="Example", type=SourceType.NEWS)
    assert first.published_at == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert first.evidence_text == "evidence 0"


def test_finish_rejects_a_selected_document_with_missing_evidence() -> None:
    state, store = _state_and_store(1)
    state.documents["document-0"].evidence_id = "missing"

    with pytest.raises(ValueError, match="missing or invalid evidence"):
        asyncio.run(
            _executor(store, RecordingSynthesizer()).execute(
                AgentAction(type=AgentActionType.FINISH),
                state,
            )
        )


def test_rewrite_scratch_builds_request_and_allows_clearing() -> None:
    state, store = _state_and_store(1)
    clearing_rewriter = StaticScratchRewriter("")
    executor = _executor(
        store,
        RecordingSynthesizer(),
        clearing_rewriter,
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(type=AgentActionType.REWRITE_SCRATCH),
            state,
        )
    )

    assert observation.type == AgentActionType.REWRITE_SCRATCH
    assert observation.content == ""
    request = clearing_rewriter.requests[0]
    assert request.current_scratch == "Coverage ledger"
    assert request.evidence_groups[0].document_ref == "D1"
    assert request.evidence_groups[0].evidence_text == "evidence 0"

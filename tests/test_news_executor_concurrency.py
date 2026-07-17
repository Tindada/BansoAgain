"""Tests for controlled concurrency in the news executor."""

import asyncio

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentState, UserQuery
from banso.core.action import AgentAction, AgentActionType
from banso.documents import (
    Document,
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceItem,
)
from banso.documents.fake import FakeDocumentReader
from banso.executors import NewsActionExecutor
from banso.retrieval import FakeRetrievalProvider
from banso.synthesis import FakeSynthesizer


class TrackingEvidenceExtractor:
    def __init__(self) -> None:
        self.active_count = 0
        self.max_active_count = 0

    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        self.active_count += 1
        self.max_active_count = max(self.max_active_count, self.active_count)

        delay = 0.03 if request.document.title == "First" else 0.01
        await asyncio.sleep(delay)

        self.active_count -= 1
        return [
            EvidenceItem(
                document_id=request.document.id,
                claim=request.document.title,
                source_url=request.document.url,
            )
        ]


class PartiallyFailingEvidenceExtractor:
    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        if request.document.title == "Invalid":
            raise EvidenceExtractionError("invalid response", reason="invalid_json")
        if request.document.title == "Empty":
            return []
        return [
            EvidenceItem(
                document_id=request.document.id,
                claim=request.document.title,
                source_url=request.document.url,
            )
        ]


class FailingEvidenceExtractor:
    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        raise EvidenceExtractionError("provider failed", reason="llm_error")


async def _run_extraction_respects_concurrency_and_document_order() -> None:
    store = InMemoryArtifactStore()
    documents = [
        Document(title="First", url="https://example.com/first", text="First"),
        Document(title="Second", url="https://example.com/second", text="Second"),
        Document(title="Third", url="https://example.com/third", text="Third"),
    ]
    state = AgentState(
        query=UserQuery(text="test query"),
        document_ids=[store.put(document) for document in documents],
    )
    extractor = TrackingEvidenceExtractor()
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_reader=FakeDocumentReader(),
        evidence_extractor=extractor,
        synthesizer=FakeSynthesizer(),
        max_extraction_concurrency=2,
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        state,
    )

    evidence = [
        store.get(evidence_id, EvidenceItem)
        for evidence_id in observation.data["evidence_ids"]
    ]
    assert extractor.max_active_count == 2
    assert [item.claim for item in evidence if item is not None] == [
        "First",
        "Second",
        "Third",
    ]
    assert observation.data["successful_document_count"] == 3
    assert observation.data["failed_document_count"] == 0
    assert observation.data["evidence_count"] == 3


def test_extraction_respects_concurrency_and_document_order() -> None:
    asyncio.run(_run_extraction_respects_concurrency_and_document_order())


def test_extraction_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        NewsActionExecutor(
            store=InMemoryArtifactStore(),
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=TrackingEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
            max_extraction_concurrency=0,
        )


async def _run_extraction_isolates_known_failures() -> None:
    store = InMemoryArtifactStore()
    documents = [
        Document(title="Valid", url="https://example.com/valid", text="Valid"),
        Document(title="Invalid", url="https://example.com/invalid", text="Invalid"),
        Document(title="Empty", url="https://example.com/empty", text="Empty"),
    ]
    state = AgentState(
        query=UserQuery(text="test query"),
        document_ids=[store.put(document) for document in documents],
    )
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_reader=FakeDocumentReader(),
        evidence_extractor=PartiallyFailingEvidenceExtractor(),
        synthesizer=FakeSynthesizer(),
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        state,
    )

    assert len(observation.data["evidence_ids"]) == 1
    assert observation.data["evidence_extraction_failures"] == [
        {
            "document_id": documents[1].id,
            "url": documents[1].url,
            "reason": "invalid_json",
            "message": "invalid response",
        }
    ]
    assert observation.data["documents_without_evidence"] == [documents[2].id]
    assert observation.data["successful_document_count"] == 2
    assert observation.data["failed_document_count"] == 1
    assert observation.data["evidence_count"] == 1


def test_extraction_isolates_known_failures() -> None:
    asyncio.run(_run_extraction_isolates_known_failures())


async def _run_extraction_reports_failed_when_all_documents_fail() -> None:
    store = InMemoryArtifactStore()
    documents = [
        Document(title="First", url="https://example.com/first", text="First"),
        Document(title="Second", url="https://example.com/second", text="Second"),
    ]
    state = AgentState(
        query=UserQuery(text="test query"),
        document_ids=[store.put(document) for document in documents],
    )
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_reader=FakeDocumentReader(),
        evidence_extractor=FailingEvidenceExtractor(),
        synthesizer=FakeSynthesizer(),
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        state,
    )

    assert observation.data["successful_document_count"] == 0
    assert observation.data["failed_document_count"] == 2
    assert observation.data["evidence_count"] == 0
    assert observation.data["evidence_ids"] == []


def test_extraction_reports_failed_when_all_documents_fail() -> None:
    asyncio.run(_run_extraction_reports_failed_when_all_documents_fail())

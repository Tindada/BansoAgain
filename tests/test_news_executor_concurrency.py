"""Tests for controlled concurrency in the news executor."""

import asyncio

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentState,
    DefaultStateReducer,
    UserQuery,
)
from banso.core.observation import ExtractionSuccess
from banso.core.state import DocumentState
from banso.core.action import AgentAction, AgentActionType
from banso.documents import (
    Document,
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceItem,
)
from banso.documents.fake import FakeDocumentFetcher
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


class MisassignedEvidenceExtractor:
    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        return [
            EvidenceItem(
                document_id="wrong-document",
                claim="wrong source",
                source_url=request.document.url,
            )
        ]


async def _run_extraction_respects_concurrency_and_document_order() -> None:
    store = InMemoryArtifactStore()
    documents = [
        Document(title="First", url="https://example.com/first", text="First"),
        Document(title="Second", url="https://example.com/second", text="Second"),
        Document(title="Third", url="https://example.com/third", text="Third"),
    ]
    state = AgentState(
        query=UserQuery(text="test query"),
        documents={
            store.put(document): DocumentState()
            for document in documents
        },
    )
    extractor = TrackingEvidenceExtractor()
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_fetcher=FakeDocumentFetcher(),
        evidence_extractor=extractor,
        synthesizer=FakeSynthesizer(),
        max_extraction_concurrency=2,
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        state,
    )

    evidence_ids = [
        evidence_id
        for outcome in observation.extraction_outcomes
        if isinstance(outcome, ExtractionSuccess)
        for evidence_id in outcome.evidence_ids
    ]
    evidence = [store.get(evidence_id, EvidenceItem) for evidence_id in evidence_ids]
    assert extractor.max_active_count == 2
    assert [item.claim for item in evidence if item is not None] == [
        "First",
        "Second",
        "Third",
    ]
    assert [
        outcome.document_id
        for outcome in observation.extraction_outcomes
    ] == [document.id for document in documents]


def test_extraction_respects_concurrency_and_document_order() -> None:
    asyncio.run(_run_extraction_respects_concurrency_and_document_order())


def test_extraction_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        NewsActionExecutor(
            store=InMemoryArtifactStore(),
            retrieval_provider=FakeRetrievalProvider(),
            document_fetcher=FakeDocumentFetcher(),
            evidence_extractor=TrackingEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
            max_extraction_concurrency=0,
        )


async def _run_extraction_rejects_misassigned_evidence() -> None:
    store = InMemoryArtifactStore()
    document = Document(
        title="Document",
        url="https://example.com/document",
        text="Document",
    )
    state = AgentState(
        query=UserQuery(text="test query"),
        documents={store.put(document): DocumentState()},
    )
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_fetcher=FakeDocumentFetcher(),
        evidence_extractor=MisassignedEvidenceExtractor(),
        synthesizer=FakeSynthesizer(),
    )

    with pytest.raises(ValueError, match="wrong-document"):
        await executor.execute(
            AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
            state,
        )


def test_extraction_rejects_misassigned_evidence() -> None:
    asyncio.run(_run_extraction_rejects_misassigned_evidence())


async def _run_extraction_isolates_known_failures() -> None:
    store = InMemoryArtifactStore()
    documents = [
        Document(title="Valid", url="https://example.com/valid", text="Valid"),
        Document(title="Invalid", url="https://example.com/invalid", text="Invalid"),
        Document(title="Empty", url="https://example.com/empty", text="Empty"),
    ]
    state = AgentState(
        query=UserQuery(text="test query"),
        documents={
            store.put(document): DocumentState()
            for document in documents
        },
    )
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_fetcher=FakeDocumentFetcher(),
        evidence_extractor=PartiallyFailingEvidenceExtractor(),
        synthesizer=FakeSynthesizer(),
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        state,
    )

    outcomes = observation.extraction_outcomes
    first = outcomes[0]
    assert isinstance(first, ExtractionSuccess)
    evidence_ids = first.evidence_ids
    assert len(evidence_ids) == 1
    assert [outcome.model_dump(mode="json") for outcome in outcomes] == [
        {
            "status": "success",
            "document_id": documents[0].id,
            "evidence_ids": evidence_ids,
        },
        {
            "status": "failure",
            "document_id": documents[1].id,
            "failure": {
                "url": documents[1].url,
                "reason": "invalid_json",
                "retryable": False,
                "message": "invalid response",
            },
        },
        {
            "status": "success",
            "document_id": documents[2].id,
            "evidence_ids": [],
        },
    ]

    state = DefaultStateReducer().apply(
        state,
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        observation,
    )
    assert state.documents[documents[0].id].evidence_ids == evidence_ids
    assert state.documents[documents[0].id].extraction.failure is None
    assert state.documents[documents[1].id].extraction.failure is not None
    assert state.documents[documents[2].id].extraction.failure is None

    next_observation = await executor.execute(
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        state,
    )
    assert next_observation.extraction_outcomes == []


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
        documents={
            store.put(document): DocumentState()
            for document in documents
        },
    )
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_fetcher=FakeDocumentFetcher(),
        evidence_extractor=FailingEvidenceExtractor(),
        synthesizer=FakeSynthesizer(),
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        state,
    )

    assert [
        outcome.model_dump(mode="json")
        for outcome in observation.extraction_outcomes
    ] == [
        {
            "status": "failure",
            "document_id": document.id,
            "failure": {
                "url": document.url,
                "reason": "llm_error",
                "retryable": True,
                "message": "provider failed",
            },
        }
        for document in documents
    ]

    action = AgentAction(type=AgentActionType.EXTRACT_EVIDENCE)
    reducer = DefaultStateReducer()
    state = reducer.apply(state, action, observation)
    second = await executor.execute(action, state)
    state = reducer.apply(state, action, second)
    third = await executor.execute(action, state)

    assert all(
        state.documents[document.id].extraction.attempt_count == 2
        for document in documents
    )
    assert third.extraction_outcomes == []


def test_extraction_reports_failed_when_all_documents_fail() -> None:
    asyncio.run(_run_extraction_reports_failed_when_all_documents_fail())

"""Tests for document-level evidence curation."""

import asyncio
from typing import Literal

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentAction,
    AgentActionType,
    AgentRuntime,
    AgentState,
    DefaultStateReducer,
    ExecutionBudget,
    UserQuery,
)
from banso.core.observation import FinishObservation
from banso.core.state import DocumentState, ExtractProgress
from banso.documents import (
    Document,
    EvidenceItem,
    FakeDocumentFetcher,
    FakeEvidenceExtractor,
)
from banso.executors import NewsActionExecutor
from banso.retrieval import FakeRetrievalProvider, SearchRequest, SearchResult
from banso.synthesis import (
    FakeSynthesizer,
    SynthesisRequest,
    SynthesisResult,
)


def _executor(
    store: InMemoryArtifactStore,
    *,
    retrieval_provider=None,
    synthesizer=None,
) -> NewsActionExecutor:
    return NewsActionExecutor(
        store=store,
        retrieval_provider=retrieval_provider or FakeRetrievalProvider(),
        document_fetcher=FakeDocumentFetcher(),
        evidence_extractor=FakeEvidenceExtractor(),
        synthesizer=synthesizer or FakeSynthesizer(),
    )


def _terminal_document(
    store: InMemoryArtifactStore,
    document_id: str,
) -> tuple[Document, EvidenceItem]:
    document = Document(
        id=document_id,
        title=document_id,
        url=f"https://example.com/{document_id}",
        text=document_id,
    )
    evidence = EvidenceItem(
        id=f"evidence-{document_id}",
        document_id=document_id,
        claim=document_id,
        source_url=document.url,
    )
    store.put(document)
    store.put(evidence)
    return document, evidence


def _extracted_state(
    evidence_id: str,
    lifecycle_status: Literal["active", "shelved"] = "active",
) -> DocumentState:
    return DocumentState(
        extraction=ExtractProgress(attempt_count=1),
        evidence_ids=[evidence_id],
        lifecycle_status=lifecycle_status,
    )


def test_curation_is_reversible_without_changing_research_artifacts() -> None:
    store = InMemoryArtifactStore()
    document, evidence = _terminal_document(store, "document-1")
    state = AgentState(
        query=UserQuery(text="query"),
        current_step=4,
        documents={
            document.id: _extracted_state(evidence.id)
        },
    )
    original_extraction = state.documents[document.id].extraction.model_copy(deep=True)
    executor = _executor(store)
    reducer = DefaultStateReducer()

    shelve = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params={
            "shelve_document_ids": [document.id],
            "reactivate_document_ids": [],
        },
        rationale="A stronger source covers the same claim.",
    )
    state = reducer.apply(
        state,
        shelve,
        asyncio.run(executor.execute(shelve, state)),
    )

    document_state = state.documents[document.id]
    assert document_state.lifecycle_status == "shelved"
    assert document_state.lifecycle_reason == shelve.rationale
    assert document_state.lifecycle_updated_at_step == 4
    assert document_state.extraction == original_extraction
    assert document_state.evidence_ids == [evidence.id]
    assert store.get(document.id, Document) == document
    assert store.get(evidence.id, EvidenceItem) == evidence

    reactivate = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params={
            "shelve_document_ids": [],
            "reactivate_document_ids": [document.id],
        },
        rationale="The source resolves a remaining conflict.",
    )
    state = reducer.apply(
        state,
        reactivate,
        asyncio.run(executor.execute(reactivate, state)),
    )

    document_state = state.documents[document.id]
    assert document_state.lifecycle_status == "active"
    assert document_state.lifecycle_reason == reactivate.rationale
    assert document_state.lifecycle_updated_at_step == 5
    assert document_state.extraction == original_extraction
    assert document_state.evidence_ids == [evidence.id]


@pytest.mark.parametrize(
    "params",
    [
        {
            "shelve_document_ids": ["unknown"],
            "reactivate_document_ids": [],
        },
        {
            "shelve_document_ids": ["document-1", "pending"],
            "reactivate_document_ids": [],
        },
        {
            "shelve_document_ids": ["shelved"],
            "reactivate_document_ids": [],
        },
    ],
)
def test_curation_rejects_invalid_batches(params: dict[str, list[str]]) -> None:
    store = InMemoryArtifactStore()
    state = AgentState(
        query=UserQuery(text="query"),
        documents={
            "document-1": _extracted_state("evidence-1"),
            "pending": DocumentState(),
            "shelved": _extracted_state("shelved-evidence", "shelved"),
        },
    )
    state_before = state.model_copy(deep=True)
    action = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params=params,
        rationale="Refine the evidence set.",
    )

    with pytest.raises(ValueError, match="invalid lifecycle transitions"):
        asyncio.run(_executor(store).execute(action, state))

    assert state == state_before


def test_curation_allows_exchange_but_rejects_active_overflow() -> None:
    store = InMemoryArtifactStore()
    state = AgentState(
        query=UserQuery(text="query"),
        budget=ExecutionBudget(
            max_document_fetches=2,
            max_active_documents=1,
        ),
        documents={
            "first": _extracted_state("first-evidence"),
            "second": _extracted_state("second-evidence", "shelved"),
        },
    )
    executor = _executor(store)
    overflow = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params={
            "shelve_document_ids": [],
            "reactivate_document_ids": ["second"],
        },
        rationale="Restore the second source.",
    )

    with pytest.raises(ValueError, match="active document limit"):
        asyncio.run(executor.execute(overflow, state))

    exchange = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params={
            "shelve_document_ids": ["first"],
            "reactivate_document_ids": ["second"],
        },
        rationale="The second source is more useful.",
    )
    observation = asyncio.run(executor.execute(exchange, state))
    next_state = DefaultStateReducer().apply(state, exchange, observation)

    assert next_state.documents["first"].lifecycle_status == "shelved"
    assert next_state.documents["second"].lifecycle_status == "active"


class _CapturingSynthesizer:
    def __init__(self) -> None:
        self.requests: list[SynthesisRequest] = []

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        return SynthesisResult(
            answer="active only",
            citations=[item.source_url for item in request.evidence],
        )


def test_finish_excludes_shelved_documents_and_evidence() -> None:
    store = InMemoryArtifactStore()
    active, active_evidence = _terminal_document(store, "active")
    shelved, shelved_evidence = _terminal_document(store, "shelved")
    state = AgentState(
        query=UserQuery(text="query"),
        documents={
            active.id: _extracted_state(active_evidence.id),
            shelved.id: _extracted_state(shelved_evidence.id, "shelved"),
        },
    )
    synthesizer = _CapturingSynthesizer()

    observation = asyncio.run(
        _executor(store, synthesizer=synthesizer).execute(
            AgentAction(type=AgentActionType.FINISH),
            state,
        )
    )

    request = synthesizer.requests[0]
    assert [document.id for document in request.documents] == [active.id]
    assert [evidence.id for evidence in request.evidence] == [active_evidence.id]
    assert isinstance(observation, FinishObservation)
    assert observation.citations == [active.url]


def test_finish_rejects_an_oversized_active_evidence_set() -> None:
    store = InMemoryArtifactStore()
    first, first_evidence = _terminal_document(store, "first")
    second, second_evidence = _terminal_document(store, "second")
    state = AgentState(
        query=UserQuery(text="query"),
        budget=ExecutionBudget(max_document_fetches=2, max_active_documents=1),
        documents={
            first.id: _extracted_state(first_evidence.id),
            second.id: _extracted_state(second_evidence.id),
        },
    )

    with pytest.raises(ValueError, match="requires curation"):
        asyncio.run(
            _executor(store).execute(
                AgentAction(type=AgentActionType.FINISH),
                state,
            )
        )


class _GapRetrievalProvider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        slug = request.query.replace(" ", "-")
        return [
            SearchResult(
                title=request.query,
                url=f"https://example.com/{slug}",
            )
        ]


class _CurationFlowPolicy:
    async def select_action(self, state: AgentState) -> AgentAction:
        if state.current_step == 3:
            return AgentAction(
                type=AgentActionType.CURATE_EVIDENCE,
                params={
                    "shelve_document_ids": [next(iter(state.documents))],
                    "reactivate_document_ids": [],
                },
                rationale="Look for a source that better fills the gap.",
            )
        actions = [
            AgentAction(
                type=AgentActionType.SEARCH,
                params={"query": "initial source"},
            ),
            AgentAction(type=AgentActionType.FETCH_DOCUMENTS),
            AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
            AgentAction(type=AgentActionType.STOP),
            AgentAction(
                type=AgentActionType.SEARCH,
                params={"query": "gap source"},
            ),
            AgentAction(type=AgentActionType.FETCH_DOCUMENTS),
            AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
            AgentAction(type=AgentActionType.FINISH),
        ]
        return actions[state.current_step]


def test_runtime_can_curate_then_fill_an_information_gap() -> None:
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=_CurationFlowPolicy(),
        executor=_executor(
            store,
            retrieval_provider=_GapRetrievalProvider(),
        ),
    )

    output = asyncio.run(
        runtime.run(
            AgentState(
                query=UserQuery(text="query"),
                budget=ExecutionBudget(
                    max_steps=9,
                    max_searches=2,
                    max_document_fetches=2,
                    max_active_documents=1,
                ),
            )
        )
    )
    state = output.result.state

    assert [entry.action.type for entry in state.action_history] == [
        AgentActionType.SEARCH,
        AgentActionType.FETCH_DOCUMENTS,
        AgentActionType.EXTRACT_EVIDENCE,
        AgentActionType.CURATE_EVIDENCE,
        AgentActionType.SEARCH,
        AgentActionType.FETCH_DOCUMENTS,
        AgentActionType.EXTRACT_EVIDENCE,
        AgentActionType.FINISH,
    ]
    assert len(state.documents) == 2
    assert [
        document.lifecycle_status for document in state.documents.values()
    ] == ["shelved", "active"]
    assert state.final_answer is not None
    assert state.citations == ["https://example.com/gap-source"]

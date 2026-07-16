"""Tests for the LLM-backed evidence extractor."""

import asyncio

import pytest

from banso.core import UserQuery
from banso.documents import (
    Document,
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    LLMEvidenceExtractor,
)
from banso.llm import (
    FakeLLMClient,
    LLMError,
    LLMMessageRole,
    LLMRequest,
    LLMResponse,
)


class FailingLLMClient:
    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMError(RuntimeError("prompt contains at least 32769 input tokens"))


class ChunkingLLMClient:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        call_number = len(self.requests)
        if call_number == self.fail_on_call:
            raise LLMError(RuntimeError("chunk request failed"))
        return LLMResponse(content=f'[{{"claim":"chunk-{call_number}"}}]')


def _document() -> Document:
    return Document(
        id="doc-1",
        url="https://example.com/news",
        title="Company A announces AI product",
        text="Company A announced a new AI product on Monday.",
    )


def _document_text_from_request(request: LLMRequest) -> str:
    user_prompt = request.messages[1].content
    document_text = user_prompt.split("Document text:\n", maxsplit=1)[1]
    return document_text.split("\n\nReturn a JSON array", maxsplit=1)[0]


async def _run_llm_evidence_extractor() -> None:
    client = FakeLLMClient(
        content=(
            "["
            '{"claim":"Company A announced a new AI product.",'
            '"supporting_text":"Company A announced a new AI product on Monday.",'
            '"confidence":0.9}'
            "]"
        )
    )
    extractor = LLMEvidenceExtractor(client=client, model="fake-model")
    request = EvidenceExtractionRequest(
        query=UserQuery(text="latest AI product news"),
        document=_document(),
    )

    evidence = await extractor.extract(request)

    assert len(evidence) == 1
    assert evidence[0].document_id == "doc-1"
    assert evidence[0].claim == "Company A announced a new AI product."
    assert evidence[0].supporting_text == (
        "Company A announced a new AI product on Monday."
    )
    assert evidence[0].source_url == "https://example.com/news"
    assert evidence[0].confidence == 0.9
    assert evidence[0].metadata["extractor"] == "llm"

    assert len(client.requests) == 1
    llm_request = client.requests[0]
    assert llm_request.model == "fake-model"
    assert llm_request.temperature == 0.0
    assert llm_request.max_tokens == 2_048
    assert [message.role for message in llm_request.messages] == [
        LLMMessageRole.SYSTEM,
        LLMMessageRole.USER,
    ]

    user_prompt = llm_request.messages[1].content
    assert "latest AI product news" in user_prompt
    assert "Company A announces AI product" in user_prompt
    assert "Company A announced a new AI product on Monday." in user_prompt


async def _run_invalid_json_case() -> None:
    client = FakeLLMClient(content="not json")
    extractor = LLMEvidenceExtractor(client=client)
    with pytest.raises(EvidenceExtractionError) as caught:
        await extractor.extract(
            EvidenceExtractionRequest(
                query=UserQuery(text="latest AI product news"),
                document=_document(),
            )
        )

    assert caught.value.reason == "invalid_json"


async def _run_empty_array_case() -> None:
    extractor = LLMEvidenceExtractor(client=FakeLLMClient(content="[]"))

    evidence = await extractor.extract(
        EvidenceExtractionRequest(
            query=UserQuery(text="latest AI product news"),
            document=_document(),
        )
    )

    assert evidence == []


async def _run_invalid_schema_case() -> None:
    extractor = LLMEvidenceExtractor(client=FakeLLMClient(content='{"claim":"x"}'))

    with pytest.raises(EvidenceExtractionError) as caught:
        await extractor.extract(
            EvidenceExtractionRequest(
                query=UserQuery(text="latest AI product news"),
                document=_document(),
            )
        )

    assert caught.value.reason == "invalid_schema"


async def _run_llm_error_case() -> None:
    document = _document()
    extractor = LLMEvidenceExtractor(client=FailingLLMClient())

    with pytest.raises(EvidenceExtractionError) as caught:
        await extractor.extract(
            EvidenceExtractionRequest(
                query=UserQuery(text="latest AI product news"),
                document=document,
            )
        )

    assert caught.value.reason == "llm_error"
    message = str(caught.value)
    assert "at least 32769 input tokens" in message
    assert "chunk_index=1" in message
    assert "chunk_count=1" in message
    assert f"document_chars={len(document.text)}" in message
    assert f"document_bytes={len(document.text.encode('utf-8'))}" in message
    assert "prompt_chars=" in message
    assert "prompt_bytes=" in message


async def _run_chunked_document_case() -> None:
    document = _document().model_copy(
        update={
            "text": ("第一段证据。" * 8) + "\n\n" + ("Second paragraph. " * 8),
        }
    )
    client = ChunkingLLMClient()
    extractor = LLMEvidenceExtractor(client=client, max_input_bytes=600)

    evidence = await extractor.extract(
        EvidenceExtractionRequest(
            query=UserQuery(text="latest AI product news"),
            document=document,
        )
    )

    assert len(client.requests) > 1
    assert "".join(_document_text_from_request(item) for item in client.requests) == (
        document.text
    )
    assert all(
        sum(len(message.content.encode("utf-8")) for message in item.messages) <= 600
        for item in client.requests
    )
    assert [item.claim for item in evidence] == [
        f"chunk-{index}" for index in range(1, len(client.requests) + 1)
    ]


async def _run_later_chunk_failure_case() -> None:
    document = _document().model_copy(update={"text": "x" * 80})
    client = ChunkingLLMClient(fail_on_call=2)
    extractor = LLMEvidenceExtractor(client=client, max_input_bytes=600)
    request = EvidenceExtractionRequest(
        query=UserQuery(text="latest AI product news"),
        document=document,
    )
    chunk_count = len(extractor._split_document(request))
    assert chunk_count > 1

    with pytest.raises(EvidenceExtractionError) as caught:
        await extractor.extract(request)

    assert caught.value.reason == "llm_error"
    assert len(client.requests) == 2
    assert "chunk request failed" in str(caught.value)
    assert "chunk_index=2" in str(caught.value)
    assert f"chunk_count={chunk_count}" in str(caught.value)


def test_llm_evidence_extractor() -> None:
    asyncio.run(_run_llm_evidence_extractor())


def test_llm_evidence_extractor_raises_for_invalid_json() -> None:
    asyncio.run(_run_invalid_json_case())


def test_llm_evidence_extractor_accepts_empty_array() -> None:
    asyncio.run(_run_empty_array_case())


def test_llm_evidence_extractor_raises_for_invalid_schema() -> None:
    asyncio.run(_run_invalid_schema_case())


def test_llm_evidence_extractor_records_llm_failure_input_sizes() -> None:
    asyncio.run(_run_llm_error_case())


def test_llm_evidence_extractor_chunks_document_within_input_budget() -> None:
    asyncio.run(_run_chunked_document_case())


def test_llm_evidence_extractor_fails_document_when_later_chunk_fails() -> None:
    asyncio.run(_run_later_chunk_failure_case())

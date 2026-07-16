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


def _document() -> Document:
    return Document(
        id="doc-1",
        url="https://example.com/news",
        title="Company A announces AI product",
        text="Company A announced a new AI product on Monday.",
    )


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
    assert llm_request.max_tokens is None
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
    assert f"document_chars={len(document.text)}" in message
    assert f"document_bytes={len(document.text.encode('utf-8'))}" in message
    assert "prompt_chars=" in message
    assert "prompt_bytes=" in message


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

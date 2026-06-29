"""Tests for the LLM-backed evidence extractor."""

import asyncio

from banso.core import UserQuery
from banso.documents import (
    Document,
    EvidenceExtractionRequest,
    LLMEvidenceExtractor,
)
from banso.llm import FakeLLMClient, LLMMessageRole


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
    evidence = await extractor.extract(
        EvidenceExtractionRequest(
            query=UserQuery(text="latest AI product news"),
            document=_document(),
        )
    )

    assert evidence == []


def test_llm_evidence_extractor() -> None:
    asyncio.run(_run_llm_evidence_extractor())


def test_llm_evidence_extractor_returns_empty_list_for_invalid_json() -> None:
    asyncio.run(_run_invalid_json_case())

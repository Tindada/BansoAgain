"""Tests for the LLM-backed synthesizer."""

import asyncio

from banso.core import UserQuery
from banso.documents import EvidenceItem
from banso.llm import FakeLLMClient, LLMMessageRole
from banso.synthesis import LLMSynthesizer, SynthesisRequest


async def _run_llm_synthesizer() -> None:
    client = FakeLLMClient(content="LLM generated news summary.")
    synthesizer = LLMSynthesizer(client=client, model="fake-model")
    evidence = [
        EvidenceItem(
            document_id="doc-1",
            claim="Company A announced a new AI product.",
            supporting_text="Company A announced the product on Monday.",
            source_url="https://example.com/a",
        ),
        EvidenceItem(
            document_id="doc-2",
            claim="Analysts expect a market impact.",
            supporting_text=None,
            source_url="https://example.com/a",
        ),
    ]

    result = await synthesizer.synthesize(
        SynthesisRequest(
            query=UserQuery(text="latest AI company news"),
            evidence=evidence,
        )
    )

    assert result.answer == "LLM generated news summary."
    assert result.citations == ["https://example.com/a"]
    assert result.metadata["llm_model"] == "fake-model"
    assert result.metadata["llm_usage"]["total_tokens"] is not None

    assert len(client.requests) == 1
    llm_request = client.requests[0]
    assert llm_request.model == "fake-model"
    assert llm_request.temperature == 0.2
    assert [message.role for message in llm_request.messages] == [
        LLMMessageRole.SYSTEM,
        LLMMessageRole.USER,
    ]

    user_prompt = llm_request.messages[1].content
    assert "latest AI company news" in user_prompt
    assert "Company A announced a new AI product." in user_prompt
    assert "Analysts expect a market impact." in user_prompt
    assert "https://example.com/a" in user_prompt


def test_llm_synthesizer() -> None:
    asyncio.run(_run_llm_synthesizer())

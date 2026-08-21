"""Tests for the LLM-backed synthesizer."""

import asyncio
from datetime import datetime, timezone

from banso.documents.models import EvidenceItem
from banso.llm.fake import FakeLLMClient
from banso.llm.models import LLMMessageRole
from banso.source import Source, SourceType
from banso.synthesis.llm_synthesizer import LLMSynthesizer
from banso.synthesis.synthesizer import (
    Citation,
    SynthesisEvidenceGroup,
    SynthesisRequest,
)


def _request() -> SynthesisRequest:
    published_at = datetime(2026, 8, 12, tzinfo=timezone.utc)
    return SynthesisRequest(
        query="latest AI company news",
        language="English",
        time_range="past 7 days",
        reference_time=datetime(2026, 8, 13, tzinfo=timezone.utc),
        evidence_groups=[
            SynthesisEvidenceGroup(
                document_id="doc-1",
                title="Company announcement",
                source_url="https://example.com/a",
                source=Source(name="Example News", type=SourceType.NEWS),
                published_at=published_at,
                evidence=[
                    EvidenceItem(
                        document_id="doc-1",
                        claim="Company A announced a new AI product.",
                        supporting_text=(
                            "Company A announced the product on Monday."
                        ),
                        source_url="https://example.com/a",
                    )
                ],
            ),
            SynthesisEvidenceGroup(
                document_id="doc-2",
                title="Market reaction",
                source_url="https://example.com/a",
                evidence=[
                    EvidenceItem(
                        document_id="doc-2",
                        claim="Analysts expect a market impact.",
                        source_url="https://example.com/a",
                    )
                ],
            ),
        ],
    )


async def _run_llm_synthesizer() -> None:
    client = FakeLLMClient(
        content=(
            "Market impact is expected [S2]. The product was announced [S1]. "
            "[S2] [S9]"
        )
    )
    synthesizer = LLMSynthesizer(client=client, model="fake-model")

    result = await synthesizer.synthesize(_request())

    assert result.answer == client.content
    assert result.citations == [
        Citation(
            reference="S2",
            document_id="doc-2",
            source_url="https://example.com/a",
        ),
        Citation(
            reference="S1",
            document_id="doc-1",
            source_url="https://example.com/a",
        ),
    ]
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
    assert "2026-08-13T00:00:00+00:00" in user_prompt
    assert "past 7 days" in user_prompt
    assert "Answer language:\nEnglish" in user_prompt
    assert "[S1]" in user_prompt
    assert "[S2]" in user_prompt
    assert "[S1-E1]" not in user_prompt
    assert "Example News" in user_prompt
    assert "Source type: news" in user_prompt
    assert "Published at: 2026-08-12T00:00:00+00:00" in user_prompt
    assert "Company A announced a new AI product." in user_prompt
    assert "Analysts expect a market impact." in user_prompt


async def _run_synthesizer_without_valid_references() -> None:
    client = FakeLLMClient(content="Uncited answer [S0] [S01] [S3].")
    result = await LLMSynthesizer(client).synthesize(_request())

    assert result.citations == []


def test_llm_synthesizer() -> None:
    asyncio.run(_run_llm_synthesizer())


def test_llm_synthesizer_ignores_invalid_references() -> None:
    asyncio.run(_run_synthesizer_without_valid_references())

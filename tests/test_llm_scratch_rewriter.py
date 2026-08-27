"""Tests for LLM-backed scratch rewriting."""

import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from banso.llm.fake import FakeLLMClient
from banso.scratch.llm_rewriter import LLMScratchRewriter
from banso.scratch.rewriter import (
    ScratchEvidenceGroup,
    ScratchRewriteRequest,
    ScratchRewriteResult,
)


def _request(evidence_text: str) -> ScratchRewriteRequest:
    return ScratchRewriteRequest(
        query="question",
        language="en",
        time_range="week",
        reference_time=datetime(2026, 8, 27, tzinfo=timezone.utc),
        current_scratch="old notes",
        evidence_groups=[
            ScratchEvidenceGroup(
                document_ref="D1",
                title="Document",
                source_url="https://example.com",
                evidence_text=evidence_text,
            )
        ],
    )


def test_rewriter_builds_the_request() -> None:
    evidence_text = "supporting evidence"
    client = FakeLLMClient('{"content":"new notes"}')

    result = asyncio.run(
        LLMScratchRewriter(client, model="fake-model").rewrite(
            _request(evidence_text)
        )
    )

    assert result.content == "new notes"
    llm_request = client.requests[0]
    assert llm_request.response_format == {"type": "json_object"}
    assert llm_request.metadata == {
        "trace": {"operation": "scratch_rewriter.rewrite"}
    }
    prompt = json.loads(llm_request.messages[1].content)
    assert prompt["current_scratch"] == "old notes"
    assert prompt["evidence_groups"][0]["document_ref"] == "D1"
    assert prompt["evidence_groups"][0]["evidence_text"] == evidence_text


def test_scratch_result_enforces_size_limit() -> None:
    with pytest.raises(ValidationError):
        ScratchRewriteResult(content="x" * 32_001)

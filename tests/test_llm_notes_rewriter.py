"""Tests for LLM-backed research notes rewriting."""

import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from banso.llm.fake import FakeLLMClient
from banso.notes.llm_rewriter import LLMNotesRewriter
from banso.notes.rewriter import (
    NotesEvidenceGroup,
    NotesRewriteRequest,
    NotesRewriteResult,
)


def _request(evidence_text: str) -> NotesRewriteRequest:
    return NotesRewriteRequest(
        query="question",
        language="en",
        time_range="week",
        reference_time=datetime(2026, 8, 27, tzinfo=timezone.utc),
        current_notes="old notes",
        evidence_groups=[
            NotesEvidenceGroup(
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
        LLMNotesRewriter(client, model="fake-model").rewrite(
            _request(evidence_text)
        )
    )

    assert result.content == "new notes"
    llm_request = client.requests[0]
    assert llm_request.response_format == {"type": "json_object"}
    assert llm_request.metadata == {
        "trace": {"operation": "notes_rewriter.rewrite"}
    }
    prompt = json.loads(llm_request.messages[1].content)
    assert prompt["current_notes"] == "old notes"
    assert prompt["evidence_groups"][0]["document_ref"] == "D1"
    assert prompt["evidence_groups"][0]["evidence_text"] == evidence_text


def test_notes_result_enforces_size_limit() -> None:
    with pytest.raises(ValidationError):
        NotesRewriteResult(content="x" * 32_001)

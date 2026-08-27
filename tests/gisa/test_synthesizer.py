"""Tests for GISA JSON synthesis and deterministic TSV conversion."""

import asyncio
from datetime import datetime, timezone

import pytest

from banso.benchmarks.gisa_synthesizer import GisaSynthesizer
from banso.llm.fake import FakeLLMClient
from banso.synthesis.synthesizer import SynthesisRequest


def _request(answer_type: str) -> SynthesisRequest:
    return SynthesisRequest(
        query="Benchmark question",
        reference_time=datetime(2026, 8, 19, tzinfo=timezone.utc),
        notes="Intermediate candidate: Answer",
        metadata={"gisa": {"answer_type": answer_type}},
    )


async def _synthesize(content: str, answer_type: str):
    client = FakeLLMClient(content)
    result = await GisaSynthesizer(client).synthesize(_request(answer_type))
    assert client.requests[0].response_format == {"type": "json_object"}
    assert "Research notes:\nIntermediate candidate: Answer" in (
        client.requests[0].messages[1].content
    )
    return result


def test_gisa_item_json_is_converted_to_tsv() -> None:
    result = asyncio.run(_synthesize('{"value":"  Answer  "}', "item"))

    assert result.answer == "```tsv\nValue\nAnswer\n```"


def test_gisa_set_json_is_deduplicated_in_source_order() -> None:
    result = asyncio.run(_synthesize('{"items":["A","B","A"]}', "set"))

    assert result.answer == "```tsv\nItem\nA\nB\n```"


def test_gisa_list_json_preserves_order_and_duplicates() -> None:
    result = asyncio.run(_synthesize('{"items":["B","A","B"]}', "list"))

    assert result.answer == "```tsv\nItem\nB\nA\nB\n```"


def test_gisa_table_json_is_validated_and_converted_to_tsv() -> None:
    result = asyncio.run(
        _synthesize(
            '{"columns":["Name","Year"],"rows":[["A",2024],["B",""]]}',
            "table",
        )
    )

    assert result.answer == (
        "```tsv\nName\tYear\nA\t2024\nB\t\n```"
    )


def test_gisa_json_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="invalid JSON schema"):
        asyncio.run(_synthesize('{"items":["A"]}', "item"))


def test_gisa_table_rejects_mismatched_row_width() -> None:
    with pytest.raises(ValueError, match="invalid JSON schema"):
        asyncio.run(
            _synthesize(
                '{"columns":["Name","Year"],"rows":[["A"]]}',
                "table",
            )
        )

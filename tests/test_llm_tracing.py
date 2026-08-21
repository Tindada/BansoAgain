"""Tests for provider-independent LLM client tracing."""

import asyncio
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Any

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentActionType, AgentState, RetrievalRoute, UserQuery
from banso.documents import (
    Document,
    EvidenceExtractionRequest,
    LLMEvidenceExtractor,
)
from banso.llm import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ThinkingModeLLMClient,
    TracingLLMClient,
)
from banso.policies import LLMNewsPolicy, LLMPolicyError
from banso.research_context import ResearchContextBuilder
from banso.synthesis import (
    LLMSynthesizer,
    SynthesisEvidenceGroup,
    SynthesisRequest,
)
from banso.tracing import InMemoryTraceSink, SpanRecord, Tracer

class StaticResponseClient:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response

    async def generate(self, request: LLMRequest) -> LLMResponse:
        del request
        return self.response.model_copy(deep=True)


class FailingClient:
    async def generate(self, request: LLMRequest) -> LLMResponse:
        del request
        raise LLMError(RuntimeError("provider unavailable"))


def _client(content: str, call_id: str) -> TracingLLMClient:
    return TracingLLMClient(
        StaticResponseClient(
            LLMResponse(
                content=content,
                model="trace-model",
                usage=LLMUsage(
                    input_tokens=11,
                    output_tokens=7,
                    total_tokens=18,
                ),
                raw={"id": call_id, "choices": [{"content": content}]},
                metadata={"provider": "test"},
            )
        )
    )


async def _run_successful_calls() -> list[SpanRecord]:
    sink = InMemoryTraceSink()
    tracer = Tracer(sink)
    with tracer.start_span("test.root") as root:
        policy = LLMNewsPolicy(
            _client(
                '{"type":"stop","params":{},"rationale":"Done."}',
                "policy-call",
            ),
            ResearchContextBuilder(
                InMemoryArtifactStore(),
                [RetrievalRoute.WEB],
            ),
        )
        action = await policy.select_action(
            AgentState(query=UserQuery(text="What happened?"))
        )
        assert action.type == AgentActionType.STOP

        document = Document(
            id="doc-trace",
            title="Announcement",
            url="https://example.com/news",
            text="Company A announced a product.",
        )
        extractor = LLMEvidenceExtractor(
            _client(
                '[{"claim":"Company A announced a product."}]',
                "extract-call",
            )
        )
        evidence = await extractor.extract(
            EvidenceExtractionRequest(
                query=UserQuery(text="Latest product news"),
                document=document,
            )
        )
        assert evidence[0].claim == "Company A announced a product."

        synthesizer = LLMSynthesizer(
            _client("Final answer.", "synthesis-call")
        )
        result = await synthesizer.synthesize(
            SynthesisRequest(
                query=UserQuery(text="Latest product news"),
                reference_time=datetime(2026, 8, 13, tzinfo=timezone.utc),
                evidence_groups=[
                    SynthesisEvidenceGroup(
                        document_id=document.id,
                        title=document.title,
                        source_url=document.url,
                        evidence=evidence,
                    )
                ],
            )
        )
        assert result.answer == "Final answer."
        trace_id = root.trace_id
    return sink.get_trace(trace_id)


def test_llm_clients_record_only_complete_model_io() -> None:
    spans = asyncio.run(_run_successful_calls())
    calls = {
        span.attributes["operation"]: span
        for span in spans
        if span.name == "llm.call"
    }

    assert set(calls) == {
        "news_policy.select_action",
        "evidence_extractor.extract",
        "synthesizer.synthesize",
    }
    for span in calls.values():
        assert span.status == "ok"
        assert span.input["request"]["messages"][0]["role"] == "system"
        assert span.input["request"]["metadata"]["trace"] == span.attributes
        assert span.output["model"] == "trace-model"
        assert span.output["usage"] == {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        }
        assert span.output["response_metadata"] == {"provider": "test"}
        assert "completion" in span.output
        assert "provider_response" in span.output
        assert "parsed_result" not in span.output
        assert "parse_error" not in span.output

    extraction_span = calls["evidence_extractor.extract"]
    assert extraction_span.attributes == {
        "operation": "evidence_extractor.extract",
        "document_id": "doc-trace",
        "chunk_index": 1,
        "chunk_count": 1,
    }
    assert [
        SpanRecord.model_validate_json(span.model_dump_json()) for span in spans
    ] == spans


async def _run_traced_thinking_strip() -> tuple[LLMResponse, SpanRecord]:
    sink = InMemoryTraceSink()
    tracer = Tracer(sink)
    raw_content = "<think>private reasoning</think>\nFinal answer."
    client = TracingLLMClient(
        ThinkingModeLLMClient(
            StaticResponseClient(
                LLMResponse(
                    content=raw_content,
                    raw={"choices": [{"content": raw_content}]},
                )
            )
        )
    )
    with tracer.start_span("test.root") as root:
        response = await client.generate(LLMRequest(messages=[]))
        trace_id = root.trace_id
    llm_span = next(
        span for span in sink.get_trace(trace_id) if span.name == "llm.call"
    )
    return response, llm_span


def test_tracing_records_parser_input_and_preserves_provider_response() -> None:
    response, span = asyncio.run(_run_traced_thinking_strip())

    assert response.content == "Final answer."
    assert span.output["completion"] == "Final answer."
    assert span.output["provider_response"] == {
        "choices": [
            {"content": "<think>private reasoning</think>\nFinal answer."}
        ]
    }


async def _run_with_trace(
    awaitable: Awaitable[Any],
) -> tuple[Exception, SpanRecord, SpanRecord]:
    sink = InMemoryTraceSink()
    tracer = Tracer(sink)
    caught: Exception | None = None
    trace_id = ""
    try:
        with tracer.start_span("test.root") as root:
            trace_id = root.trace_id
            await awaitable
    except Exception as error:
        caught = error
    assert caught is not None
    spans = sink.get_trace(trace_id)
    llm_span = next(span for span in spans if span.name == "llm.call")
    root_span = next(span for span in spans if span.name == "test.root")
    return caught, llm_span, root_span


def test_parse_failure_keeps_successful_llm_call_separate() -> None:
    policy = LLMNewsPolicy(
        _client("not json", "invalid-call"),
        ResearchContextBuilder(
            InMemoryArtifactStore(),
            [RetrievalRoute.WEB],
        ),
    )
    error, llm_span, root_span = asyncio.run(
        _run_with_trace(
            policy.select_action(
                AgentState(query=UserQuery(text="Latest AI news"))
            )
        )
    )

    assert isinstance(error, LLMPolicyError)
    assert llm_span.status == "ok"
    assert llm_span.output["completion"] == "not json"
    assert llm_span.output["provider_response"]["id"] == "invalid-call"
    assert llm_span.output["usage"]["total_tokens"] == 18
    assert root_span.status == "error"
    assert root_span.error is not None
    assert root_span.error.error_type == "LLMPolicyError"


def test_provider_failure_marks_llm_call_as_error_and_keeps_prompt() -> None:
    policy = LLMNewsPolicy(
        TracingLLMClient(FailingClient()),
        ResearchContextBuilder(
            InMemoryArtifactStore(),
            [RetrievalRoute.WEB],
        ),
    )
    error, llm_span, root_span = asyncio.run(
        _run_with_trace(
            policy.select_action(
                AgentState(query=UserQuery(text="Latest AI news"))
            )
        )
    )

    assert isinstance(error, LLMPolicyError)
    assert llm_span.status == "error"
    assert "Latest AI news" in llm_span.input["request"]["messages"][1]["content"]
    assert llm_span.output is None
    assert llm_span.error is not None
    assert llm_span.error.error_type == "LLMError"
    assert root_span.status == "error"


async def _run_concurrent_extractions() -> list[SpanRecord]:
    sink = InMemoryTraceSink()
    tracer = Tracer(sink)
    extractor = LLMEvidenceExtractor(
        _client('[{"claim":"A claim"}]', "concurrent-call")
    )
    documents = [
        Document(
            id=f"doc-{index}",
            title=f"Document {index}",
            url=f"https://example.com/{index}",
            text=f"Text {index}",
        )
        for index in range(2)
    ]
    with tracer.start_span("agent.action.execute") as root:
        await asyncio.gather(
            *(
                extractor.extract(
                    EvidenceExtractionRequest(
                        query=UserQuery(text="query"),
                        document=document,
                    )
                )
                for document in documents
            )
        )
        trace_id = root.trace_id
        parent_span_id = root.span_id
    spans = [
        span for span in sink.get_trace(trace_id) if span.name == "llm.call"
    ]
    assert all(span.parent_span_id == parent_span_id for span in spans)
    return spans


def test_concurrent_evidence_calls_keep_trace_and_document_identity() -> None:
    spans = asyncio.run(_run_concurrent_extractions())

    assert len(spans) == 2
    assert {span.attributes["document_id"] for span in spans} == {"doc-0", "doc-1"}
    assert len({span.span_id for span in spans}) == 2
    assert len({span.trace_id for span in spans}) == 1

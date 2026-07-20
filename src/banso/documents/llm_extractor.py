"""LLM-backed evidence extractor implementation."""

import json
import re

from banso.documents.extractor import EvidenceExtractionError, EvidenceExtractionRequest
from banso.documents.models import EvidenceItem
from banso.llm import LLMClient, LLMError, LLMMessage, LLMMessageRole, LLMRequest


SYSTEM_PROMPT = (
    "You are a news evidence extraction assistant. Extract factual claims from "
    "the provided document that are relevant to the user query. Use only the "
    "provided document. Return only valid JSON, with no markdown or explanation."
)

EVIDENCE_OUTPUT_FORMAT = (
    "Return a JSON array in this schema:\n"
    "[\n"
    "  {\n"
    '    "claim": "...",\n'
    '    "supporting_text": "...",\n'
    '    "confidence": 0.8\n'
    "  }\n"
    "]"
)

DEFAULT_MAX_INPUT_BYTES = 24000
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 20
DEFAULT_MAX_OUTPUT_TOKENS = 2048


def _split_text_by_bytes(text: str, max_bytes: int) -> list[str]:
    if not text:
        return [""]

    segments = re.split(r"(\n[ \t]*\n)", text)  # Capture and keep separators.
    paragraph_segments = [
        "".join(segments[index : index + 2])  # Pair text with its separator.
        for index in range(0, len(segments), 2)
        if "".join(segments[index : index + 2])
    ]
    chunks: list[str] = []
    current = ""
    current_bytes = 0

    for segment in paragraph_segments:
        for part in _split_segment_by_bytes(segment, max_bytes):
            part_bytes = len(part.encode("utf-8"))
            if current and current_bytes + part_bytes > max_bytes:
                chunks.append(current)
                current = ""
                current_bytes = 0
            current += part
            current_bytes += part_bytes

    if current:
        chunks.append(current)
    return chunks


def _split_segment_by_bytes(segment: str, max_bytes: int) -> list[str]:
    if len(segment.encode("utf-8")) <= max_bytes:
        return [segment]

    parts: list[str] = []
    start = 0
    while start < len(segment):
        end = start
        used_bytes = 0
        last_whitespace = None

        while end < len(segment):
            character_bytes = len(segment[end].encode("utf-8"))
            if used_bytes + character_bytes > max_bytes:
                break
            used_bytes += character_bytes
            end += 1
            if segment[end - 1].isspace():
                last_whitespace = end

        if end == start:
            raise ValueError("max_bytes is smaller than one encoded character")
        if end < len(segment) and last_whitespace is not None:
            end = last_whitespace

        parts.append(segment[start:end])
        start = end

    return parts


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


class LLMEvidenceExtractor:
    """Extracts evidence by calling an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    ) -> None:
        if max_input_bytes <= 0:
            raise ValueError("max_input_bytes must be greater than zero")
        if max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be greater than zero")

        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_input_bytes = max_input_bytes
        self.max_chunks_per_document = max_chunks_per_document

    async def extract(self, request: EvidenceExtractionRequest) -> list[EvidenceItem]:
        document_chunks = self._split_document(request)
        evidence: list[EvidenceItem] = []
        chunk_count = len(document_chunks)

        for chunk_index, document_text in enumerate(document_chunks, start=1):
            user_prompt = self._build_user_prompt(request, document_text)
            try:
                response = await self.client.generate(
                    LLMRequest(
                        messages=[
                            LLMMessage(
                                role=LLMMessageRole.SYSTEM,
                                content=SYSTEM_PROMPT,
                            ),
                            LLMMessage(
                                role=LLMMessageRole.USER,
                                content=user_prompt,
                            ),
                        ],
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                )
            except LLMError as error:
                raise EvidenceExtractionError(
                    self._chunk_failure_message(
                        f"LLM evidence request failed: {error}",
                        request,
                        user_prompt,
                        chunk_index,
                        chunk_count,
                    ),
                    reason="llm_error",
                ) from error

            try:
                chunk_evidence = self._parse_items(response.content, request)
            except EvidenceExtractionError as error:
                raise EvidenceExtractionError(
                    self._chunk_failure_message(
                        str(error),
                        request,
                        user_prompt,
                        chunk_index,
                        chunk_count,
                    ),
                    reason=error.reason,
                ) from error

            evidence.extend(chunk_evidence)

        return evidence

    def _split_document(self, request: EvidenceExtractionRequest) -> list[str]:
        empty_prompt = self._build_user_prompt(request, "")
        fixed_prompt_bytes = len((SYSTEM_PROMPT + empty_prompt).encode("utf-8"))
        document_byte_budget = self.max_input_bytes - fixed_prompt_bytes

        if document_byte_budget <= 0:
            document = request.document
            raise EvidenceExtractionError(
                "Evidence prompt exceeds the input budget before document text; "
                f"max_input_bytes={self.max_input_bytes}; "
                f"fixed_prompt_bytes={fixed_prompt_bytes}; "
                f"document_chars={len(document.text)}; "
                f"document_bytes={len(document.text.encode('utf-8'))}",
                reason="input_budget",
            )

        try:
            chunks = _split_text_by_bytes(
                request.document.text,
                document_byte_budget,
            )
        except ValueError as error:
            raise EvidenceExtractionError(
                "Document text cannot fit within the input budget; "
                f"max_input_bytes={self.max_input_bytes}; "
                f"document_byte_budget={document_byte_budget}",
                reason="input_budget",
            ) from error

        if len(chunks) > self.max_chunks_per_document:
            document = request.document
            raise EvidenceExtractionError(
                "Document requires too many evidence extraction chunks; "
                f"max_chunks_per_document={self.max_chunks_per_document}; "
                f"chunk_count={len(chunks)}; "
                f"document_chars={len(document.text)}; "
                f"document_bytes={len(document.text.encode('utf-8'))}",
                reason="document_too_large",
            )

        return chunks

    def _build_user_prompt(
        self,
        request: EvidenceExtractionRequest,
        chunk_text: str,
    ) -> str:
        document = request.document
        return (
            f"User query:\n{request.query.text}\n\n"
            "Maximum evidence items for this document chunk: "
            f"{request.max_items_per_chunk}\n\n"
            f"Document title:\n{document.title}\n\n"
            f"Document URL:\n{document.url}\n\n"
            f"Document text:\n{chunk_text}\n\n"
            f"{EVIDENCE_OUTPUT_FORMAT}"
        )

    def _chunk_failure_message(
        self,
        message: str,
        request: EvidenceExtractionRequest,
        user_prompt: str,
        chunk_index: int,
        chunk_count: int,
    ) -> str:
        document_text = request.document.text
        prompt_text = SYSTEM_PROMPT + user_prompt
        return (
            f"{message}; "
            f"chunk_index={chunk_index}; "
            f"chunk_count={chunk_count}; "
            f"document_chars={len(document_text)}; "
            f"document_bytes={len(document_text.encode('utf-8'))}; "
            f"prompt_chars={len(prompt_text)}; "
            f"prompt_bytes={len(prompt_text.encode('utf-8'))}"
        )

    def _parse_items(
        self,
        content: str,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        try:
            raw_items = json.loads(content)
        except json.JSONDecodeError as error:
            raise EvidenceExtractionError(
                "LLM evidence response is not valid JSON",
                reason="invalid_json",
            ) from error

        if not isinstance(raw_items, list):
            raise EvidenceExtractionError(
                "LLM evidence response must be a JSON array",
                reason="invalid_schema",
            )

        evidence: list[EvidenceItem] = []
        for raw_item in raw_items[: request.max_items_per_chunk]:
            item = self._parse_item(raw_item, request)
            if item is None:
                raise EvidenceExtractionError(
                    "LLM evidence response contains an invalid item",
                    reason="invalid_schema",
                )
            evidence.append(item)
        return evidence

    def _parse_item(
        self,
        raw_item: object,
        request: EvidenceExtractionRequest,
    ) -> EvidenceItem | None:
        if not isinstance(raw_item, dict):
            return None

        claim = raw_item.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            return None

        supporting_text = _optional_string(raw_item.get("supporting_text"))
        confidence = _optional_float(raw_item.get("confidence"))

        return EvidenceItem(
            document_id=request.document.id,
            claim=claim,
            supporting_text=supporting_text,
            source_url=request.document.url,
            published_at=request.document.published_at,
            confidence=confidence,
            metadata={"extractor": "llm"},
        )

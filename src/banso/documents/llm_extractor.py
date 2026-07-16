"""LLM-backed evidence extractor implementation."""

import json

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
    '    "confidence": 0.8,\n'
    '    "source_url": "optional source URL"\n'
    "  }\n"
    "]"
)


class LLMEvidenceExtractor:
    """Extracts evidence by calling an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def extract(self, request: EvidenceExtractionRequest) -> list[EvidenceItem]:
        user_prompt = self._build_user_prompt(request)
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
                self._llm_failure_message(error, request, user_prompt),
                reason="llm_error",
            ) from error

        return self._parse_items(response.content, request)

    def _build_user_prompt(self, request: EvidenceExtractionRequest) -> str:
        document = request.document
        return (
            f"User query:\n{request.query.text}\n\n"
            f"Maximum evidence items: {request.max_items}\n\n"
            f"Document title:\n{document.title}\n\n"
            f"Document URL:\n{document.url}\n\n"
            f"Document text:\n{document.text}\n\n"
            f"{EVIDENCE_OUTPUT_FORMAT}"
        )

    def _llm_failure_message(
        self,
        error: LLMError,
        request: EvidenceExtractionRequest,
        user_prompt: str,
    ) -> str:
        document_text = request.document.text
        prompt_text = SYSTEM_PROMPT + user_prompt
        return (
            f"LLM evidence request failed: {error}; "
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
        for raw_item in raw_items[: request.max_items]:
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

        supporting_text = self._optional_string(raw_item.get("supporting_text"))
        confidence = self._optional_float(raw_item.get("confidence"))
        source_url = self._optional_string(raw_item.get("source_url"))

        return EvidenceItem(
            document_id=request.document.id,
            claim=claim,
            supporting_text=supporting_text,
            source_url=source_url or request.document.url,
            published_at=request.document.published_at,
            confidence=confidence,
            metadata={"extractor": "llm"},
        )

    def _optional_string(self, value: object) -> str | None:
        if isinstance(value, str) and value.strip():
            return value
        return None

    def _optional_float(self, value: object) -> float | None:
        if isinstance(value, int | float):
            return float(value)
        return None

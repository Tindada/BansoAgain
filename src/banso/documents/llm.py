"""LLM-backed evidence extractor implementation."""

import json

from banso.documents.extractor import EvidenceExtractionRequest
from banso.documents.models import EvidenceItem
from banso.llm import LLMClient, LLMMessage, LLMMessageRole, LLMRequest


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
        response = await self.client.generate(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role=LLMMessageRole.SYSTEM,
                        content=SYSTEM_PROMPT,
                    ),
                    LLMMessage(
                        role=LLMMessageRole.USER,
                        content=self._build_user_prompt(request),
                    ),
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

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

    def _parse_items(
        self,
        content: str,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        try:
            raw_items = json.loads(content)
        except json.JSONDecodeError:
            return []

        if not isinstance(raw_items, list):
            return []

        evidence: list[EvidenceItem] = []
        for raw_item in raw_items[: request.max_items]:
            item = self._parse_item(raw_item, request)
            if item is not None:
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

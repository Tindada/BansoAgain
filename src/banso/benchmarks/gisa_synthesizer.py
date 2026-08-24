"""GISA-specific JSON synthesis with deterministic TSV conversion."""

from typing import Annotated, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from banso.benchmarks.gisa import GisaAnswerType, render_gisa_tsv
from banso.llm.client import LLMClient
from banso.llm.models import LLMMessage, LLMMessageRole, LLMRequest
from banso.synthesis.synthesizer import (
    SynthesisEvidenceGroup,
    SynthesisRequest,
    SynthesisResult,
)


SYSTEM_PROMPT = (
    "Answer the user's question precisely using only the supplied evidence. Return "
    "only the values and fields requested by the question; use contextual details to "
    "determine eligibility, but omit unrequested titles, offices, dates, categories, "
    "and explanations from output values. Every textual output value must be in "
    "English. Use a standard English name when available; otherwise translate "
    "descriptive terms and conventionally transliterate proper names. Do not invent "
    "aliases, expand abbreviations, or add unsupported information. For collection "
    "and ranking requests, include every supported result that satisfies the requested "
    "scope, eligibility, ordering, and time frame. Return exactly one JSON object "
    "with the requested schema, using JSON strings for textual and numeric values. "
    "Do not "
    "include Markdown, citations, source labels, or confidence notes."
)

NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
JsonScalar: TypeAlias = NonBlankString | int | float | bool
JsonCell: TypeAlias = JsonScalar | None


class _StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ItemOutput(_StrictOutput):
    value: JsonScalar


class _ItemsOutput(_StrictOutput):
    items: list[JsonScalar]


class _TableOutput(_StrictOutput):
    columns: list[NonBlankString] = Field(min_length=1)
    rows: list[list[JsonCell]]


class GisaSynthesizer:
    """Synthesize validated GISA JSON and convert it to official TSV."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        answer_type = self._answer_type(request)
        response = await self.client.generate(
            LLMRequest(
                messages=[
                    LLMMessage(role=LLMMessageRole.SYSTEM, content=SYSTEM_PROMPT),
                    LLMMessage(
                        role=LLMMessageRole.USER,
                        content=self._build_user_prompt(request, answer_type),
                    ),
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                metadata={"trace": {"operation": "gisa.synthesizer.synthesize"}},
            )
        )
        answer = self._parse_and_render(answer_type, response.content)
        return SynthesisResult(
            answer=answer,
            citations=[],
            metadata={
                "answer_type": answer_type.value,
                "llm_model": response.model,
                "llm_usage": response.usage.model_dump()
                if response.usage is not None
                else None,
            },
        )

    @staticmethod
    def _answer_type(request: SynthesisRequest) -> GisaAnswerType:
        try:
            return GisaAnswerType(request.metadata["gisa"]["answer_type"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid metadata.gisa.answer_type") from error

    @staticmethod
    def _parse_and_render(answer_type: GisaAnswerType, content: str) -> str:
        try:
            if answer_type == GisaAnswerType.ITEM:
                output = _ItemOutput.model_validate_json(content)
                return render_gisa_tsv(answer_type, [[output.value]])
            if answer_type in {GisaAnswerType.SET, GisaAnswerType.LIST}:
                output = _ItemsOutput.model_validate_json(content)
                items = output.items
                if answer_type == GisaAnswerType.SET:
                    items = list(dict.fromkeys(str(item) for item in items))
                return render_gisa_tsv(answer_type, [[item] for item in items])
            output = _TableOutput.model_validate_json(content)
            return render_gisa_tsv(
                answer_type,
                output.rows,
                columns=output.columns,
            )
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid JSON schema for GISA {answer_type.value} answer") from error

    def _build_user_prompt(
        self,
        request: SynthesisRequest,
        answer_type: GisaAnswerType,
    ) -> str:
        evidence_blocks = [
            self._build_evidence_block(index, group)
            for index, group in enumerate(request.evidence_groups, start=1)
        ]
        evidence = "\n\n".join(evidence_blocks) or "No evidence was collected."
        return (
            f"Question:\n{request.query}\n\n"
            f"Reference time:\n{request.reference_time.isoformat()}\n\n"
            f"Answer type:\n{answer_type.value}\n\n"
            f"Required JSON schema:\n{self._json_schema_instruction(answer_type)}\n\n"
            f"Evidence:\n{evidence}"
        )

    @staticmethod
    def _json_schema_instruction(answer_type: GisaAnswerType) -> str:
        if answer_type == GisaAnswerType.ITEM:
            return (
                '{"value": "the single answer"}. Return only the value requested '
                "by the question."
            )
        if answer_type == GisaAnswerType.SET:
            return (
                '{"items": ["every distinct matching item"]}. Include all and only '
                "matching items. Each item must contain only the requested entity "
                "value, without unrequested titles or explanatory details. Order is "
                "not significant."
            )
        if answer_type == GisaAnswerType.LIST:
            return (
                '{"items": ["first item", "second item"]}. Include all requested '
                "items in the requested order. Each item must contain only the "
                "requested entity value, without unrequested titles or explanatory "
                "details."
            )
        return (
            '{"columns": ["Column 1", "Column 2"], '
            '"rows": [["value 1", "value 2"]]}. Column names must correspond '
            "exactly to the fields requested by the question. Every row must have "
            "the same number of values as columns."
        )

    @staticmethod
    def _build_evidence_block(
        index: int,
        group: SynthesisEvidenceGroup,
    ) -> str:
        published_at = (
            group.published_at.isoformat()
            if group.published_at is not None
            else "Unknown"
        )
        return (
            f"[Source {index}]\n"
            f"Title: {group.title}\n"
            f"URL: {group.source_url}\n"
            f"Published at: {published_at}\n"
            f"Evidence:\n{group.evidence_text}"
        )

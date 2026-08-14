"""LLM-backed synthesizer implementation."""

import re

from banso.llm import LLMClient, LLMMessage, LLMMessageRole, LLMRequest
from banso.synthesis.synthesizer import (
    SynthesisEvidenceGroup,
    SynthesisRequest,
    SynthesisResult,
)


SYSTEM_PROMPT = (
    "You are a news synthesis assistant. Answer the user query using only the "
    "provided evidence. Cite factual claims with the corresponding source-group "
    "reference, such as [S1]. Be concise and mention uncertainty when evidence is "
    "limited or conflicting."
)

_GROUP_REFERENCE_PATTERN = re.compile(r"\[S([1-9]\d*)\]")


class LLMSynthesizer:
    """Synthesizes evidence by calling an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        model: str | None = None,
        temperature: float | None = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
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
                metadata={"trace": {"operation": "synthesizer.synthesize"}},
            )
        )
        cited_group_refs, citations = self._citations(
            response.content,
            request.evidence_groups,
        )
        return SynthesisResult(
            answer=response.content,
            citations=citations,
            metadata={
                "llm_model": response.model,
                "llm_usage": response.usage.model_dump()
                if response.usage is not None
                else None,
                "cited_group_refs": cited_group_refs,
            },
        )

    def _build_user_prompt(self, request: SynthesisRequest) -> str:
        group_blocks = [
            self._build_group_block(index, group)
            for index, group in enumerate(request.evidence_groups, start=1)
        ]
        evidence_block = "\n\n".join(group_blocks) or "No evidence provided."
        time_range = request.query.time_range or "Not specified"
        answer_language = (
            request.query.language or "Match the language of the user query"
        )

        return (
            f"User query:\n{request.query.text}\n\n"
            f"Reference time:\n{request.reference_time.isoformat()}\n\n"
            f"Requested time range:\n{time_range}\n\n"
            f"Answer language:\n{answer_language}\n\n"
            f"Evidence groups:\n{evidence_block}\n\n"
            "Write the final news summary in the specified answer language. Cite each "
            "factual statement with one or more applicable source-group references "
            "such as [S1]. Use publication dates to respect the requested time range, "
            "but do not treat a publication date as the event date unless the evidence "
            "supports that conclusion. Prefer relevant, credible, and diverse sources. "
            "Explicitly describe material conflicts or uncertainty. Do not cite a "
            "source group that does not support the statement."
        )

    def _build_group_block(
        self,
        index: int,
        group: SynthesisEvidenceGroup,
    ) -> str:
        source_name = group.source.name if group.source is not None else "Unknown"
        source_type = group.source.type.value if group.source is not None else "unknown"
        published_at = (
            group.published_at.isoformat()
            if group.published_at is not None
            else "Unknown"
        )
        evidence_lines = [
            (
                f"- Claim: {item.claim}\n"
                f"  Supporting text: {item.supporting_text or 'N/A'}"
            )
            for item in group.evidence
        ]
        evidence_block = "\n".join(evidence_lines) or "- No extracted evidence."
        return (
            f"[S{index}]\n"
            f"Title: {group.title}\n"
            f"Source: {source_name}\n"
            f"Source type: {source_type}\n"
            f"URL: {group.source_url}\n"
            f"Published at: {published_at}\n"
            f"Evidence:\n{evidence_block}"
        )

    def _citations(
        self,
        answer: str,
        evidence_groups: list[SynthesisEvidenceGroup],
    ) -> tuple[list[str], list[str]]:
        cited_group_refs: list[str] = []
        citations: list[str] = []
        for match in _GROUP_REFERENCE_PATTERN.finditer(answer):
            index = int(match.group(1))
            if index > len(evidence_groups):
                continue
            group_ref = f"S{index}"
            if group_ref not in cited_group_refs:
                cited_group_refs.append(group_ref)
            source_url = evidence_groups[index - 1].source_url
            if source_url not in citations:
                citations.append(source_url)
        return cited_group_refs, citations

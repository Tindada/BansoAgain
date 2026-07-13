"""LLM-backed synthesizer implementation."""

from banso.documents import EvidenceItem
from banso.llm import LLMClient, LLMMessage, LLMMessageRole, LLMRequest
from banso.synthesis.synthesizer import SynthesisRequest, SynthesisResult


SYSTEM_PROMPT = (
    "You are a news synthesis assistant. Answer the user query using only the "
    "provided evidence. Be concise and mention uncertainty when evidence is limited."
)


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
            )
        )

        return SynthesisResult(
            answer=response.content,
            citations=self._citations(request.evidence),
            metadata={
                "llm_model": response.model,
                "llm_usage": response.usage.model_dump()
                if response.usage is not None
                else None,
            },
        )

    def _build_user_prompt(self, request: SynthesisRequest) -> str:
        evidence_lines = [
            (
                f"{index}. Claim: {item.claim}\n"
                f"   Source: {item.source_url}\n"
                f"   Supporting text: {item.supporting_text or 'N/A'}"
            )
            for index, item in enumerate(request.evidence, start=1)
        ]
        evidence_block = "\n".join(evidence_lines) or "No evidence provided."

        return (
            f"User query:\n{request.query.text}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Write the final news summary."
        )

    def _citations(self, evidence: list[EvidenceItem]) -> list[str]:
        citations: list[str] = []
        for item in evidence:
            if item.source_url not in citations:
                citations.append(item.source_url)
        return citations

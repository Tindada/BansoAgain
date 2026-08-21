"""Fake synthesizer for local smoke tests."""

from banso.core.observation import Citation
from banso.synthesis.synthesizer import SynthesisRequest, SynthesisResult


class FakeSynthesizer:
    """Returns deterministic synthesis output without LLM calls."""

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if not request.evidence_groups:
            return SynthesisResult(answer="No evidence was available to synthesize.")

        summaries = [
            f"{' '.join(item.claim for item in group.evidence)} [S{index}]"
            for index, group in enumerate(request.evidence_groups, start=1)
        ]
        citations = [
            Citation(
                reference=f"S{index}",
                document_id=group.document_id,
                source_url=group.source_url,
            )
            for index, group in enumerate(request.evidence_groups, start=1)
        ]
        return SynthesisResult(
            answer=f"Fake summary for '{request.query.text}': {' '.join(summaries)}",
            citations=citations,
        )

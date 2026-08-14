"""Fake synthesizer for local smoke tests."""

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
        citations = list(
            dict.fromkeys(group.source_url for group in request.evidence_groups)
        )
        return SynthesisResult(
            answer=f"Fake summary for '{request.query.text}': {' '.join(summaries)}",
            citations=citations,
        )

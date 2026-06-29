"""Fake synthesizer for local smoke tests."""

from banso.synthesis.synthesizer import SynthesisRequest, SynthesisResult


class FakeSynthesizer:
    """Returns deterministic synthesis output without LLM calls."""

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if not request.evidence:
            return SynthesisResult(answer="No evidence was available to synthesize.")

        claims = " ".join(item.claim for item in request.evidence)
        citations = [item.source_url for item in request.evidence]
        return SynthesisResult(
            answer=f"Fake summary for '{request.query.text}': {claims}",
            citations=citations,
        )

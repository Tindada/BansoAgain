"""Multi-source synthesis components."""

from banso.core.observation import Citation
from banso.synthesis.fake import FakeSynthesizer
from banso.synthesis.llm_synthesizer import LLMSynthesizer
from banso.synthesis.synthesizer import (
    SynthesisEvidenceGroup,
    SynthesisRequest,
    SynthesisResult,
    Synthesizer,
)

__all__ = [
    "Citation",
    "FakeSynthesizer",
    "LLMSynthesizer",
    "SynthesisEvidenceGroup",
    "SynthesisRequest",
    "SynthesisResult",
    "Synthesizer",
]

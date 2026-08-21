"""Deterministic source classification for search results."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from banso.retrieval.models import (
    SearchResult,
    Source,
    SourceClassificationRecord,
    SourceClassificationReport,
)
from banso.retrieval.url_utils import publisher_domain, publisher_home_url
from banso.source_types import SourceType


class SourceClassifierConfig(BaseModel):
    """Configuration for domain-based source classification."""

    source_domains: dict[str, SourceType] = Field(default_factory=dict)

    @field_validator("source_domains")
    @classmethod
    def _normalize_domains(
        cls, values: dict[str, SourceType]
    ) -> dict[str, SourceType]:
        return {
            domain.strip().lower().removeprefix("www."): source_type
            for domain, source_type in values.items()
        }


class SourceClassification(BaseModel):
    """Source metadata inferred for one search result."""

    search_result_id: str
    publisher_domain: str
    source_type: SourceType
    classification_source: Literal["domain", "provider", "unknown"]


class SourceClassificationResult(BaseModel):
    """Classified search results and their classification metadata."""

    results: list[SearchResult]
    classifications: list[SourceClassification]

    @property
    def input_count(self) -> int:
        return len(self.classifications)

    @property
    def recognized_count(self) -> int:
        return sum(
            classification.source_type != SourceType.UNKNOWN
            for classification in self.classifications
        )

    @property
    def unknown_count(self) -> int:
        return self.input_count - self.recognized_count

    @property
    def report(self) -> SourceClassificationReport:
        """Return a trace-safe summary without duplicating result payloads."""

        return SourceClassificationReport(
            input_count=self.input_count,
            recognized_count=self.recognized_count,
            unknown_count=self.unknown_count,
            classifications=[
                SourceClassificationRecord.model_validate(
                    classification.model_dump(mode="json")
                )
                for classification in self.classifications
            ],
        )


class SourceClassifier:
    """Enrich search results with source metadata without selecting them."""

    def __init__(self, config: SourceClassifierConfig | None = None) -> None:
        self.config = config or SourceClassifierConfig()

    def classify(self, result: SearchResult) -> SourceClassification:
        """Classify one result using the registry before provider metadata."""

        domain = publisher_domain(result.url)
        source_type, classification_source = self._source_classification(
            result, domain
        )
        return SourceClassification(
            search_result_id=result.id,
            publisher_domain=domain,
            source_type=source_type,
            classification_source=classification_source,
        )

    def apply(self, results: list[SearchResult]) -> SourceClassificationResult:
        """Classify and enrich every result while preserving input order."""

        classifications = [self.classify(result) for result in results]
        enriched_results = [
            self._with_classified_source(result, classification)
            for result, classification in zip(results, classifications)
        ]
        return SourceClassificationResult(
            results=enriched_results,
            classifications=classifications,
        )

    def _source_classification(
        self, result: SearchResult, domain: str
    ) -> tuple[SourceType, Literal["domain", "provider", "unknown"]]:
        source_type = self.config.source_domains.get(domain)
        if source_type is not None:
            return source_type, "domain"
        if result.source is not None and result.source.type != SourceType.UNKNOWN:
            return result.source.type, "provider"
        return SourceType.UNKNOWN, "unknown"

    @staticmethod
    def _with_classified_source(
        result: SearchResult,
        classification: SourceClassification,
    ) -> SearchResult:
        source = result.source or Source(
            name=classification.publisher_domain or "Unknown publisher",
            url=publisher_home_url(result.url),
        )
        source = source.model_copy(update={"type": classification.source_type})
        return result.model_copy(update={"source": source})

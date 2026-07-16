"""Deterministic source classification for search results."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from banso.retrieval.models import SearchResult, Source, SourceType
from banso.retrieval.url_utils import publisher_domain, publisher_home_url


def _default_source_domains() -> dict[str, SourceType]:
    """Return the AI-news-focused source registry."""

    official = {
        "about.fb.com",
        "ai.google",
        "ai.google.dev",
        "ai.meta.com",
        "anthropic.com",
        "aws.amazon.com",
        "blog.google",
        "cloud.google.com",
        "cloudflare.com",
        "cohere.com",
        "databricks.com",
        "deepmind.google",
        "ibm.com",
        "microsoft.com",
        "mistral.ai",
        "nvidia.com",
        "openai.com",
        "research.google",
        "salesforce.com",
        "stability.ai",
        "x.ai",
        # Government and regulator publications use the existing OFFICIAL type.
        "bis.gov",
        "canada.ca",
        "commerce.gov",
        "congress.gov",
        "europa.eu",
        "federalregister.gov",
        "ftc.gov",
        "gov.uk",
        "legislation.gov.uk",
        "nist.gov",
        "whitehouse.gov",
    }
    research = {
        "aaai.org",
        "aclanthology.org",
        "arxiv.org",
        "dl.acm.org",
        "icml.cc",
        "ieee.org",
        "jmlr.org",
        "nature.com",
        "neurips.cc",
        "openreview.net",
        "proceedings.mlr.press",
        "science.org",
    }
    leaderboard = {"lmarena.ai", "swebench.com"}
    news = {
        "apnews.com",
        "arstechnica.com",
        "axios.com",
        "bbc.co.uk",
        "bbc.com",
        "bloomberg.com",
        "cnbc.com",
        "cnn.com",
        "ft.com",
        "nytimes.com",
        "reuters.com",
        "semafor.com",
        "techcrunch.com",
        "technologyreview.com",
        "theguardian.com",
        "theverge.com",
        "washingtonpost.com",
        "wired.com",
        "wsj.com",
    }
    social = {
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "medium.com",
        "quora.com",
        "reddit.com",
        "substack.com",
        "twitter.com",
        "x.com",
        "youtube.com",
    }
    aggregators = {"news.google.com", "newser.com"}

    registry: dict[str, SourceType] = {}
    for domains, source_type in (
        (official, SourceType.OFFICIAL),
        (research, SourceType.RESEARCH),
        (leaderboard, SourceType.LEADERBOARD),
        (news, SourceType.NEWS),
        (social, SourceType.SOCIAL),
        (aggregators, SourceType.AGGREGATOR),
    ):
        registry.update(dict.fromkeys(domains, source_type))
    return registry


class SourceClassifierConfig(BaseModel):
    """Configuration for domain-based source classification."""

    source_domains: dict[str, SourceType] = Field(
        default_factory=_default_source_domains
    )


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

    def report(self) -> dict[str, Any]:
        """Return a trace-safe summary without duplicating result payloads."""

        return {
            "input_count": self.input_count,
            "recognized_count": self.recognized_count,
            "unknown_count": self.unknown_count,
            "classifications": [
                classification.model_dump(mode="json")
                for classification in self.classifications
            ],
        }


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
        for configured_domain, source_type in self.config.source_domains.items():
            if self._domain_matches(domain, configured_domain):
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

    @staticmethod
    def _domain_matches(domain: str, candidate: str) -> bool:
        return domain == candidate or domain.endswith(f".{candidate}")

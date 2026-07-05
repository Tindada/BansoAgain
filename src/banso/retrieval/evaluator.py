"""Deterministic evaluation of search results before document reading."""

from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from banso.retrieval.models import SearchResult, Source, SourceType
from banso.retrieval.publisher import publisher_domain, publisher_home_url


def _default_source_domains() -> dict[str, SourceType]:
    return {
        "ai.google": SourceType.OFFICIAL,
        "anthropic.com": SourceType.OFFICIAL,
        "deepmind.google": SourceType.OFFICIAL,
        "openai.com": SourceType.OFFICIAL,
        "arxiv.org": SourceType.RESEARCH,
        "openreview.net": SourceType.RESEARCH,
        "proceedings.mlr.press": SourceType.RESEARCH,
        "lmarena.ai": SourceType.LEADERBOARD,
        "swebench.com": SourceType.LEADERBOARD,
        "apnews.com": SourceType.NEWS,
        "reuters.com": SourceType.NEWS,
        "twitter.com": SourceType.SOCIAL,
        "x.com": SourceType.SOCIAL,
        "news.google.com": SourceType.AGGREGATOR,
        "newser.com": SourceType.AGGREGATOR,
    }


class SearchResultEvaluatorConfig(BaseModel):
    """Configuration for deterministic source admission decisions."""

    source_domains: dict[str, SourceType] = Field(
        default_factory=_default_source_domains
    )
    blocked_domains: set[str] = Field(default_factory=set)
    approved_social_accounts: set[str] = Field(default_factory=set)
    reject_aggregators: bool = True
    accept_unknown_sources: bool = False


class SearchResultEvaluation(BaseModel):
    """Structured admission decision for one search result."""

    search_result_id: str
    accepted: bool
    score: float
    publisher_domain: str
    source_type: SourceType
    reasons: list[str] = Field(default_factory=list)


class SearchResultEvaluationResult(BaseModel):
    """Admitted search results and all source admission decisions."""

    results: list[SearchResult]
    evaluations: list[SearchResultEvaluation]

    @property
    def input_count(self) -> int:
        return len(self.evaluations)

    @property
    def accepted_count(self) -> int:
        return len(self.results)

    @property
    def rejected_count(self) -> int:
        return self.input_count - self.accepted_count

    def report(self) -> dict[str, Any]:
        """Return a trace-safe summary without duplicating result payloads."""

        return {
            "input_count": self.input_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "evaluations": [
                evaluation.model_dump(mode="json")
                for evaluation in self.evaluations
            ],
        }


class SearchResultEvaluator:
    """Evaluates source suitability without I/O or downstream dependencies."""

    def __init__(self, config: SearchResultEvaluatorConfig | None = None) -> None:
        self.config = config or SearchResultEvaluatorConfig()

    def evaluate(
        self,
        result: SearchResult,
    ) -> SearchResultEvaluation:
        """Evaluate one result using publisher identity and provider metadata."""

        domain = publisher_domain(result.url)
        source_type = self._source_type(result, domain)
        score = self._quality_score(source_type, result.metadata)
        reasons: list[str] = []
        accepted = True

        if self._domain_matches(domain, self.config.blocked_domains):
            accepted = False
            reasons.append("blocked_domain")
        elif source_type == SourceType.AGGREGATOR and self.config.reject_aggregators:
            accepted = False
            reasons.append("aggregator_source")
        elif source_type == SourceType.SOCIAL:
            account = self._social_account(result.url)
            if account not in self.config.approved_social_accounts:
                accepted = False
                reasons.append("unverified_social_source")
            else:
                reasons.append("approved_social_source")
        elif source_type == SourceType.UNKNOWN:
            reasons.append("unknown_source")
            accepted = self.config.accept_unknown_sources
        else:
            reasons.append(f"recognized_{source_type.value}_source")

        if not domain:
            accepted = False
            reasons.append("missing_publisher_domain")

        return SearchResultEvaluation(
            search_result_id=result.id,
            accepted=accepted,
            score=score,
            publisher_domain=domain,
            source_type=source_type,
            reasons=reasons,
        )

    def apply(
        self,
        results: list[SearchResult],
    ) -> SearchResultEvaluationResult:
        """Evaluate a batch of search results."""

        admitted: list[SearchResult] = []
        evaluations: list[SearchResultEvaluation] = []

        for result in results:
            evaluation = self.evaluate(result)
            if evaluation.accepted:
                admitted.append(self._with_evaluated_source(result, evaluation))
            evaluations.append(evaluation)

        return SearchResultEvaluationResult(
            results=admitted,
            evaluations=evaluations,
        )

    def _source_type(self, result: SearchResult, domain: str) -> SourceType:
        for configured_domain, source_type in self.config.source_domains.items():
            if self._domain_matches(domain, {configured_domain}):
                return source_type
        if result.source is not None and result.source.type != SourceType.UNKNOWN:
            return result.source.type
        return SourceType.UNKNOWN

    @staticmethod
    def _with_evaluated_source(
        result: SearchResult,
        evaluation: SearchResultEvaluation,
    ) -> SearchResult:
        source = result.source or Source(
            name=evaluation.publisher_domain or "Unknown publisher",
            url=publisher_home_url(result.url),
        )
        source = source.model_copy(update={"type": evaluation.source_type})
        return result.model_copy(update={"source": source})

    @staticmethod
    def _domain_matches(domain: str, candidates: set[str]) -> bool:
        return any(
            domain == candidate or domain.endswith(f".{candidate}")
            for candidate in candidates
        )

    @staticmethod
    def _social_account(url: str) -> str:
        parsed = urlsplit(url)
        domain = publisher_domain(url)
        path_parts = [part for part in parsed.path.split("/") if part]
        if not domain or not path_parts:
            return ""
        return f"{domain}/{path_parts[0].lower()}"

    @staticmethod
    def _quality_score(
        source_type: SourceType,
        metadata: dict[str, Any],
    ) -> float:
        # TODO: Calibrate source weights and normalize provider-specific scores
        # using event-aware evaluation data before treating this as a rank score.
        source_scores = {
            SourceType.OFFICIAL: 1.0,
            SourceType.RESEARCH: 0.95,
            SourceType.LEADERBOARD: 0.9,
            SourceType.NEWS: 0.8,
            SourceType.SOCIAL: 0.7,
            SourceType.BLOG: 0.5,
            SourceType.UNKNOWN: 0.3,
            SourceType.AGGREGATOR: 0.1,
        }
        source_score = source_scores[source_type]
        value = metadata.get("score")
        if isinstance(value, int | float) and not isinstance(value, bool):
            provider_score = min(1.0, max(0.0, float(value)))
            return round(0.7 * source_score + 0.3 * provider_score, 4)
        return source_score

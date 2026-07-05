"""Tests for deterministic search result evaluation."""

from banso.retrieval import (
    SearchResult,
    SearchResultEvaluator,
    SearchResultEvaluatorConfig,
    SourceType,
)


def test_evaluator_accepts_recognized_sources_and_enriches_result() -> None:
    result = SearchResult(
        title="Model release",
        url="https://www.openai.com/index/model-release",
        metadata={"provider": "tavily", "score": 0.9},
    )

    output = SearchResultEvaluator().apply([result])

    assert len(output.results) == 1
    admitted = output.results[0]
    assert admitted.source is not None
    assert admitted.source.type == SourceType.OFFICIAL
    assert output.evaluations[0].publisher_domain == "openai.com"
    assert output.evaluations[0].score == 0.97
    assert output.input_count == 1
    assert output.accepted_count == 1
    assert output.rejected_count == 0
    assert output.evaluations[0].reasons == [
        "recognized_official_source"
    ]


def test_evaluator_rejects_aggregators_unknown_and_unapproved_social_sources() -> None:
    results = [
        SearchResult(title="Summary", url="https://newser.com/story/1"),
        SearchResult(title="Unknown", url="https://unknown.example/story/1"),
        SearchResult(title="Post", url="https://x.com/someone/status/1"),
    ]

    output = SearchResultEvaluator().apply(results)

    assert output.results == []
    assert [evaluation.reasons for evaluation in output.evaluations] == [
        ["aggregator_source"],
        ["unknown_source"],
        ["unverified_social_source"],
    ]


def test_evaluator_accepts_all_results_from_configured_source() -> None:
    evaluator = SearchResultEvaluator(
        SearchResultEvaluatorConfig(
            source_domains={"lab.example": SourceType.RESEARCH},
        )
    )
    results = [
        SearchResult(title="First", url="https://lab.example/first"),
        SearchResult(title="Second", url="https://lab.example/second"),
    ]

    output = evaluator.apply(results)

    assert [result.title for result in output.results] == ["First", "Second"]
    assert output.evaluations[1].reasons == ["recognized_research_source"]


def test_evaluator_accepts_only_explicitly_approved_social_account() -> None:
    url = "https://x.com/researcher/status/1"
    evaluator = SearchResultEvaluator(
        SearchResultEvaluatorConfig(approved_social_accounts={"x.com/researcher"})
    )

    output = evaluator.apply(
        [SearchResult(title="Researcher announcement", url=url)],
    )

    assert len(output.results) == 1
    assert output.evaluations[0].reasons == ["approved_social_source"]

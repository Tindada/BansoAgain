"""Tests for deterministic search result source classification."""

from banso.retrieval import (
    SearchResult,
    Source,
    SourceClassifier,
    SourceClassifierConfig,
    SourceType,
)


def test_classifier_enriches_recognized_source() -> None:
    result = SearchResult(
        title="Model release",
        url="https://www.openai.com/index/model-release",
        metadata={"provider": "tavily", "score": 0.9},
    )

    output = SourceClassifier().apply([result])

    assert len(output.results) == 1
    assert output.results[0].source is not None
    assert output.results[0].source.type == SourceType.OFFICIAL
    assert output.classifications[0].publisher_domain == "openai.com"
    assert output.classifications[0].classification_source == "domain"
    assert output.input_count == 1
    assert output.recognized_count == 1
    assert output.unknown_count == 0


def test_classifier_retains_aggregator_unknown_and_social_results() -> None:
    results = [
        SearchResult(title="Summary", url="https://newser.com/story/1"),
        SearchResult(title="Unknown", url="https://unknown.example/story/1"),
        SearchResult(title="Post", url="https://x.com/someone/status/1"),
    ]

    output = SourceClassifier().apply(results)

    assert [result.title for result in output.results] == [
        "Summary",
        "Unknown",
        "Post",
    ]
    assert [item.source_type for item in output.classifications] == [
        SourceType.AGGREGATOR,
        SourceType.UNKNOWN,
        SourceType.SOCIAL,
    ]
    assert output.recognized_count == 2
    assert output.unknown_count == 1


def test_classifier_supports_configured_source_registry() -> None:
    classifier = SourceClassifier(
        SourceClassifierConfig(
            source_domains={"lab.example": SourceType.RESEARCH},
        )
    )
    results = [
        SearchResult(title="First", url="https://lab.example/first"),
        SearchResult(title="Second", url="https://sub.lab.example/second"),
    ]

    output = classifier.apply(results)

    assert [result.source.type for result in output.results if result.source] == [
        SourceType.RESEARCH,
        SourceType.RESEARCH,
    ]
    assert all(
        item.classification_source == "domain"
        for item in output.classifications
    )


def test_classifier_uses_expanded_registry_before_provider_type() -> None:
    results = [
        SearchResult(
            title="Google Cloud release",
            url="https://cloud.google.com/blog/products/ai-machine-learning/release",
            source=Source(name="Google", type=SourceType.BLOG),
        ),
        SearchResult(
            title="Meta AI release",
            url="https://ai.meta.com/blog/model-release/",
        ),
        SearchResult(
            title="Microsoft documentation",
            url="https://learn.microsoft.com/en-us/azure/ai-services/update",
        ),
        SearchResult(
            title="EU policy",
            url="https://consilium.europa.eu/en/policies/ai/",
        ),
    ]

    output = SourceClassifier().apply(results)

    assert [item.source_type for item in output.classifications] == [
        SourceType.OFFICIAL,
        SourceType.OFFICIAL,
        SourceType.OFFICIAL,
        SourceType.OFFICIAL,
    ]
    assert all(
        item.classification_source == "domain"
        for item in output.classifications
    )


def test_classifier_uses_provider_type_for_unregistered_domain() -> None:
    result = SearchResult(
        title="Provider-classified report",
        url="https://new-lab.example/report",
        source=Source(name="New Lab", type=SourceType.RESEARCH),
    )

    output = SourceClassifier().apply([result])

    assert output.results[0].source is not None
    assert output.results[0].source.type == SourceType.RESEARCH
    assert output.classifications[0].classification_source == "provider"


def test_classifier_marks_unregistered_untyped_source_unknown() -> None:
    result = SearchResult(
        title="Unknown report",
        url="https://unknown.example/report",
        metadata={"provider": "tavily", "score": 0.99},
    )

    output = SourceClassifier().apply([result])

    assert output.results[0].source is not None
    assert output.results[0].source.type == SourceType.UNKNOWN
    assert output.classifications[0].classification_source == "unknown"
    assert output.report.unknown_count == 1

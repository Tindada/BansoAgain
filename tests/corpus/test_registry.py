"""Tests for curated source registry validation and lookup."""

import json
from pathlib import Path

import pytest

from banso.corpus import SourceRegistry, SourceRegistryError


def _write_registry(path: Path, sources: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "sources": sources}),
        encoding="utf-8",
    )
    return path


def _source(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": "example-research",
        "name": "Example Research",
        "allowed_domains": ["research.example.org"],
        "allowed_path_prefixes": ["/publications"],
        "feeds": ["https://research.example.org/feed.xml"],
        "sitemaps": ["https://research.example.org/sitemap.xml"],
    }
    values.update(overrides)
    return values


def test_registry_loads_and_matches_only_approved_content_scope(
    tmp_path: Path,
) -> None:
    registry = SourceRegistry.load(
        _write_registry(tmp_path / "sources.json", [_source()])
    )

    source = registry.get("example-research")
    assert source is not None
    assert source.allowed_domains == ("research.example.org",)
    assert registry.match_url(
        "https://research.example.org/publications/report"
    ) == source
    assert registry.match_url("https://research.example.org/publication") is None
    assert registry.match_url("https://other.example.org/publications/report") is None


def test_registry_ignores_disabled_sources(tmp_path: Path) -> None:
    registry = SourceRegistry.load(
        _write_registry(tmp_path / "sources.json", [_source(enabled=False)])
    )

    assert registry.enabled_sources() == ()
    assert (
        registry.match_url("https://research.example.org/publications/report") is None
    )


@pytest.mark.parametrize(
    "sources, expected_message",
    [
        ([_source(), _source(name="Duplicate")], "source ids must be unique"),
        (
            [_source(feeds=["https://feeds.example.org/research.xml"])],
            "outside allowed_domains",
        ),
        (
            [_source(allowed_path_prefixes=["publications"])],
            "invalid path",
        ),
        (
            [_source(allowed_domains=["https://research.example.org"])],
            "invalid domain",
        ),
        (
            [_source(allowed_domains=["research.example.org."])],
            "invalid domain",
        ),
    ],
)
def test_registry_rejects_invalid_scope_configuration(
    tmp_path: Path,
    sources: list[dict[str, object]],
    expected_message: str,
) -> None:
    path = _write_registry(tmp_path / "sources.json", sources)

    with pytest.raises(SourceRegistryError, match=expected_message):
        SourceRegistry.load(path)


def test_registry_wraps_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(SourceRegistryError, match="not valid JSON"):
        SourceRegistry.load(path)

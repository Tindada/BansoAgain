"""End-to-end coverage for local corpus management commands."""

import json
from pathlib import Path
from typing import Sequence

from banso.apps import corpus as corpus_app
from banso.corpus import (
    CorpusDocumentStatus,
    CorpusDocumentWrite,
    CorpusSyncResult,
)


class _FakeSyncService:
    def __init__(self, store) -> None:
        self._store = store

    async def sync_source(self, source) -> CorpusSyncResult:
        document = self._store.upsert(
            CorpusDocumentWrite(
                source_id=source.id,
                url="https://example.org/reports/official-update",
                status=CorpusDocumentStatus.ACTIVE,
                title="Official update",
                text="The official laboratory released a new agentic AI system.",
            )
        )
        return CorpusSyncResult(documents=(document,), failures=())


class _FakeEmbeddingProvider:
    model = "test-embedding"
    dimensions = 2

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


def test_sync_rebuild_and_search_commands(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    registry_path = tmp_path / "sources.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "example",
                        "name": "Example",
                        "allowed_domains": ["example.org"],
                        "allowed_path_prefixes": ["/reports"],
                        "feeds": ["https://example.org/feed.xml"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    database_path = tmp_path / "data" / "corpus.sqlite3"
    index_path = tmp_path / "data" / "corpus.lance"
    monkeypatch.setattr(corpus_app, "CorpusSyncService", _FakeSyncService)
    monkeypatch.setattr(
        corpus_app,
        "OpenAIEmbeddingProvider",
        lambda **_kwargs: _FakeEmbeddingProvider(),
    )
    monkeypatch.setenv("BANSO_EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("BANSO_EMBEDDING_DIMENSIONS", "2")

    assert corpus_app.main(
        [
            "sync",
            "--registry",
            str(registry_path),
            "--database",
            str(database_path),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 1

    assert corpus_app.main(
        [
            "reindex",
            "--database",
            str(database_path),
            "--index",
            str(index_path),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == {"chunks": 1}

    assert corpus_app.main(
        [
            "search",
            "agentic AI",
            "--index",
            str(index_path),
            "--mode",
            "hybrid",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "hybrid"
    assert output["results"][0]["title"] == "Official update"

    monkeypatch.delenv("BANSO_EMBEDDING_MODEL")
    monkeypatch.delenv("BANSO_EMBEDDING_DIMENSIONS")
    assert corpus_app.main(
        [
            "search",
            "official",
            "--index",
            str(index_path),
            "--mode",
            "bm25",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["results"]

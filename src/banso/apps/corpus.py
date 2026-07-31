"""Command-line management for the local trusted-source corpus."""

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from banso.corpus import (
    CorpusSearchMode,
    CorpusSyncService,
    LanceCorpusIndex,
    OpenAIEmbeddingProvider,
    SQLiteCorpusStore,
    SourceRegistry,
)

DEFAULT_REGISTRY_PATH = Path("config/trusted_sources.json")
DEFAULT_DATABASE_PATH = Path("data/corpus.sqlite3")
DEFAULT_INDEX_PATH = Path("data/corpus.lance")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="synchronize enabled sources")
    sync_parser.add_argument(
        "--registry",
        type=Path,
        default=Path(os.getenv("BANSO_CORPUS_REGISTRY", DEFAULT_REGISTRY_PATH)),
    )
    sync_parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.getenv("BANSO_CORPUS_DATABASE", DEFAULT_DATABASE_PATH)),
    )

    reindex_parser = subparsers.add_parser(
        "reindex",
        help="rebuild the derived hybrid index",
    )
    reindex_parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.getenv("BANSO_CORPUS_DATABASE", DEFAULT_DATABASE_PATH)),
    )
    reindex_parser.add_argument(
        "--index",
        type=Path,
        default=Path(os.getenv("BANSO_CORPUS_INDEX", DEFAULT_INDEX_PATH)),
    )

    search_parser = subparsers.add_parser("search", help="search the local index")
    search_parser.add_argument("query")
    search_parser.add_argument(
        "--index",
        type=Path,
        default=Path(os.getenv("BANSO_CORPUS_INDEX", DEFAULT_INDEX_PATH)),
    )
    search_parser.add_argument(
        "--mode",
        choices=tuple(CorpusSearchMode),
        type=CorpusSearchMode,
        default=CorpusSearchMode.HYBRID,
    )
    search_parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    if args.command == "sync":
        return asyncio.run(_sync(args.registry, args.database))

    if args.command == "reindex":
        args.database.parent.mkdir(parents=True, exist_ok=True)
        args.index.parent.mkdir(parents=True, exist_ok=True)
        index = LanceCorpusIndex(
            args.index,
            embedding_provider=_embedding_provider_from_env(),
        )
        with SQLiteCorpusStore(args.database) as store:
            chunk_count = index.rebuild(store)
        _print_json({"chunks": chunk_count})
        return 0

    index = LanceCorpusIndex(
        args.index,
        embedding_provider=(
            None
            if args.mode == CorpusSearchMode.BM25
            else _embedding_provider_from_env()
        ),
    )
    results = index.search(args.query, limit=args.limit, mode=args.mode)
    _print_json(
        {
            "mode": args.mode,
            "results": [
                {
                    "score": result.score,
                    **asdict(result.chunk),
                }
                for result in results
            ],
        }
    )
    return 0


async def _sync(registry_path: Path, database_path: Path) -> int:
    registry = SourceRegistry.load(registry_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    document_count = 0
    failure_count = 0

    with SQLiteCorpusStore(database_path) as store:
        service = CorpusSyncService(store)
        for source in registry.enabled_sources():
            result = await service.sync_source(source)
            document_count += len(result.documents)
            failure_count += len(result.failures)
            summaries.append(
                {
                    "source_id": source.id,
                    "documents": len(result.documents),
                    "failures": [
                        {"url": failure.url, "reason": failure.reason}
                        for failure in result.failures
                    ],
                }
            )

    _print_json(
        {
            "sources": summaries,
            "documents": document_count,
            "failures": failure_count,
        }
    )
    return 1 if failure_count else 0


def _embedding_provider_from_env() -> OpenAIEmbeddingProvider:
    model = os.getenv("BANSO_EMBEDDING_MODEL")
    dimensions = os.getenv("BANSO_EMBEDDING_DIMENSIONS")
    if model is None or dimensions is None:
        raise RuntimeError(
            "BANSO_EMBEDDING_MODEL and BANSO_EMBEDDING_DIMENSIONS are required"
        )
    return OpenAIEmbeddingProvider(
        model=model,
        dimensions=int(dimensions),
        base_url=os.getenv("BANSO_EMBEDDING_BASE_URL"),
        api_key=os.getenv("BANSO_EMBEDDING_API_KEY"),
    )

def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, default=str))

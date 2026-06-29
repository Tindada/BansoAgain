"""Artifact store interface and in-memory implementation."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

TArtifact = TypeVar("TArtifact", bound=BaseModel)


class ArtifactStore(Protocol):
    """Stores structured artifacts produced during an agent run."""

    def put(self, artifact: BaseModel) -> str:
        """Store an artifact and return its id."""
        ...

    def get(self, artifact_id: str, artifact_type: type[TArtifact]) -> TArtifact | None:
        """Return an artifact by id and expected type."""
        ...

    def list(self, artifact_type: type[TArtifact]) -> list[TArtifact]:
        """Return all artifacts of a given type."""
        ...


class InMemoryArtifactStore:
    """Simple in-memory artifact store for local runs and smoke tests."""

    def __init__(self) -> None:
        self._artifacts: dict[str, BaseModel] = {}

    def put(self, artifact: BaseModel) -> str:
        artifact_id = getattr(artifact, "id", None)
        if not isinstance(artifact_id, str):
            raise ValueError("artifact must expose a string 'id' field")

        self._artifacts[artifact_id] = artifact
        return artifact_id

    def get(self, artifact_id: str, artifact_type: type[TArtifact]) -> TArtifact | None:
        artifact = self._artifacts.get(artifact_id)
        if isinstance(artifact, artifact_type):
            return artifact
        return None

    def list(self, artifact_type: type[TArtifact]) -> list[TArtifact]:
        return [
            artifact
            for artifact in self._artifacts.values()
            if isinstance(artifact, artifact_type)
        ]

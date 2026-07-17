"""Tests for structured action observations."""

from banso.core import Observation


def test_observation_defaults_to_empty_data() -> None:
    observation = Observation()

    assert observation.data == {}


def test_observation_round_trips_as_json() -> None:
    observation = Observation(
        data={
            "successfully_read_document_count": 1,
            "failed_document_count": 1,
            "document_ids": ["document-1"],
            "document_read_failures": [
                {"reason": "timeout", "url": "https://example.com"}
            ],
        },
    )

    restored = Observation.model_validate_json(observation.model_dump_json())

    assert restored == observation

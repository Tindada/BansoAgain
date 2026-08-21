"""Tests for structured action observations."""

import pytest
from pydantic import ValidationError

from banso.core.action import AgentAction, AgentActionType
from banso.core.observation import (
    Citation,
    ExtractionSuccess,
    FinishObservation,
)
from banso.core.state import ActionHistoryEntry


def test_observation_round_trips_as_json_inside_history() -> None:
    entry = ActionHistoryEntry(
        step_index=0,
        action=AgentAction(type=AgentActionType.FINISH),
        observation=FinishObservation(
            final_answer="Answer",
            citations=[
                Citation(
                    reference="S1",
                    document_id="document-1",
                    source_url="https://example.com",
                )
            ],
        ),
    )

    restored = ActionHistoryEntry.model_validate_json(entry.model_dump_json())

    assert restored == entry
    assert isinstance(restored.observation, FinishObservation)


def test_extraction_success_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(ValidationError, match="evidence_ids must be unique"):
        ExtractionSuccess(
            document_id="document-1",
            evidence_ids=["evidence-1", "evidence-1"],
        )

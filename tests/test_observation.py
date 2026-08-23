"""Tests for structured action observations."""

from banso.agent.action import AgentAction, AgentActionType
from banso.agent.observation import FinishObservation
from banso.agent.state import ActionHistoryEntry
from banso.synthesis.synthesizer import Citation


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

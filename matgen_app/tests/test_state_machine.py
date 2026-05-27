# tests/test_state_machine.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state_machine import StructureState, StateTransition


class TestStructureState:
    """Test StructureState enum values."""

    def test_all_states_defined(self):
        """All 8 states should be defined."""
        expected_states = {
            "generated", "rejected_precheck", "predicted",
            "filtered_in", "filtered_out", "validated", "rejected",
            "exported_for_training"
        }
        actual = {s.value for s in StructureState}
        assert actual == expected_states

    def test_state_is_string_enum(self):
        """State values should be string type for database storage."""
        assert StructureState.GENERATED == "generated"
        assert StructureState.PREDICTED == "predicted"
        assert StructureState.VALIDATED == "validated"


class TestStateTransition:
    """Test StateTransition validation logic."""

    def test_valid_transition_generated_to_predicted(self):
        """Generated can transition to predicted (after Eqv2-Lite prediction)."""
        assert StateTransition.can_transition(
            StructureState.GENERATED, StructureState.PREDICTED
        ) is True

    def test_valid_transition_generated_to_rejected_precheck(self):
        """Generated can transition to rejected_precheck (parsing failure)."""
        assert StateTransition.can_transition(
            StructureState.GENERATED, StructureState.REJECTED_PRECHECK
        ) is True

    def test_valid_transition_predicted_to_filtered_in(self):
        """Predicted can transition to filtered_in (within tolerance)."""
        assert StateTransition.can_transition(
            StructureState.PREDICTED, StructureState.FILTERED_IN
        ) is True

    def test_valid_transition_predicted_to_filtered_out(self):
        """Predicted can transition to filtered_out (outside tolerance)."""
        assert StateTransition.can_transition(
            StructureState.PREDICTED, StructureState.FILTERED_OUT
        ) is True

    def test_valid_transition_filtered_in_to_validated(self):
        """Filtered_in can transition to validated (expert approval)."""
        assert StateTransition.can_transition(
            StructureState.FILTERED_IN, StructureState.VALIDATED
        ) is True

    def test_valid_transition_filtered_in_to_rejected(self):
        """Filtered_in can transition to rejected (expert rejection)."""
        assert StateTransition.can_transition(
            StructureState.FILTERED_IN, StructureState.REJECTED
        ) is True

    def test_valid_transition_validated_to_exported(self):
        """Validated can transition to exported_for_training."""
        assert StateTransition.can_transition(
            StructureState.VALIDATED, StructureState.EXPORTED_FOR_TRAINING
        ) is True

    def test_invalid_transition_backward_forbidden(self):
        """No backward transitions allowed (单向性)."""
        assert StateTransition.can_transition(
            StructureState.PREDICTED, StructureState.GENERATED
        ) is False

    def test_invalid_transition_skip_forbidden(self):
        """Cannot skip states (must go through predicted)."""
        assert StateTransition.can_transition(
            StructureState.GENERATED, StructureState.FILTERED_IN
        ) is False

    def test_invalid_transition_filtered_out_terminal(self):
        """Terminal states have no outgoing transitions."""
        assert StateTransition.can_transition(
            StructureState.FILTERED_OUT, StructureState.VALIDATED
        ) is False

    def test_terminal_states(self):
        """Terminal states should be correctly identified."""
        terminal = {
            StructureState.REJECTED_PRECHECK,
            StructureState.FILTERED_OUT,
            StructureState.REJECTED,
            StructureState.EXPORTED_FOR_TRAINING,
        }
        assert StateTransition.TERMINAL_STATES == terminal

    def test_is_terminal_true(self):
        """Terminal states return True for is_terminal."""
        assert StateTransition.is_terminal(StructureState.FILTERED_OUT) is True
        assert StateTransition.is_terminal(StructureState.REJECTED) is True

    def test_is_terminal_false(self):
        """Non-terminal states return False for is_terminal."""
        assert StateTransition.is_terminal(StructureState.GENERATED) is False
        assert StateTransition.is_terminal(StructureState.PREDICTED) is False
        assert StateTransition.is_terminal(StructureState.FILTERED_IN) is False

    def test_get_next_states_from_generated(self):
        """Get all valid next states from generated."""
        next_states = StateTransition.get_next_states(StructureState.GENERATED)
        assert next_states == {StructureState.REJECTED_PRECHECK, StructureState.PREDICTED}

    def test_get_next_states_from_filtered_in(self):
        """Get all valid next states from filtered_in."""
        next_states = StateTransition.get_next_states(StructureState.FILTERED_IN)
        assert next_states == {StructureState.VALIDATED, StructureState.REJECTED}

    def test_get_next_states_terminal_empty(self):
        """Terminal states have no next states."""
        next_states = StateTransition.get_next_states(StructureState.FILTERED_OUT)
        assert next_states == set()

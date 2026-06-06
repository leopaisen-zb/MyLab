# core/state_machine.py
from enum import Enum
from typing import Optional, Set

class StructureState(str, Enum):
    GENERATED = "generated"
    REJECTED_PRECHECK = "rejected_precheck"
    PREDICTED = "predicted"
    FILTERED_IN = "filtered_in"
    FILTERED_OUT = "filtered_out"
    DFT_VERIFIED = "dft_verified"          # P1-1：DFT 回填后进入此中间态
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXPORTED_FOR_TRAINING = "exported_for_training"

class StateTransition:
    VALID_TRANSITIONS = {
        StructureState.GENERATED: {StructureState.REJECTED_PRECHECK, StructureState.PREDICTED},
        StructureState.REJECTED_PRECHECK: set(),
        StructureState.PREDICTED: {StructureState.FILTERED_IN, StructureState.FILTERED_OUT},
        # P1-1：filtered_in 不再直接到 validated，须经 dft_verified
        StructureState.FILTERED_IN: {StructureState.DFT_VERIFIED, StructureState.REJECTED},
        StructureState.FILTERED_OUT: set(),
        # P1-1：dft_verified 可通过专家审查转到 validated 或 rejected
        StructureState.DFT_VERIFIED: {StructureState.VALIDATED, StructureState.REJECTED},
        StructureState.VALIDATED: {StructureState.EXPORTED_FOR_TRAINING},
        StructureState.REJECTED: set(),
        StructureState.EXPORTED_FOR_TRAINING: set(),
    }

    TERMINAL_STATES = {
        StructureState.REJECTED_PRECHECK,
        StructureState.FILTERED_OUT,
        StructureState.REJECTED,
        StructureState.EXPORTED_FOR_TRAINING,
    }

    @classmethod
    def can_transition(cls, from_state: StructureState, to_state: StructureState) -> bool:
        return to_state in cls.VALID_TRANSITIONS.get(from_state, set())

    @classmethod
    def is_terminal(cls, state: StructureState) -> bool:
        return state in cls.TERMINAL_STATES

    @classmethod
    def get_next_states(cls, current_state: StructureState) -> Set[StructureState]:
        return cls.VALID_TRANSITIONS.get(current_state, set())
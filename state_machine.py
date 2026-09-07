"""AI trader state machine.

WAITING -> ANALYZING -> PLANNING -> RISK_CHECK -> EXECUTING -> MONITORING -> REVIEW
"""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    WAITING = "WAITING"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    RISK_CHECK = "RISK_CHECK"
    EXECUTING = "EXECUTING"
    MONITORING = "MONITORING"
    REVIEW = "REVIEW"
    STOPPED = "STOPPED"


_TRANSITIONS = {
    State.WAITING: {State.ANALYZING},
    State.ANALYZING: {State.PLANNING, State.WAITING},
    State.PLANNING: {State.RISK_CHECK, State.WAITING},
    State.RISK_CHECK: {State.EXECUTING, State.WAITING, State.MONITORING},
    State.EXECUTING: {State.MONITORING, State.WAITING},
    State.MONITORING: {State.REVIEW, State.WAITING},
    State.REVIEW: {State.WAITING, State.STOPPED},
    State.STOPPED: set(),
}


class StateMachine:
    def __init__(self) -> None:
        self.state = State.WAITING
        self.history: list[tuple[State, State, str]] = []

    def can(self, target: State) -> bool:
        return target in _TRANSITIONS.get(self.state, set())

    def transition(self, target: State, note: str = "") -> State:
        if not self.can(target):
            raise ValueError(
                f"Illegal transition {self.state.value} -> {target.value}"
            )
        self.history.append((self.state, target, note))
        self.state = target
        return self.state

    def reset(self) -> None:
        self.state = State.WAITING
        self.history.clear()

    def summary(self) -> str:
        if not self.history:
            return "no transitions"
        return " -> ".join(
            f"{a.value}>{b.value}" for a, b, _ in self.history[-8:]
        )

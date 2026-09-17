from __future__ import annotations

from enum import Enum


class AppState(Enum):
    STOPPED = "STOPPED"
    CALIBRATING = "CALIBRATING"
    READY = "READY"
    DRY_RUN = "DRY_RUN"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"


_VALID_TRANSITIONS: dict[AppState, set[AppState]] = {
    AppState.STOPPED: {AppState.CALIBRATING, AppState.READY, AppState.DRY_RUN, AppState.ERROR},
    AppState.CALIBRATING: {AppState.READY, AppState.STOPPED, AppState.ERROR},
    AppState.READY: {AppState.DRY_RUN, AppState.RUNNING, AppState.STOPPED, AppState.ERROR},
    AppState.DRY_RUN: {AppState.RUNNING, AppState.READY, AppState.STOPPED, AppState.EMERGENCY_STOP, AppState.ERROR},
    AppState.RUNNING: {AppState.READY, AppState.STOPPED, AppState.EMERGENCY_STOP, AppState.ERROR},
    AppState.ERROR: {AppState.STOPPED},
    AppState.EMERGENCY_STOP: {AppState.STOPPED},
}


class App:
    def __init__(self) -> None:
        self._state = AppState.STOPPED
        self._history: list[tuple[AppState, AppState]] = []

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def history(self) -> list[tuple[AppState, AppState]]:
        return list(self._history)

    def transition(self, new_state: AppState) -> None:
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {self._state.value} -> {new_state.value}. "
                f"Allowed: {sorted(s.value for s in allowed)}"
            )
        old = self._state
        self._state = new_state
        self._history.append((old, new_state))

    def can_transition(self, new_state: AppState) -> bool:
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        return new_state in allowed

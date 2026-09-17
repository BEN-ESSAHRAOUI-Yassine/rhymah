from __future__ import annotations

import logging
from enum import Enum

from app.rhythm.note import KeyAction, KeyboardEvent

logger = logging.getLogger(__name__)


class KeyState(Enum):
    UP = "UP"
    DOWN = "DOWN"


class KeyboardStateMachine:
    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = keys or ["A", "S", "D", "J", "K", "L", "F8"]
        self._state: dict[str, KeyState] = {k: KeyState.UP for k in self._keys}
        self._log: list[tuple[float, str, str]] = []

    @property
    def state(self) -> dict[str, KeyState]:
        return dict(self._state)

    def is_down(self, key: str) -> bool:
        return self._state.get(key, KeyState.UP) == KeyState.DOWN

    def apply_event(self, event: KeyboardEvent) -> bool:
        key = event.key
        if key not in self._state:
            logger.warning("Unknown key: %s", key)
            return False

        if event.action == KeyAction.DOWN:
            if self._state[key] == KeyState.DOWN:
                logger.debug("Duplicate DOWN ignored: %s", key)
                return False
            self._state[key] = KeyState.DOWN
            self._log.append((event.timestamp, key, "DOWN"))
            return True

        elif event.action == KeyAction.UP:
            if self._state[key] == KeyState.UP:
                logger.debug("Duplicate UP ignored: %s", key)
                return False
            self._state[key] = KeyState.UP
            self._log.append((event.timestamp, key, "UP"))
            return True

        return False

    def release_all(self) -> list[str]:
        released = []
        for key in self._keys:
            if self._state[key] == KeyState.DOWN:
                self._state[key] = KeyState.UP
                released.append(key)
        if released:
            logger.info("Released all keys: %s", released)
        return released

    def get_log(self) -> list[tuple[float, str, str]]:
        return list(self._log)

    def clear_log(self) -> None:
        self._log.clear()

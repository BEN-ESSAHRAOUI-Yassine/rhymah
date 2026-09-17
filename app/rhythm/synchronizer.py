from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rhythm.note import KeyAction, KeyboardEvent

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    window_ms: int = 8


class Synchronizer:
    def __init__(self, config: SyncConfig | None = None) -> None:
        self._config = config or SyncConfig()
        self._next_group = 1

    def group(self, events: list[KeyboardEvent]) -> list[KeyboardEvent]:
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: (e.timestamp, e.action.value))
        window = self._config.window_ms / 1000.0

        groups: list[list[KeyboardEvent]] = []
        current: list[KeyboardEvent] = [sorted_events[0]]

        for event in sorted_events[1:]:
            if event.timestamp - current[0].timestamp <= window:
                current.append(event)
            else:
                groups.append(current)
                current = [event]
        groups.append(current)

        result: list[KeyboardEvent] = []
        for group in groups:
            gid = self._next_group
            self._next_group += 1
            for e in group:
                result.append(KeyboardEvent(
                    timestamp=e.timestamp,
                    key=e.key,
                    action=e.action,
                    note_id=e.note_id,
                    synchronization_group=gid,
                ))

        return result

    def reset(self) -> None:
        self._next_group = 1

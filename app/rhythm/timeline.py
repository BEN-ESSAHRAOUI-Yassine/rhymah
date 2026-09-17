from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rhythm.note import KeyAction, KeyboardEvent, NoteType
from app.rhythm.timing import TimingEstimate

logger = logging.getLogger(__name__)


@dataclass
class TimelineConfig:
    synchronization_window_ms: int = 8
    short_note_duration_ms: int = 80


class Timeline:
    def __init__(self, config: TimelineConfig | None = None) -> None:
        self._config = config or TimelineConfig()
        self._events: list[KeyboardEvent] = []
        self._next_group = 1

    @property
    def events(self) -> list[KeyboardEvent]:
        return sorted(self._events, key=lambda e: (e.timestamp, e.action.value))

    def add_estimate(self, estimate: TimingEstimate, lane_to_key: dict[str, str]) -> None:
        key = lane_to_key.get(estimate.lane, estimate.lane)

        hit_event = KeyboardEvent(
            timestamp=estimate.hit_time,
            key=key,
            action=KeyAction.DOWN,
            note_id=estimate.track_id,
        )
        self._events.append(hit_event)

        if estimate.note_type == NoteType.LONG and estimate.release_time is not None:
            release_event = KeyboardEvent(
                timestamp=estimate.release_time,
                key=key,
                action=KeyAction.UP,
                note_id=estimate.track_id,
            )
            self._events.append(release_event)
        elif estimate.note_type == NoteType.SHORT:
            release_time = estimate.hit_time + self._config.short_note_duration_ms / 1000.0
            release_event = KeyboardEvent(
                timestamp=release_time,
                key=key,
                action=KeyAction.UP,
                note_id=estimate.track_id,
            )
            self._events.append(release_event)

    def synchronize(self) -> list[KeyboardEvent]:
        events = self.events
        if not events:
            return []

        window = self._config.synchronization_window_ms / 1000.0
        groups: list[list[KeyboardEvent]] = []
        current_group: list[KeyboardEvent] = [events[0]]

        for event in events[1:]:
            if event.timestamp - current_group[0].timestamp <= window:
                current_group.append(event)
            else:
                groups.append(current_group)
                current_group = [event]
        groups.append(current_group)

        result: list[KeyboardEvent] = []
        for group in groups:
            group_id = self._next_group
            self._next_group += 1
            for event in group:
                synchronized = KeyboardEvent(
                    timestamp=event.timestamp,
                    key=event.key,
                    action=event.action,
                    note_id=event.note_id,
                    synchronization_group=group_id,
                )
                result.append(synchronized)

        return result

    def clear(self) -> None:
        self._events.clear()
        self._next_group = 1

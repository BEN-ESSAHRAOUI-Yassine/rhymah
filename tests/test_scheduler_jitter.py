from __future__ import annotations

import time

import pytest

from app.input.key_state import KeyboardStateMachine
from app.input.keyboard import KeyboardDriver
from app.input.scheduler import Scheduler
from app.rhythm.note import KeyAction, KeyboardEvent


def _make_event(key: str = "A", action: KeyAction = KeyAction.DOWN, ts: float = 10.0) -> KeyboardEvent:
    return KeyboardEvent(timestamp=ts, key=key, action=action, note_id=1)


class TestKeyboardEventSerialization:
    def test_to_dict(self):
        event = KeyboardEvent(
            timestamp=10.250,
            key="D",
            action=KeyAction.DOWN,
            note_id=3,
            synchronization_group=1,
            confidence=0.95,
        )
        d = event.to_dict()
        assert d["timestamp"] == 10.250
        assert d["key"] == "D"
        assert d["action"] == "DOWN"
        assert d["note_id"] == 3
        assert d["synchronization_group"] == 1
        assert d["confidence"] == 0.95

    def test_from_dict(self):
        d = {
            "timestamp": 10.250,
            "key": "D",
            "action": "DOWN",
            "note_id": 3,
            "synchronization_group": 1,
            "confidence": 0.95,
        }
        event = KeyboardEvent.from_dict(d)
        assert event.timestamp == 10.250
        assert event.key == "D"
        assert event.action == KeyAction.DOWN
        assert event.note_id == 3
        assert event.synchronization_group == 1
        assert event.confidence == 0.95

    def test_roundtrip(self):
        original = KeyboardEvent(
            timestamp=10.250,
            key="J",
            action=KeyAction.UP,
            note_id=5,
            synchronization_group=2,
            confidence=0.88,
        )
        d = original.to_dict()
        restored = KeyboardEvent.from_dict(d)
        assert restored.timestamp == original.timestamp
        assert restored.key == original.key
        assert restored.action == original.action
        assert restored.note_id == original.note_id
        assert restored.confidence == original.confidence

    def test_from_dict_defaults(self):
        d = {"timestamp": 1.0, "key": "A", "action": "DOWN"}
        event = KeyboardEvent.from_dict(d)
        assert event.confidence == 1.0
        assert event.synchronization_group is None

    def test_confidence_field_default(self):
        event = KeyboardEvent(timestamp=1.0, key="A", action=KeyAction.DOWN, note_id=1)
        assert event.confidence == 1.0


class TestSchedulerJitter:
    def _make_scheduler(self, jitter_enabled=False, min_ms=0.0, max_ms=15.0):
        driver = KeyboardDriver(dry_run=True)
        key_state = KeyboardStateMachine()
        return Scheduler(
            driver, key_state, dry_run=True,
            jitter_enabled=jitter_enabled,
            jitter_min_ms=min_ms,
            jitter_max_ms=max_ms,
        )

    def test_jitter_disabled_by_default(self):
        sched = self._make_scheduler()
        events = [_make_event(ts=10.0)]
        sched.schedule(events)
        assert sched.pending_count == 1

    def test_configure_jitter(self):
        sched = self._make_scheduler()
        sched.configure_jitter(enabled=True, min_ms=5.0, max_ms=10.0)
        assert sched._jitter_enabled is True
        assert sched._jitter_min_ms == 5.0
        assert sched._jitter_max_ms == 10.0

    def test_jitter_adds_variation(self):
        sched = self._make_scheduler(jitter_enabled=True, min_ms=5.0, max_ms=15.0)
        timestamps = []
        for _ in range(20):
            sched._queue.clear()
            event = _make_event(ts=10.0)
            sched.schedule([event])
            timestamps.append(sched._queue[0].timestamp)

        min_ts = min(timestamps)
        max_ts = max(timestamps)
        assert max_ts - min_ts > 0.001, "Jitter should add variation"

    def test_jitter_within_bounds(self):
        sched = self._make_scheduler(jitter_enabled=True, min_ms=5.0, max_ms=15.0)
        for _ in range(50):
            sched._queue.clear()
            event = _make_event(ts=10.0)
            sched.schedule([event])
            ts = sched._queue[0].timestamp
            jitter_ms = (ts - 10.0) * 1000
            assert 5.0 <= jitter_ms <= 15.0, f"Jitter {jitter_ms:.2f}ms out of range"

    def test_jitter_disabled_no_change(self):
        sched = self._make_scheduler(jitter_enabled=False)
        for _ in range(20):
            sched._queue.clear()
            event = _make_event(ts=10.0)
            sched.schedule([event])
            ts = sched._queue[0].timestamp
            assert ts == 10.0

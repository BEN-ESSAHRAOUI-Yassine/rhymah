from __future__ import annotations

from app.rhythm.note import KeyAction, KeyboardEvent
from app.rhythm.synchronizer import SyncConfig, Synchronizer


def _e(key: str, ts: float, action: KeyAction = KeyAction.DOWN) -> KeyboardEvent:
    return KeyboardEvent(timestamp=ts, key=key, action=action, note_id=1)


class TestSynchronizer:
    def test_empty(self):
        s = Synchronizer()
        assert s.group([]) == []

    def test_single_event(self):
        s = Synchronizer()
        result = s.group([_e("A", 1.0)])
        assert len(result) == 1
        assert result[0].synchronization_group == 1

    def test_two_close_events_same_group(self):
        cfg = SyncConfig(window_ms=10)
        s = Synchronizer(config=cfg)
        result = s.group([_e("A", 1.0), _e("D", 1.005)])
        groups = {e.synchronization_group for e in result}
        assert len(groups) == 1

    def test_two_distant_events_different_groups(self):
        cfg = SyncConfig(window_ms=5)
        s = Synchronizer(config=cfg)
        result = s.group([_e("A", 1.0), _e("D", 1.1)])
        groups = {e.synchronization_group for e in result}
        assert len(groups) == 2

    def test_mixed_down_up_same_group(self):
        cfg = SyncConfig(window_ms=20)
        s = Synchronizer(config=cfg)
        events = [
            _e("A", 1.0, KeyAction.DOWN),
            _e("D", 1.005, KeyAction.DOWN),
            _e("A", 1.015, KeyAction.UP),
        ]
        result = s.group(events)
        groups = {e.synchronization_group for e in result}
        assert len(groups) == 1

    def test_sorted_output(self):
        s = Synchronizer()
        result = s.group([_e("D", 2.0), _e("A", 1.0)])
        assert result[0].timestamp <= result[1].timestamp

    def test_reset(self):
        s = Synchronizer()
        s.group([_e("A", 1.0)])
        s.reset()
        result = s.group([_e("A", 2.0)])
        assert result[0].synchronization_group == 1

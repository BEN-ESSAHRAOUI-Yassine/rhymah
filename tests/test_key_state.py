from __future__ import annotations

from app.input.key_state import KeyState, KeyboardStateMachine
from app.rhythm.note import KeyAction, KeyboardEvent


def _down(key: str = "A", ts: float = 1.0) -> KeyboardEvent:
    return KeyboardEvent(timestamp=ts, key=key, action=KeyAction.DOWN, note_id=1)


def _up(key: str = "A", ts: float = 1.1) -> KeyboardEvent:
    return KeyboardEvent(timestamp=ts, key=key, action=KeyAction.UP, note_id=1)


class TestKeyboardStateMachine:
    def test_initial_state_all_up(self):
        ks = KeyboardStateMachine()
        for key in ["A", "S", "D", "J", "K", "L"]:
            assert ks.state[key] == KeyState.UP

    def test_down_event(self):
        ks = KeyboardStateMachine()
        assert ks.apply_event(_down("A"))
        assert ks.is_down("A")

    def test_up_event(self):
        ks = KeyboardStateMachine()
        ks.apply_event(_down("A"))
        assert ks.apply_event(_up("A"))
        assert not ks.is_down("A")

    def test_duplicate_down_rejected(self):
        ks = KeyboardStateMachine()
        ks.apply_event(_down("A"))
        assert not ks.apply_event(_down("A"))
        assert ks.is_down("A")

    def test_duplicate_up_rejected(self):
        ks = KeyboardStateMachine()
        assert not ks.apply_event(_up("A"))

    def test_release_all(self):
        ks = KeyboardStateMachine()
        ks.apply_event(_down("A"))
        ks.apply_event(_down("S"))
        released = ks.release_all()
        assert "A" in released
        assert "S" in released
        assert not ks.is_down("A")
        assert not ks.is_down("S")

    def test_release_all_nothing_held(self):
        ks = KeyboardStateMachine()
        released = ks.release_all()
        assert released == []

    def test_unknown_key_rejected(self):
        ks = KeyboardStateMachine()
        event = KeyboardEvent(timestamp=1.0, key="X", action=KeyAction.DOWN, note_id=1)
        assert not ks.apply_event(event)

    def test_log_recorded(self):
        ks = KeyboardStateMachine()
        ks.apply_event(_down("D", ts=1.0))
        ks.apply_event(_up("D", ts=1.1))
        log = ks.get_log()
        assert len(log) == 2
        assert log[0] == (1.0, "D", "DOWN")
        assert log[1] == (1.1, "D", "UP")

    def test_clear_log(self):
        ks = KeyboardStateMachine()
        ks.apply_event(_down("A"))
        ks.clear_log()
        assert ks.get_log() == []

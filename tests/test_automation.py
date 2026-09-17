from __future__ import annotations

import pytest

from app.automation import AutomationController, AutomationMetrics, EMERGENCY_KEY
from app.input.key_state import KeyboardStateMachine
from app.input.keyboard import KeyboardDriver
from app.input.scheduler import Scheduler
from app.rhythm.note import KeyAction, KeyboardEvent
from app.state import App, AppState


def _make_controller(dry_run: bool = True) -> AutomationController:
    app = App()
    driver = KeyboardDriver(dry_run=dry_run)
    ks = KeyboardStateMachine()
    sched = Scheduler(driver, ks, dry_run=dry_run)
    return AutomationController(app, driver, ks, sched)


def _event(key: str = "A", action: KeyAction = KeyAction.DOWN, ts: float = 1.0) -> KeyboardEvent:
    return KeyboardEvent(timestamp=ts, key=key, action=action, note_id=1)


class TestAutomationMetrics:
    def test_initial(self):
        m = AutomationMetrics()
        assert m.total_cycles == 0
        assert m.emergency_stops == 0
        assert m.state_violations_prevented == 0


class TestStateGate:
    def test_initial_state_stopped(self):
        ctrl = _make_controller()
        assert ctrl._app.state == AppState.STOPPED
        assert not ctrl.is_real_input_allowed

    def test_dry_run_allowed(self):
        ctrl = _make_controller()
        assert ctrl.start_dry_run()
        assert ctrl._app.state == AppState.DRY_RUN
        assert not ctrl.is_real_input_allowed

    def test_real_mode_allowed_from_ready(self):
        ctrl = _make_controller()
        ctrl._app.transition(AppState.CALIBRATING)
        ctrl._app.transition(AppState.READY)
        assert ctrl.start_real()
        assert ctrl._app.state == AppState.RUNNING
        assert ctrl.is_real_input_allowed

    def test_real_mode_blocked_from_stopped(self):
        ctrl = _make_controller()
        assert not ctrl.start_real()
        assert ctrl._app.state == AppState.STOPPED

    def test_stop_from_running(self):
        ctrl = _make_controller()
        ctrl._app.transition(AppState.CALIBRATING)
        ctrl._app.transition(AppState.READY)
        ctrl.start_real()
        assert ctrl.stop()
        assert ctrl._app.state == AppState.READY

    def test_stop_from_stopped_is_idempotent(self):
        ctrl = _make_controller()
        assert ctrl.stop()
        assert ctrl._app.state == AppState.STOPPED


class TestEmergencyStop:
    def test_emergency_from_dry_run(self):
        ctrl = _make_controller()
        ctrl.start_dry_run()
        ctrl.emergency_stop()
        assert ctrl._app.state == AppState.EMERGENCY_STOP
        assert not ctrl.is_real_input_allowed

    def test_emergency_from_running(self):
        ctrl = _make_controller()
        ctrl._app.transition(AppState.CALIBRATING)
        ctrl._app.transition(AppState.READY)
        ctrl.start_real()
        ctrl.emergency_stop()
        assert ctrl._app.state == AppState.EMERGENCY_STOP

    def test_emergency_releases_keys(self):
        ctrl = _make_controller()
        ctrl.start_dry_run()
        ctrl._key_state.apply_event(_event("A", KeyAction.DOWN))
        ctrl._key_state.apply_event(_event("S", KeyAction.DOWN))
        assert ctrl._key_state.is_down("A")
        assert ctrl._key_state.is_down("S")

        ctrl.emergency_stop()
        assert not ctrl._key_state.is_down("A")
        assert not ctrl._key_state.is_down("S")

    def test_emergency_cancels_pending(self):
        ctrl = _make_controller()
        ctrl.start_dry_run()
        ctrl._scheduler.schedule([_event(ts=999.0)])
        assert ctrl._scheduler.pending_count > 0

        ctrl.emergency_stop()
        assert ctrl._scheduler.pending_count == 0

    def test_release_emergency(self):
        ctrl = _make_controller()
        ctrl.start_dry_run()
        ctrl.emergency_stop()
        assert ctrl.release_emergency()
        assert ctrl._app.state == AppState.STOPPED

    def test_release_emergency_wrong_state(self):
        ctrl = _make_controller()
        assert not ctrl.release_emergency()

    def test_emergency_counter(self):
        ctrl = _make_controller()
        ctrl.start_dry_run()
        ctrl.emergency_stop()
        ctrl.release_emergency()
        ctrl.start_dry_run()
        ctrl.emergency_stop()
        assert ctrl.metrics.emergency_stops == 2


class TestSafeDispatch:
    def test_dispatch_blocked_when_stopped(self):
        ctrl = _make_controller()
        result = ctrl.safe_dispatch(_event())
        assert not result
        assert ctrl.metrics.state_violations_prevented == 1

    def test_dispatch_allowed_when_running(self):
        ctrl = _make_controller()
        ctrl._app.transition(AppState.CALIBRATING)
        ctrl._app.transition(AppState.READY)
        ctrl.start_real()
        result = ctrl.safe_dispatch(_event("A", KeyAction.DOWN))
        assert result
        assert ctrl._key_state.is_down("A")
        ctrl.release_all_keys()

    def test_dispatch_blocked_after_emergency(self):
        ctrl = _make_controller()
        ctrl.start_dry_run()
        ctrl.emergency_stop()
        result = ctrl.safe_dispatch(_event())
        assert not result
        assert ctrl.metrics.state_violations_prevented >= 1

    def test_emergency_key_triggers_stop(self):
        ctrl = _make_controller()
        ctrl._app.transition(AppState.CALIBRATING)
        ctrl._app.transition(AppState.READY)
        ctrl.start_real()
        ctrl.safe_dispatch(_event(EMERGENCY_KEY, KeyAction.DOWN))
        assert ctrl._app.state == AppState.EMERGENCY_STOP

    def test_real_input_sends_to_driver(self):
        ctrl = _make_controller(dry_run=True)
        ctrl._app.transition(AppState.CALIBRATING)
        ctrl._app.transition(AppState.READY)
        ctrl.start_real()
        ctrl.safe_dispatch(_event("A", KeyAction.DOWN))
        assert ctrl.metrics.total_events_dispatched == 1
        ctrl.release_all_keys()

    def test_dry_run_does_not_send_to_driver(self):
        ctrl = _make_controller(dry_run=True)
        ctrl.start_dry_run()
        ctrl.safe_dispatch(_event("A", KeyAction.DOWN))
        assert ctrl.metrics.total_events_dispatched == 0
        assert ctrl._key_state.is_down("A")
        ctrl.release_all_keys()


class TestReleaseAllKeys:
    def test_releases_all(self):
        ctrl = _make_controller()
        ctrl._key_state.apply_event(_event("A", KeyAction.DOWN))
        ctrl._key_state.apply_event(_event("D", KeyAction.DOWN))
        released = ctrl.release_all_keys()
        assert "A" in released
        assert "D" in released
        assert not ctrl._key_state.is_down("A")
        assert not ctrl._key_state.is_down("D")

from __future__ import annotations

import pytest

from app.state import App, AppState


def test_initial_state():
    app = App()
    assert app.state == AppState.STOPPED


def test_valid_transitions():
    app = App()

    app.transition(AppState.CALIBRATING)
    assert app.state == AppState.CALIBRATING

    app.transition(AppState.READY)
    assert app.state == AppState.READY

    app.transition(AppState.DRY_RUN)
    assert app.state == AppState.DRY_RUN

    app.transition(AppState.RUNNING)
    assert app.state == AppState.RUNNING

    app.transition(AppState.STOPPED)
    assert app.state == AppState.STOPPED


def test_invalid_transition_raises():
    app = App()
    with pytest.raises(ValueError, match="Invalid transition"):
        app.transition(AppState.RUNNING)


def test_emergency_stop_from_dry_run():
    app = App()
    app.transition(AppState.CALIBRATING)
    app.transition(AppState.READY)
    app.transition(AppState.DRY_RUN)
    app.transition(AppState.EMERGENCY_STOP)
    assert app.state == AppState.EMERGENCY_STOP


def test_emergency_stop_from_running():
    app = App()
    app.transition(AppState.CALIBRATING)
    app.transition(AppState.READY)
    app.transition(AppState.RUNNING)
    app.transition(AppState.EMERGENCY_STOP)
    assert app.state == AppState.EMERGENCY_STOP


def test_emergency_stop_to_stopped():
    app = App()
    app.transition(AppState.CALIBRATING)
    app.transition(AppState.READY)
    app.transition(AppState.DRY_RUN)
    app.transition(AppState.EMERGENCY_STOP)
    app.transition(AppState.STOPPED)
    assert app.state == AppState.STOPPED


def test_error_from_any_active_state():
    for start in [AppState.STOPPED, AppState.CALIBRATING, AppState.READY,
                  AppState.DRY_RUN, AppState.RUNNING]:
        app = App()
        if start != AppState.STOPPED:
            # Navigate to start state
            if start == AppState.CALIBRATING:
                app.transition(AppState.CALIBRATING)
            elif start == AppState.READY:
                app.transition(AppState.CALIBRATING)
                app.transition(AppState.READY)
            elif start == AppState.DRY_RUN:
                app.transition(AppState.CALIBRATING)
                app.transition(AppState.READY)
                app.transition(AppState.DRY_RUN)
            elif start == AppState.RUNNING:
                app.transition(AppState.CALIBRATING)
                app.transition(AppState.READY)
                app.transition(AppState.RUNNING)
        app.transition(AppState.ERROR)
        assert app.state == AppState.ERROR


def test_can_transition():
    app = App()
    assert app.can_transition(AppState.CALIBRATING)
    assert app.can_transition(AppState.READY)
    assert not app.can_transition(AppState.RUNNING)


def test_history_recorded():
    app = App()
    app.transition(AppState.CALIBRATING)
    app.transition(AppState.READY)
    history = app.history
    assert len(history) == 2
    assert history[0] == (AppState.STOPPED, AppState.CALIBRATING)
    assert history[1] == (AppState.CALIBRATING, AppState.READY)

from __future__ import annotations

import time

from app.input.key_state import KeyboardStateMachine
from app.input.keyboard import KeyboardDriver
from app.input.scheduler import Scheduler, SchedulerMetrics
from app.rhythm.note import KeyAction, KeyboardEvent


def _event(key: str = "A", action: KeyAction = KeyAction.DOWN, ts: float = 1.0) -> KeyboardEvent:
    return KeyboardEvent(timestamp=ts, key=key, action=action, note_id=1)


class TestScheduler:
    def test_initial_state(self):
        driver = KeyboardDriver(dry_run=True)
        ks = KeyboardStateMachine()
        sched = Scheduler(driver, ks, dry_run=True)
        assert sched.pending_count == 0
        assert sched.metrics.total_scheduled == 0

    def test_schedule_events(self):
        driver = KeyboardDriver(dry_run=True)
        ks = KeyboardStateMachine()
        sched = Scheduler(driver, ks, dry_run=True)
        events = [_event(), _event("S", KeyAction.DOWN, 1.01)]
        sched.schedule(events)
        assert sched.pending_count == 2
        assert sched.metrics.total_scheduled == 2

    def test_cancel_all(self):
        driver = KeyboardDriver(dry_run=True)
        ks = KeyboardStateMachine()
        sched = Scheduler(driver, ks, dry_run=True)
        sched.schedule([_event(), _event("S", ts=1.01)])
        cancelled = sched.cancel_all()
        assert cancelled == 2
        assert sched.pending_count == 0

    def test_metrics_initial(self):
        m = SchedulerMetrics()
        assert m.total_scheduled == 0
        assert m.mean_error_ms == 0.0
        assert m.max_error_ms == 0.0

    def test_metrics_with_errors(self):
        m = SchedulerMetrics()
        m.timing_errors_ms = [1.0, 2.0, 3.0]
        assert m.mean_error_ms == 2.0
        assert m.max_error_ms == 3.0

    def test_dry_run_dispatch(self):
        driver = KeyboardDriver(dry_run=True)
        ks = KeyboardStateMachine()
        sched = Scheduler(driver, ks, dry_run=True)

        now = time.perf_counter()
        sched.schedule([_event(ts=now + 0.01)])
        sched.start()
        time.sleep(0.05)
        sched.stop()

        assert sched.metrics.total_dispatched >= 1
        assert ks.is_down("A")
        ks.release_all()

    def test_real_dispatch(self):
        driver = KeyboardDriver(dry_run=True)
        ks = KeyboardStateMachine()
        sched = Scheduler(driver, ks, dry_run=False)

        now = time.perf_counter()
        sched.schedule([_event(ts=now + 0.01)])
        sched.start()
        time.sleep(0.05)
        sched.stop()

        assert sched.metrics.total_dispatched >= 1
        assert ks.is_down("A")
        ks.release_all()

    def test_release_all(self):
        driver = KeyboardDriver(dry_run=True)
        ks = KeyboardStateMachine()
        sched = Scheduler(driver, ks, dry_run=True)

        ks.apply_event(_event())
        ks.apply_event(_event("S", KeyAction.DOWN))
        sched.release_all()
        assert not ks.is_down("A")
        assert not ks.is_down("S")

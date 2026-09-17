from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from app.input.keyboard import KeyboardDriver
from app.input.key_state import KeyboardStateMachine
from app.input.scheduler import Scheduler
from app.state import App, AppState

logger = logging.getLogger(__name__)

EMERGENCY_KEY = "F8"


@dataclass
class AutomationMetrics:
    total_cycles: int = 0
    total_events_dispatched: int = 0
    emergency_stops: int = 0
    state_violations_prevented: int = 0


class AutomationController:
    def __init__(
        self,
        app: App,
        driver: KeyboardDriver,
        key_state: KeyboardStateMachine,
        scheduler: Scheduler,
    ) -> None:
        self._app = app
        self._driver = driver
        self._key_state = key_state
        self._scheduler = scheduler
        self._metrics = AutomationMetrics()
        self._lock = threading.Lock()
        self._emergency_held = False

    @property
    def metrics(self) -> AutomationMetrics:
        return self._metrics

    @property
    def is_real_input_allowed(self) -> bool:
        return self._app.state == AppState.RUNNING

    @property
    def is_active(self) -> bool:
        return self._app.state in (AppState.DRY_RUN, AppState.RUNNING)

    def start_dry_run(self) -> bool:
        if not self._app.can_transition(AppState.DRY_RUN):
            logger.warning("Cannot start dry run from state %s", self._app.state.value)
            return False

        self._driver.dry_run = True
        self._scheduler._dry_run = True
        self._app.transition(AppState.DRY_RUN)
        self._scheduler.start()
        logger.info("Dry-run mode started")
        return True

    def start_real(self) -> bool:
        if not self._app.can_transition(AppState.RUNNING):
            logger.warning("Cannot start real mode from state %s", self._app.state.value)
            return False

        self._driver.dry_run = False
        self._scheduler._dry_run = False
        self._app.transition(AppState.RUNNING)
        self._scheduler.start()
        logger.info("Real input mode started")
        return True

    def stop(self) -> bool:
        self._scheduler.stop()
        self._scheduler.cancel_all()
        self._key_state.release_all()
        if not self._driver.dry_run:
            self._driver.release_all()

        if self._app.state in (AppState.DRY_RUN, AppState.RUNNING):
            if self._app.can_transition(AppState.READY):
                self._app.transition(AppState.READY)
            elif self._app.can_transition(AppState.STOPPED):
                self._app.transition(AppState.STOPPED)
        logger.info("Automation stopped")
        return True

    def emergency_stop(self) -> None:
        logger.warning("EMERGENCY STOP triggered")
        self._metrics.emergency_stops += 1

        self._scheduler.stop()
        self._scheduler.cancel_all()
        released = self._key_state.release_all()
        if not self._driver.dry_run:
            self._driver.release_all()

        if self._app.can_transition(AppState.EMERGENCY_STOP):
            self._app.transition(AppState.EMERGENCY_STOP)
        elif self._app.state == AppState.STOPPED:
            pass
        else:
            try:
                self._app.transition(AppState.EMERGENCY_STOP)
            except ValueError:
                self._app._state = AppState.EMERGENCY_STOP

        self._emergency_held = True
        logger.warning("Emergency stop complete. Released: %s", released)

    def release_emergency(self) -> bool:
        if self._app.state != AppState.EMERGENCY_STOP:
            return False
        self._emergency_held = False
        self._app.transition(AppState.STOPPED)
        logger.info("Emergency released, state: STOPPED")
        return True

    def safe_dispatch(self, event: "KeyboardEvent") -> bool:
        if not self.is_active:
            self._metrics.state_violations_prevented += 1
            logger.debug("Event blocked: not in active state (current: %s)", self._app.state.value)
            return False

        if self._emergency_held:
            self._metrics.state_violations_prevented += 1
            logger.debug("Event blocked: emergency hold active")
            return False

        if event.key.upper() == EMERGENCY_KEY and event.action.value == "DOWN":
            self.emergency_stop()
            return False

        accepted = self._key_state.apply_event(event)
        if not accepted:
            return False

        if self.is_real_input_allowed:
            if event.action.value == "DOWN":
                self._driver.key_down(event.key)
            elif event.action.value == "UP":
                self._driver.key_up(event.key)
            self._metrics.total_events_dispatched += 1
            return True

        return True

    def release_all_keys(self) -> list[str]:
        released = self._key_state.release_all()
        if not self._driver.dry_run:
            self._driver.release_all()
        return released

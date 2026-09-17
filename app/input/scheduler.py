from __future__ import annotations

import heapq
import logging
import random
import threading
import time
from dataclasses import dataclass, field

from app.input.key_state import KeyboardStateMachine
from app.input.keyboard import KeyboardDriver
from app.rhythm.note import KeyAction, KeyboardEvent

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _ScheduledEvent:
    timestamp: float
    event: KeyboardEvent = field(compare=False)


@dataclass
class SchedulerMetrics:
    total_scheduled: int = 0
    total_dispatched: int = 0
    total_errors: int = 0
    timing_errors_ms: list[float] = field(default_factory=list)

    @property
    def mean_error_ms(self) -> float:
        if not self.timing_errors_ms:
            return 0.0
        return sum(self.timing_errors_ms) / len(self.timing_errors_ms)

    @property
    def max_error_ms(self) -> float:
        if not self.timing_errors_ms:
            return 0.0
        return max(self.timing_errors_ms)


class Scheduler:
    def __init__(
        self,
        driver: KeyboardDriver,
        key_state: KeyboardStateMachine,
        dry_run: bool = True,
        jitter_enabled: bool = False,
        jitter_min_ms: float = 0.0,
        jitter_max_ms: float = 15.0,
    ) -> None:
        self._driver = driver
        self._key_state = key_state
        self._dry_run = dry_run
        self._jitter_enabled = jitter_enabled
        self._jitter_min_ms = jitter_min_ms
        self._jitter_max_ms = jitter_max_ms
        self._queue: list[_ScheduledEvent] = []
        self._lock = threading.Lock()
        self._running = False
        self._metrics = SchedulerMetrics()
        self._thread: threading.Thread | None = None

    @property
    def metrics(self) -> SchedulerMetrics:
        return self._metrics

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def configure_jitter(
        self,
        enabled: bool,
        min_ms: float = 0.0,
        max_ms: float = 15.0,
    ) -> None:
        self._jitter_enabled = enabled
        self._jitter_min_ms = min_ms
        self._jitter_max_ms = max_ms
        logger.info(
            "Jitter: enabled=%s, range=[%.1f, %.1f] ms",
            enabled, min_ms, max_ms,
        )

    def _apply_jitter(self, timestamp: float) -> float:
        if not self._jitter_enabled:
            return timestamp
        jitter_s = random.uniform(
            self._jitter_min_ms / 1000.0,
            self._jitter_max_ms / 1000.0,
        )
        return timestamp + jitter_s

    def schedule(self, events: list[KeyboardEvent]) -> None:
        with self._lock:
            for event in events:
                jittered_ts = self._apply_jitter(event.timestamp)
                jittered_event = KeyboardEvent(
                    timestamp=jittered_ts,
                    key=event.key,
                    action=event.action,
                    note_id=event.note_id,
                    synchronization_group=event.synchronization_group,
                )
                heapq.heappush(self._queue, _ScheduledEvent(
                    timestamp=jittered_ts,
                    event=jittered_event,
                ))
                self._metrics.total_scheduled += 1

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started (dry_run=%s)", self._dry_run)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Scheduler stopped")

    def cancel_all(self) -> int:
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def _run_loop(self) -> None:
        while self._running:
            event = self._next_event()
            if event is None:
                time.sleep(0.001)
                continue

            self._wait_until(event.timestamp)
            self._dispatch(event.event)

    def _next_event(self) -> _ScheduledEvent | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def _wait_until(self, target_time: float) -> None:
        remaining = target_time - time.perf_counter()
        if remaining > 0.01:
            time.sleep(remaining - 0.005)

        while time.perf_counter() < target_time:
            pass

    def _dispatch(self, event: KeyboardEvent) -> None:
        if event.action == KeyAction.DOWN:
            accepted = self._key_state.apply_event(event)
            if accepted and not self._dry_run:
                self._driver.key_down(event.key)
        elif event.action == KeyAction.UP:
            accepted = self._key_state.apply_event(event)
            if accepted and not self._dry_run:
                self._driver.key_up(event.key)

        actual_time = time.perf_counter()
        error_ms = (actual_time - event.timestamp) * 1000
        self._metrics.total_dispatched += 1
        self._metrics.timing_errors_ms.append(error_ms)

        if len(self._metrics.timing_errors_ms) > 1000:
            self._metrics.timing_errors_ms = self._metrics.timing_errors_ms[-500:]

    def release_all(self) -> None:
        released = self._key_state.release_all()
        if not self._dry_run:
            self._driver.release_all()
        logger.info("Emergency release: %s", released)

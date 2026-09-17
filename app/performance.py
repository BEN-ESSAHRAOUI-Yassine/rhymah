from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    capture_fps: float = 0.0
    processing_fps: float = 0.0
    frame_latency_ms: float = 0.0
    detection_latency_ms: float = 0.0
    prediction_latency_ms: float = 0.0
    scheduler_error_ms: float = 0.0
    cpu_usage_percent: float = 0.0

    def to_dict(self) -> dict:
        return {
            "capture_fps": round(self.capture_fps, 1),
            "processing_fps": round(self.processing_fps, 1),
            "frame_latency_ms": round(self.frame_latency_ms, 2),
            "detection_latency_ms": round(self.detection_latency_ms, 2),
            "prediction_latency_ms": round(self.prediction_latency_ms, 2),
            "scheduler_error_ms": round(self.scheduler_error_ms, 2),
            "cpu_usage_percent": round(self.cpu_usage_percent, 1),
        }


class PerformanceTracker:
    def __init__(self, window: int = 60) -> None:
        self._window = window
        self._capture_times: deque[float] = deque(maxlen=window)
        self._process_times: deque[float] = deque(maxlen=window)
        self._detection_times: deque[float] = deque(maxlen=window)
        self._prediction_times: deque[float] = deque(maxlen=window)
        self._scheduler_errors: deque[float] = deque(maxlen=window)

    def tick_capture(self) -> None:
        self._capture_times.append(time.perf_counter())

    def tick_process(self, duration_ms: float) -> None:
        self._process_times.append(duration_ms)

    def tick_detection(self, duration_ms: float) -> None:
        self._detection_times.append(duration_ms)

    def tick_prediction(self, duration_ms: float) -> None:
        self._prediction_times.append(duration_ms)

    def record_scheduler_error(self, error_ms: float) -> None:
        self._scheduler_errors.append(error_ms)

    @property
    def capture_fps(self) -> float:
        if len(self._capture_times) < 2:
            return 0.0
        elapsed = self._capture_times[-1] - self._capture_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._capture_times) - 1) / elapsed

    @property
    def processing_fps(self) -> float:
        if not self._process_times:
            return 0.0
        avg_ms = sum(self._process_times) / len(self._process_times)
        if avg_ms <= 0:
            return 0.0
        return 1000.0 / avg_ms

    @property
    def mean_frame_latency_ms(self) -> float:
        if not self._process_times:
            return 0.0
        return sum(self._process_times) / len(self._process_times)

    @property
    def mean_detection_latency_ms(self) -> float:
        if not self._detection_times:
            return 0.0
        return sum(self._detection_times) / len(self._detection_times)

    @property
    def mean_prediction_latency_ms(self) -> float:
        if not self._prediction_times:
            return 0.0
        return sum(self._prediction_times) / len(self._prediction_times)

    @property
    def mean_scheduler_error_ms(self) -> float:
        if not self._scheduler_errors:
            return 0.0
        return sum(self._scheduler_errors) / len(self._scheduler_errors)

    @property
    def max_scheduler_error_ms(self) -> float:
        if not self._scheduler_errors:
            return 0.0
        return max(self._scheduler_errors)

    def snapshot(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            capture_fps=self.capture_fps,
            processing_fps=self.processing_fps,
            frame_latency_ms=self.mean_frame_latency_ms,
            detection_latency_ms=self.mean_detection_latency_ms,
            prediction_latency_ms=self.mean_prediction_latency_ms,
            scheduler_error_ms=self.mean_scheduler_error_ms,
        )

    def reset(self) -> None:
        self._capture_times.clear()
        self._process_times.clear()
        self._detection_times.clear()
        self._prediction_times.clear()
        self._scheduler_errors.clear()

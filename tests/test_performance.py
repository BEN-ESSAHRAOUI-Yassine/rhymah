from __future__ import annotations

import time

from app.performance import PerformanceMetrics, PerformanceTracker


class TestPerformanceMetrics:
    def test_to_dict(self):
        m = PerformanceMetrics(
            capture_fps=60.0,
            processing_fps=55.0,
            frame_latency_ms=18.0,
            detection_latency_ms=5.0,
            prediction_latency_ms=1.5,
            scheduler_error_ms=0.8,
            cpu_usage_percent=25.0,
        )
        d = m.to_dict()
        assert d["capture_fps"] == 60.0
        assert d["scheduler_error_ms"] == 0.8

    def test_defaults(self):
        m = PerformanceMetrics()
        assert m.capture_fps == 0.0
        assert m.cpu_usage_percent == 0.0


class TestPerformanceTracker:
    def test_initial_state(self):
        t = PerformanceTracker()
        assert t.capture_fps == 0.0
        assert t.processing_fps == 0.0

    def test_capture_fps(self):
        t = PerformanceTracker(window=10)
        start = time.perf_counter()
        for _ in range(10):
            t.tick_capture()
            time.sleep(0.01)
        fps = t.capture_fps
        assert fps > 0.0
        assert fps < 200.0

    def test_processing_fps(self):
        t = PerformanceTracker()
        t.tick_process(16.0)
        t.tick_process(18.0)
        fps = t.processing_fps
        assert fps > 0.0

    def test_detection_latency(self):
        t = PerformanceTracker()
        t.tick_detection(5.0)
        t.tick_detection(7.0)
        assert t.mean_detection_latency_ms == 6.0

    def test_prediction_latency(self):
        t = PerformanceTracker()
        t.tick_prediction(1.0)
        t.tick_prediction(2.0)
        assert t.mean_prediction_latency_ms == 1.5

    def test_scheduler_error(self):
        t = PerformanceTracker()
        t.record_scheduler_error(0.5)
        t.record_scheduler_error(1.5)
        assert t.mean_scheduler_error_ms == 1.0
        assert t.max_scheduler_error_ms == 1.5

    def test_snapshot(self):
        t = PerformanceTracker()
        t.tick_process(16.0)
        snap = t.snapshot()
        assert isinstance(snap, PerformanceMetrics)
        assert snap.frame_latency_ms == 16.0

    def test_reset(self):
        t = PerformanceTracker()
        t.tick_process(16.0)
        t.tick_detection(5.0)
        t.reset()
        assert t.capture_fps == 0.0
        assert t.mean_detection_latency_ms == 0.0

    def test_window_enforced(self):
        t = PerformanceTracker(window=3)
        for i in range(5):
            t.tick_process(float(i))
        assert len(t._process_times) == 3

    def test_empty_tracker(self):
        t = PerformanceTracker()
        assert t.mean_frame_latency_ms == 0.0
        assert t.mean_detection_latency_ms == 0.0
        assert t.max_scheduler_error_ms == 0.0

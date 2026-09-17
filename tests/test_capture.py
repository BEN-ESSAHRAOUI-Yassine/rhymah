from __future__ import annotations

import time

import numpy as np

from app.capture.frame_buffer import FrameBuffer
from app.capture.screen_capture import CaptureRegion, _FPSCounter, ScreenCapture
from app.rhythm.note import Frame


def _make_frame(seq: int = 1, ts: float | None = None) -> Frame:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    return Frame(image=img, timestamp=ts or time.perf_counter(), sequence_number=seq)


class TestFrameBuffer:
    def test_empty_buffer(self):
        buf = FrameBuffer(capacity=3)
        assert len(buf) == 0
        assert buf.latest() is None
        assert buf.pop() is None

    def test_push_and_latest(self):
        buf = FrameBuffer(capacity=3)
        f1 = _make_frame(1)
        f2 = _make_frame(2)
        buf.push(f1)
        assert len(buf) == 1
        assert buf.latest() is f1

        buf.push(f2)
        assert len(buf) == 2
        assert buf.latest() is f2

    def test_pop_returns_last(self):
        buf = FrameBuffer(capacity=3)
        f1 = _make_frame(1)
        f2 = _make_frame(2)
        buf.push(f1)
        buf.push(f2)
        popped = buf.pop()
        assert popped is f2
        assert len(buf) == 1
        assert buf.latest() is f1

    def test_capacity_enforced(self):
        buf = FrameBuffer(capacity=2)
        f1 = _make_frame(1)
        f2 = _make_frame(2)
        f3 = _make_frame(3)
        buf.push(f1)
        buf.push(f2)
        buf.push(f3)
        assert len(buf) == 2
        assert buf.latest() is f3

    def test_clear(self):
        buf = FrameBuffer(capacity=3)
        buf.push(_make_frame(1))
        buf.push(_make_frame(2))
        buf.clear()
        assert len(buf) == 0
        assert buf.latest() is None

    def test_bool(self):
        buf = FrameBuffer(capacity=3)
        assert not buf
        buf.push(_make_frame(1))
        assert buf


class TestCaptureRegion:
    def test_to_mss_dict(self):
        r = CaptureRegion(left=100, top=200, width=640, height=480)
        d = r.to_mss_dict()
        assert d == {"left": 100, "top": 200, "width": 640, "height": 480}


class TestFPSCounter:
    def test_initial_fps_zero(self):
        c = _FPSCounter()
        assert c.fps == 0.0

    def test_fps_after_ticks(self):
        c = _FPSCounter(window=10)
        start = time.perf_counter()
        for _ in range(10):
            c.tick()
            time.sleep(0.01)
        fps = c.fps
        assert fps > 0.0
        assert fps < 200.0


class TestScreenCapture:
    def test_no_region_returns_none(self):
        cap = ScreenCapture(region=None)
        result = cap.grab()
        assert result is None

    def test_region_setter(self):
        cap = ScreenCapture(region=None)
        r = CaptureRegion(left=0, top=0, width=100, height=100)
        cap.region = r
        assert cap.region is r

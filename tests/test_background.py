from __future__ import annotations

import numpy as np

from app.calibration.profile import ColorSample
from app.vision.background import BackgroundModel


def _make_frame(shape: tuple[int, int] = (100, 100), color: int = 128) -> np.ndarray:
    frame = np.full((shape[0], shape[1], 3), color, dtype=np.uint8)
    return frame


def _make_static_scene() -> np.ndarray:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:60, 40:60] = [100, 50, 200]
    return frame


def _make_moving_object(scene: np.ndarray, x: int, y: int) -> np.ndarray:
    frame = scene.copy()
    frame[y:y + 15, x:x + 15] = [200, 100, 50]
    return frame


class TestBackgroundModel:
    def test_initial_state(self):
        bm = BackgroundModel(history_size=10)
        assert not bm.ready
        assert len(bm._frames) == 0

    def test_not_ready_until_history_full(self):
        bm = BackgroundModel(history_size=5)
        for _ in range(4):
            ready = bm.add_frame(_make_frame())
            assert not ready
        assert not bm.ready

    def test_ready_after_history_full(self):
        bm = BackgroundModel(history_size=5)
        for _ in range(5):
            bm.add_frame(_make_frame())
        assert bm.ready

    def test_reset(self):
        bm = BackgroundModel(history_size=5)
        for _ in range(5):
            bm.add_frame(_make_frame())
        assert bm.ready
        bm.reset()
        assert not bm.ready
        assert len(bm._frames) == 0

    def test_motion_mask_when_not_ready(self):
        bm = BackgroundModel(history_size=5)
        mask = bm.get_motion_mask(_make_frame())
        assert mask.shape == (100, 100)
        assert np.all(mask == 255)

    def test_motion_mask_when_ready(self):
        bm = BackgroundModel(history_size=10)
        static = _make_static_scene()
        for _ in range(10):
            bm.add_frame(static)

        frame_with_motion = _make_moving_object(static, 70, 70)
        mask = bm.get_motion_mask(frame_with_motion)

        assert mask.shape == (100, 100)
        assert mask[75, 75] == 255
        assert mask[10, 10] == 0

    def test_filter_frame_no_color(self):
        bm = BackgroundModel(history_size=10)
        static = _make_static_scene()
        for _ in range(10):
            bm.add_frame(static)

        frame_with_motion = _make_moving_object(static, 70, 70)
        result = bm.filter_frame(frame_with_motion)
        assert result.shape == (100, 100)

    def test_filter_frame_with_color(self):
        bm = BackgroundModel(history_size=10)
        static = _make_static_scene()
        for _ in range(10):
            bm.add_frame(static)

        color = ColorSample(h_min=0, h_max=180, s_min=0, s_max=255, v_min=0, v_max=255)
        result = bm.filter_frame(static, color_sample=color)
        assert result.shape == (100, 100)

    def test_history_size_property(self):
        bm = BackgroundModel(history_size=20)
        assert bm.history_size == 20

    def test_pops_old_frames(self):
        bm = BackgroundModel(history_size=5)
        for i in range(8):
            bm.add_frame(np.full((10, 10, 3), i * 10, dtype=np.uint8))
        assert len(bm._frames) == 5

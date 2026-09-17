from __future__ import annotations

import cv2
import numpy as np

from app.calibration.profile import ColorSample
from app.vision.color_detector import (
    PRESET_COLORS,
    ColorDetector,
    ColorDetectorConfig,
    ColorRange,
)


def _make_frame_with_color(
    shape: tuple[int, int] = (200, 200, 3),
    color_bgr: tuple[int, int, int] = (255, 0, 0),
    region: tuple[int, int, int, int] = (50, 50, 150, 150),
) -> np.ndarray:
    frame = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = region
    frame[y1:y2, x1:x2] = color_bgr
    return frame


class TestColorRange:
    def test_creation(self):
        cr = ColorRange(
            name="test",
            h_min=0, h_max=10,
            s_min=50, s_max=200,
            v_min=50, v_max=200,
            priority=1,
        )
        assert cr.name == "test"
        assert cr.h_min == 0
        assert cr.priority == 1


class TestPresets:
    def test_purple_exists(self):
        assert "purple" in PRESET_COLORS
        p = PRESET_COLORS["purple"]
        assert p.h_min == 120
        assert p.h_max == 160

    def test_orange_exists(self):
        assert "orange" in PRESET_COLORS
        o = PRESET_COLORS["orange"]
        assert o.h_min == 5
        assert o.h_max == 25

    def test_yellow_exists(self):
        assert "yellow" in PRESET_COLORS
        y = PRESET_COLORS["yellow"]
        assert y.h_min == 25
        assert y.h_max == 40


class TestColorDetectorConfig:
    def test_default_colors(self):
        cfg = ColorDetectorConfig()
        assert cfg.colors == ["purple"]
        assert cfg.merge_masks is True

    def test_custom_colors(self):
        cfg = ColorDetectorConfig(colors=["purple", "orange", "yellow"])
        assert len(cfg.colors) == 3


class TestColorDetector:
    def test_purple_detection(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple"]))
        bgr_purple = (180, 50, 150)
        frame = _make_frame_with_color(color_bgr=bgr_purple)
        mask = detector.detect(frame)
        assert mask is not None
        assert mask.shape == frame.shape[:2]
        non_zero = cv2.countNonZero(mask)
        assert non_zero > 0

    def test_orange_detection(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["orange"]))
        hsv_orange = np.uint8([[[15, 200, 200]]])
        bgr_orange = cv2.cvtColor(hsv_orange, cv2.COLOR_HSV2BGR)[0][0]
        bgr_tuple = (int(bgr_orange[0]), int(bgr_orange[1]), int(bgr_orange[2]))
        frame = _make_frame_with_color(color_bgr=bgr_tuple)
        mask = detector.detect(frame)
        assert mask is not None
        non_zero = cv2.countNonZero(mask)
        assert non_zero > 0

    def test_yellow_detection(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["yellow"]))
        hsv_yellow = np.uint8([[[30, 200, 200]]])
        bgr_yellow = cv2.cvtColor(hsv_yellow, cv2.COLOR_HSV2BGR)[0][0]
        bgr_tuple = (int(bgr_yellow[0]), int(bgr_yellow[1]), int(bgr_yellow[2]))
        frame = _make_frame_with_color(color_bgr=bgr_tuple)
        mask = detector.detect(frame)
        assert mask is not None
        non_zero = cv2.countNonZero(mask)
        assert non_zero > 0

    def test_multi_color_detection(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple", "orange"]))
        assert len(detector.active_colors) == 2

    def test_empty_frame(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple"]))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mask = detector.detect(frame)
        assert cv2.countNonZero(mask) == 0

    def test_custom_color(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple"]))
        sample = ColorSample(h_min=0, h_max=10, s_min=100, s_max=255, v_min=100, v_max=255)
        detector.set_custom_color(sample)
        hsv_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        hsv_frame[:, :, 0] = 5
        hsv_frame[:, :, 1] = 200
        hsv_frame[:, :, 2] = 200
        bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
        mask = detector.detect(bgr_frame)
        assert cv2.countNonZero(mask) > 0

    def test_clear_custom_color(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple"]))
        sample = ColorSample(h_min=0, h_max=10, s_min=100, s_max=255, v_min=100, v_max=255)
        detector.set_custom_color(sample)
        detector.clear_custom_color()
        assert detector._custom_color is None

    def test_add_custom_color(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple"]))
        custom = ColorRange(
            name="pink",
            h_min=140, h_max=170,
            s_min=50, s_max=255,
            v_min=50, v_max=255,
        )
        detector.add_color("pink", custom)
        assert "pink" in detector.active_colors

    def test_remove_color(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple", "orange"]))
        detector.remove_color("orange")
        assert "orange" not in detector.active_colors

    def test_detect_per_color(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple", "orange"]))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        masks = detector.detect_per_color(frame)
        assert "purple" in masks
        assert "orange" in masks

    def test_get_color_ranges(self):
        detector = ColorDetector(ColorDetectorConfig(colors=["purple"]))
        ranges = detector.get_color_ranges()
        assert len(ranges) == 1
        assert ranges[0].name == "purple"

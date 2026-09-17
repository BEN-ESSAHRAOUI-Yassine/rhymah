from __future__ import annotations

import cv2
import numpy as np

from app.calibration.profile import LaneTrajectory, Point
from app.rhythm.note import BarOrientation
from app.vision.note_classifier import LongNoteDetector
from app.vision.note_detector import Candidate, DetectionConfig, ShortNoteDetector


def _make_horizontal_bar(
    shape: tuple[int, int] = (300, 300),
    center: tuple[int, int] = (150, 150),
    length: int = 120,
    width: int = 20,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    rect = ((float(center[0]), float(center[1])), (float(length), float(width)), 0.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
    return mask


def _make_vertical_bar(
    shape: tuple[int, int] = (300, 300),
    center: tuple[int, int] = (150, 150),
    length: int = 120,
    width: int = 20,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    rect = ((float(center[0]), float(center[1])), (float(width), float(length)), 0.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
    return mask


def _make_diagonal_bar(
    shape: tuple[int, int] = (300, 300),
    center: tuple[int, int] = (150, 150),
    length: int = 120,
    width: int = 20,
    angle: float = 45.0,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    rect = ((float(center[0]), float(center[1])), (float(width), float(length)), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
    return mask


def _make_circular_head(
    shape: tuple[int, int] = (300, 300),
    center: tuple[int, int] = (150, 80),
    radius: int = 15,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def _detect_heads(mask: np.ndarray) -> list[Candidate]:
    det = ShortNoteDetector(DetectionConfig(min_area=50, max_area=5000, min_circularity=0.3))
    return det.detect(mask)


def _trajectories() -> dict[str, LaneTrajectory]:
    return {
        "A": LaneTrajectory(points=[Point(50, 150)]),
        "S": LaneTrajectory(points=[Point(100, 150)]),
        "D": LaneTrajectory(points=[Point(150, 150)]),
        "J": LaneTrajectory(points=[Point(200, 150)]),
        "K": LaneTrajectory(points=[Point(250, 150)]),
        "L": LaneTrajectory(points=[Point(300, 150)]),
    }


class TestBarOrientationClassification:
    def test_horizontal_bar(self):
        mask = _make_horizontal_bar(center=(150, 150), length=120, width=20)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        bars = det.detect_bars(mask, candidates=[])
        assert len(bars) >= 1
        assert bars[0].orientation == BarOrientation.HORIZONTAL

    def test_vertical_bar(self):
        mask = _make_vertical_bar(center=(150, 150), length=120, width=20)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        bars = det.detect_bars(mask, candidates=[])
        assert len(bars) >= 1
        assert bars[0].orientation == BarOrientation.VERTICAL

    def test_diagonal_bar(self):
        mask = _make_diagonal_bar(center=(150, 150), length=120, width=20, angle=45.0)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        bars = det.detect_bars(mask, candidates=[])
        assert len(bars) >= 1
        assert bars[0].orientation == BarOrientation.DIAGONAL


class TestHorizontalBarDetection:
    def test_detects_horizontal_bar(self):
        mask = _make_horizontal_bar(center=(150, 150), length=120, width=20)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        bars = det.detect_bars(mask, candidates=[])
        assert len(bars) >= 1
        assert bars[0].length > bars[0].width

    def test_horizontal_bar_confidence(self):
        mask = _make_horizontal_bar(center=(150, 150), length=120, width=20)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        bars = det.detect_bars(mask, candidates=[])
        assert len(bars) >= 1
        assert 0.0 <= bars[0].confidence <= 1.0

    def test_horizontal_bar_center(self):
        mask = _make_horizontal_bar(center=(150, 150), length=120, width=20)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        bars = det.detect_bars(mask, candidates=[])
        assert len(bars) >= 1
        cx, cy = bars[0].center
        assert 140 <= cx <= 160
        assert 140 <= cy <= 160

    def test_rejects_small_bar(self):
        mask = _make_horizontal_bar(center=(150, 150), length=10, width=5)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        bars = det.detect_bars(mask, candidates=[])
        assert len(bars) == 0

    def test_empty_mask_no_bars(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        det = LongNoteDetector()
        bars = det.detect_bars(mask, candidates=[])
        assert bars == []


class TestBarLaneAssignment:
    def test_bar_assigned_to_lane(self):
        mask = _make_horizontal_bar(center=(100, 150), length=120, width=20)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        trajs = _trajectories()
        bars = det.detect_bars(mask, candidates=[], trajectories=trajs)
        assert len(bars) >= 1
        assert bars[0].lane in ["A", "S", "D", "J", "K", "L"]

    def test_bar_confidence_with_lane(self):
        mask = _make_horizontal_bar(center=(100, 150), length=120, width=20)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        trajs = _trajectories()
        bars_no_lane = det.detect_bars(mask, candidates=[])
        bars_with_lane = det.detect_bars(mask, candidates=[], trajectories=trajs)
        assert len(bars_no_lane) >= 1
        assert len(bars_with_lane) >= 1
        assert bars_with_lane[0].confidence <= bars_no_lane[0].confidence

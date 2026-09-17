from __future__ import annotations

import numpy as np

from app.calibration.profile import LaneTrajectory, Point
from app.rhythm.note import NoteType
from app.vision.note_detector import Candidate, DetectionConfig, ShortNoteDetector


def _make_mask_with_circle(
    shape: tuple[int, int] = (200, 200),
    center: tuple[int, int] = (100, 100),
    radius: int = 20,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2_circle(mask, center, radius, 255, -1)
    return mask


def cv2_circle(mask: np.ndarray, center: tuple[int, int], radius: int, color: int, thickness: int) -> None:
    import cv2
    cv2.circle(mask, center, radius, color, thickness)


def _make_mask_with_rectangle(
    shape: tuple[int, int] = (200, 200),
    top_left: tuple[int, int] = (80, 40),
    bottom_right: tuple[int, int] = (120, 160),
) -> np.ndarray:
    import cv2
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    return mask


def _trajectories() -> dict[str, LaneTrajectory]:
    return {
        "A": LaneTrajectory(points=[Point(50, 100)]),
        "S": LaneTrajectory(points=[Point(80, 100)]),
        "D": LaneTrajectory(points=[Point(110, 100)]),
        "J": LaneTrajectory(points=[Point(140, 100)]),
        "K": LaneTrajectory(points=[Point(170, 100)]),
        "L": LaneTrajectory(points=[Point(200, 100)]),
    }


class TestCandidate:
    def test_creation(self):
        import cv2
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
        c = Candidate(
            contour=contour,
            center=(5.0, 5.0),
            area=100.0,
            perimeter=40.0,
            circularity=0.785,
            radius=7.07,
            confidence=0.8,
            lane="A",
            note_type=NoteType.SHORT,
        )
        assert c.lane == "A"
        assert c.note_type == NoteType.SHORT
        assert c.confidence == 0.8


class TestDetectionConfig:
    def test_defaults(self):
        cfg = DetectionConfig()
        assert cfg.min_area == 100.0
        assert cfg.max_area == 10000.0
        assert cfg.min_circularity == 0.4


class TestShortNoteDetector:
    def test_no_candidates_on_empty_mask(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        det = ShortNoteDetector()
        results = det.detect(mask)
        assert results == []

    def test_detects_circular_candidate(self):
        mask = _make_mask_with_circle(center=(100, 100), radius=20)
        det = ShortNoteDetector(DetectionConfig(min_area=50, max_area=5000))
        results = det.detect(mask)
        assert len(results) >= 1
        c = results[0]
        assert c.area > 0
        assert c.circularity > 0
        assert c.confidence > 0

    def test_rejects_too_small(self):
        mask = _make_mask_with_circle(center=(100, 100), radius=3)
        det = ShortNoteDetector(DetectionConfig(min_area=100))
        results = det.detect(mask)
        assert len(results) == 0

    def test_rejects_too_large(self):
        mask = _make_mask_with_circle(center=(100, 100), radius=80)
        det = ShortNoteDetector(DetectionConfig(max_area=100))
        results = det.detect(mask)
        assert len(results) == 0

    def test_rejects_non_circular(self):
        mask = _make_mask_with_rectangle(top_left=(10, 10), bottom_right=(190, 30))
        det = ShortNoteDetector(DetectionConfig(min_area=10, min_circularity=0.8))
        results = det.detect(mask)
        assert len(results) == 0

    def test_lane_assignment(self):
        mask = _make_mask_with_circle(center=(110, 100), radius=20)
        det = ShortNoteDetector(DetectionConfig(min_area=50, max_area=5000))
        trajs = _trajectories()
        results = det.detect(mask, trajectories=trajs)
        assert len(results) >= 1
        assert results[0].lane in ["A", "S", "D", "J", "K", "L"]

    def test_confidence_range(self):
        mask = _make_mask_with_circle(center=(100, 100), radius=20)
        det = ShortNoteDetector(DetectionConfig(min_area=50, max_area=5000))
        results = det.detect(mask)
        for c in results:
            assert 0.0 <= c.confidence <= 1.0

    def test_multiple_candidates(self):
        import cv2
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (50, 150), 20, 255, -1)
        cv2.circle(mask, (150, 150), 20, 255, -1)
        cv2.circle(mask, (250, 150), 20, 255, -1)

        det = ShortNoteDetector(DetectionConfig(min_area=50, max_area=5000))
        results = det.detect(mask)
        assert len(results) == 3

    def test_custom_config(self):
        cfg = DetectionConfig(min_area=10, max_area=100, min_circularity=0.3)
        det = ShortNoteDetector(cfg)
        assert det.config.min_area == 10

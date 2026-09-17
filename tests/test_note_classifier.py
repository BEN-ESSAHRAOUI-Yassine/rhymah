from __future__ import annotations

import cv2
import numpy as np

from app.calibration.profile import LaneTrajectory, Point
from app.rhythm.note import NoteType
from app.vision.note_classifier import LongNoteDetector, LongNoteCandidate
from app.vision.note_detector import Candidate, DetectionConfig, ShortNoteDetector


def _make_circular_head(
    shape: tuple[int, int] = (300, 300),
    center: tuple[int, int] = (100, 150),
    radius: int = 15,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def _make_elongated_body(
    shape: tuple[int, int] = (300, 300),
    center: tuple[int, int] = (100, 150),
    length: int = 120,
    width: int = 20,
    angle: float = 90.0,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    rect = ((float(center[0]), float(center[1])), (float(width), float(length)), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
    return mask


def _make_long_note_mask(
    head_center: tuple[int, int] = (100, 100),
    body_center: tuple[int, int] = (100, 180),
    head_radius: int = 15,
    body_length: int = 100,
    body_width: int = 20,
) -> np.ndarray:
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, head_center, head_radius, 255, -1)
    rect = ((float(body_center[0]), float(body_center[1])), (float(body_width), float(body_length)), 0.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
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


class TestLongNoteCandidate:
    def test_creation(self):
        import cv2
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
        head = Candidate(
            contour=contour,
            center=(5.0, 5.0),
            area=100.0,
            perimeter=40.0,
            circularity=0.785,
            radius=7.07,
            confidence=0.8,
            lane="S",
            note_type=NoteType.LONG,
        )
        note = LongNoteCandidate(
            head=head,
            body_length=100.0,
            body_width=20.0,
            aspect_ratio=5.0,
            tail_position=(100.0, 200.0),
            confidence=0.85,
            lane="S",
        )
        assert note.head is head
        assert note.body_length == 100.0
        assert note.lane == "S"


class TestLongNoteDetector:
    def test_no_candidates_on_empty_mask(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        det = LongNoteDetector()
        results = det.detect(mask, candidates=[])
        assert results == []

    def test_detects_elongated_shape(self):
        mask = _make_elongated_body(center=(100, 150), length=120, width=20)
        heads = _detect_heads(mask)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        results = det.detect(mask, candidates=heads)
        assert len(results) >= 1
        assert results[0].body_length > 0

    def test_rejects_square_shape(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(mask, (80, 130), (120, 170), 255, -1)
        heads = _detect_heads(mask)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        results = det.detect(mask, candidates=heads)
        assert len(results) == 0

    def test_long_note_with_head(self):
        mask = _make_long_note_mask(
            head_center=(100, 80),
            body_center=(100, 160),
            head_radius=15,
            body_length=100,
            body_width=20,
        )
        heads = _detect_heads(mask)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        results = det.detect(mask, candidates=heads)
        assert len(results) >= 1
        note = results[0]
        assert note.body_length > 50
        assert note.confidence > 0

    def test_lane_assignment(self):
        mask = _make_long_note_mask(
            head_center=(250, 80),
            body_center=(250, 160),
            head_radius=15,
            body_length=100,
            body_width=20,
        )
        heads = _detect_heads(mask)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        trajs = _trajectories()
        results = det.detect(mask, candidates=heads, trajectories=trajs)
        assert len(results) >= 1
        assert results[0].lane in ["A", "S", "D", "J", "K", "L"]

    def test_confidence_range(self):
        mask = _make_long_note_mask(head_radius=15, body_length=100, body_width=20)
        heads = _detect_heads(mask)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        results = det.detect(mask, candidates=heads)
        for note in results:
            assert 0.0 <= note.confidence <= 1.0

    def test_tail_position(self):
        mask = _make_long_note_mask(
            head_center=(100, 60),
            body_center=(100, 140),
            body_length=100,
            body_width=20,
        )
        heads = _detect_heads(mask)
        det = LongNoteDetector(DetectionConfig(min_area=50))
        results = det.detect(mask, candidates=heads)
        assert len(results) >= 1
        tail = results[0].tail_position
        assert isinstance(tail[0], float)
        assert isinstance(tail[1], float)

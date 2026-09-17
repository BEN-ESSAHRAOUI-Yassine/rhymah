from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.calibration.calibration import assign_lane_by_trajectory
from app.calibration.profile import LaneTrajectory, Point
from app.rhythm.note import NoteType
from app.vision.note_detector import Candidate, DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class LongNoteCandidate:
    head: Candidate
    body_contour: np.ndarray | None = None
    body_center: tuple[float, float] = (0.0, 0.0)
    body_length: float = 0.0
    body_width: float = 0.0
    aspect_ratio: float = 0.0
    tail_position: tuple[float, float] = (0.0, 0.0)
    orientation: float = 0.0
    confidence: float = 0.0
    lane: str = ""


class LongNoteDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self._config = config or DetectionConfig()
        self._min_aspect_ratio = 2.0
        self._min_body_length = 30.0
        self._max_body_width = 80.0
        self._head_body_connect_dist = 40.0

    def detect(
        self,
        mask: np.ndarray,
        candidates: list[Candidate],
        trajectories: dict[str, LaneTrajectory] | None = None,
    ) -> list[LongNoteCandidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        elongated = self._filter_elongated(contours)
        heads = {self._contour_key(c.contour): c for c in candidates}

        long_notes: list[LongNoteCandidate] = []

        for body_contour in elongated:
            result = self._analyze_body(body_contour, heads)
            if result is None:
                continue

            if trajectories:
                self._assign_lane(result, trajectories)

            long_notes.append(result)

        logger.debug("Long-note candidates detected: %d", len(long_notes))
        return long_notes

    def _filter_elongated(self, contours: list[np.ndarray]) -> list[np.ndarray]:
        result = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._config.min_area:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect

            if w == 0 or h == 0:
                continue

            aspect = max(w, h) / min(w, h)
            length = max(w, h)
            width = min(w, h)

            if aspect >= self._min_aspect_ratio and length >= self._min_body_length:
                result.append(contour)

        return result

    def _analyze_body(
        self,
        body_contour: np.ndarray,
        heads: dict[str, Candidate],
    ) -> LongNoteCandidate | None:
        rect = cv2.minAreaRect(body_contour)
        (cx, cy), (w, h), angle = rect

        length = max(w, h)
        width = min(w, h)
        aspect = length / width if width > 0 else 0

        head = self._find_associated_head((cx, cy), body_contour, heads)
        if head is None:
            return None

        tail = self._estimate_tail((cx, cy), length, angle)

        confidence = self._compute_confidence(
            aspect_ratio=aspect,
            body_length=length,
            body_width=width,
            head_confidence=head.confidence,
        )

        return LongNoteCandidate(
            head=head,
            body_contour=body_contour,
            body_center=(float(cx), float(cy)),
            body_length=length,
            body_width=width,
            aspect_ratio=aspect,
            tail_position=tail,
            orientation=angle,
            confidence=confidence,
        )

    def _find_associated_head(
        self,
        body_center: tuple[float, float],
        body_contour: np.ndarray,
        heads: dict[str, Candidate],
    ) -> Candidate | None:
        if not heads:
            return None

        body_rect = cv2.boundingRect(body_contour)
        bx, by, bw, bh = body_rect

        best_head: Candidate | None = None
        best_dist = float("inf")

        for key, head in heads.items():
            hx, hy = head.center

            inside_body = bx <= hx <= bx + bw and by <= hy <= by + bh
            if inside_body:
                dist = 0.0
            else:
                dist = math.dist(head.center, body_center)

            if dist < best_dist and dist < self._head_body_connect_dist:
                best_dist = dist
                best_head = head

        return best_head

    def _estimate_tail(
        self,
        center: tuple[float, float],
        length: float,
        angle: float,
    ) -> tuple[float, float]:
        rad = math.radians(angle)
        half_len = length / 2.0
        tx = center[0] + half_len * math.cos(rad)
        ty = center[1] + half_len * math.sin(rad)
        return (float(tx), float(ty))

    def _compute_confidence(
        self,
        aspect_ratio: float,
        body_length: float,
        body_width: float,
        head_confidence: float,
    ) -> float:
        ar_score = min(1.0, (aspect_ratio - 1.0) / 4.0)
        length_score = min(1.0, body_length / 100.0)
        width_penalty = max(0.0, 1.0 - body_width / self._max_body_width)

        score = 0.3 * ar_score + 0.3 * length_score + 0.2 * width_penalty + 0.2 * head_confidence
        return round(min(1.0, max(0.0, score)), 3)

    def _assign_lane(
        self,
        note: LongNoteCandidate,
        trajectories: dict[str, LaneTrajectory],
    ) -> None:
        point = Point(x=note.head.center[0], y=note.head.center[1])
        lane, conf = assign_lane_by_trajectory(point, trajectories)
        note.lane = lane
        note.confidence = round(note.confidence * conf, 3)

    @staticmethod
    def _contour_key(contour: np.ndarray) -> str:
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return "0_0"
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return f"{cx}_{cy}"

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.calibration.calibration import assign_lane_by_trajectory
from app.calibration.profile import ColorSample, LaneTrajectory, Point
from app.rhythm.note import NoteType

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    contour: np.ndarray
    center: tuple[float, float]
    area: float
    perimeter: float
    circularity: float
    radius: float
    confidence: float = 0.0
    lane: str = ""
    note_type: NoteType = NoteType.SHORT


@dataclass
class DetectionConfig:
    min_area: float = 100.0
    max_area: float = 10000.0
    min_circularity: float = 0.4
    min_convexity: float = 0.5
    min_inertia: float = 0.2


class ShortNoteDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self._config = config or DetectionConfig()

    @property
    def config(self) -> DetectionConfig:
        return self._config

    def detect(
        self,
        mask: np.ndarray,
        trajectories: dict[str, LaneTrajectory] | None = None,
    ) -> list[Candidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[Candidate] = []

        for contour in contours:
            candidate = self._analyze_contour(contour)
            if candidate is None:
                continue

            if trajectories:
                self._assign_lane(candidate, trajectories)

            candidates.append(candidate)

        logger.debug("Short-note candidates detected: %d", len(candidates))
        return candidates

    def _analyze_contour(self, contour: np.ndarray) -> Candidate | None:
        area = cv2.contourArea(contour)
        if area < self._config.min_area or area > self._config.max_area:
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < self._config.min_circularity:
            return None

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        if convexity < self._config.min_convexity:
            return None

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        center = (float(cx), float(cy))

        confidence = self._compute_confidence(
            area=area,
            circularity=circularity,
            convexity=convexity,
            perimeter=perimeter,
        )

        return Candidate(
            contour=contour,
            center=center,
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            radius=radius,
            confidence=confidence,
        )

    def _compute_confidence(
        self,
        area: float,
        circularity: float,
        convexity: float,
        perimeter: float,
    ) -> float:
        area_norm = min(1.0, area / 1000.0)
        circ_score = min(1.0, circularity)
        conv_score = min(1.0, convexity)

        score = 0.3 * area_norm + 0.4 * circ_score + 0.3 * conv_score
        return round(min(1.0, max(0.0, score)), 3)

    def _assign_lane(
        self,
        candidate: Candidate,
        trajectories: dict[str, LaneTrajectory],
    ) -> None:
        point = Point(x=candidate.center[0], y=candidate.center[1])
        lane, conf = assign_lane_by_trajectory(point, trajectories)
        candidate.lane = lane
        candidate.confidence = round(candidate.confidence * conf, 3)

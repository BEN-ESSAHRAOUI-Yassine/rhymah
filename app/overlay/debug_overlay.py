from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.calibration.profile import CalibrationProfile, LaneTrajectory, Point
from app.rhythm.note import KeyAction, KeyboardEvent, NoteType
from app.vision.note_detector import Candidate
from app.vision.note_classifier import LongNoteCandidate
from app.vision.tracker import TrackedNote

logger = logging.getLogger(__name__)

LANE_COLORS = {
    "A": (0, 255, 0),
    "S": (0, 200, 255),
    "D": (0, 128, 255),
    "J": (255, 128, 0),
    "K": (255, 0, 128),
    "L": (255, 0, 255),
}


@dataclass
class OverlayState:
    candidates: list[Candidate] = field(default_factory=list)
    long_notes: list[LongNoteCandidate] = field(default_factory=list)
    tracks: list[TrackedNote] = field(default_factory=list)
    events: list[KeyboardEvent] = field(default_factory=list)
    key_state: dict[str, bool] = field(default_factory=dict)
    fps: float = 0.0
    detection_latency_ms: float = 0.0
    message: str = ""


class DebugOverlay:
    def __init__(self) -> None:
        self._visible = True
        self._scale = 1.0

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value

    def render(
        self,
        frame: np.ndarray,
        calibration: CalibrationProfile | None,
        state: OverlayState,
    ) -> np.ndarray:
        if not self._visible:
            return frame

        display = frame.copy()
        h, w = display.shape[:2]

        if calibration:
            self._draw_roi(display, calibration)
            self._draw_trajectories(display, calibration)
            self._draw_hit_points(display, calibration)

        self._draw_candidates(display, state.candidates)
        self._draw_long_notes(display, state.long_notes)
        self._draw_tracks(display, state.tracks)
        self._draw_events(display, state.events, h)
        self._draw_key_state(display, state.key_state, h)
        self._draw_stats(display, state, w)

        if state.message:
            cv2.putText(display, state.message, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return display

    def _draw_roi(self, frame: np.ndarray, cal: CalibrationProfile) -> None:
        if cal.roi and cal.roi_size:
            x = int(cal.roi.x)
            y = int(cal.roi.y)
            w = int(cal.roi_size.x)
            h = int(cal.roi_size.y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

    def _draw_trajectories(self, frame: np.ndarray, cal: CalibrationProfile) -> None:
        for lane, traj in cal.lane_trajectories.items():
            color = LANE_COLORS.get(lane, (200, 200, 200))
            points = [(int(p.x), int(p.y)) for p in traj.points]
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, 1, cv2.LINE_AA)

    def _draw_hit_points(self, frame: np.ndarray, cal: CalibrationProfile) -> None:
        for lane, pt in cal.hit_points.items():
            color = LANE_COLORS.get(lane, (200, 200, 200))
            cx, cy = int(pt.x), int(pt.y)
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)
            cv2.putText(frame, lane, (cx - 5, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_candidates(self, frame: np.ndarray, candidates: list[Candidate]) -> None:
        for c in candidates:
            color = (0, 255, 0)
            cv2.drawContours(frame, [c.contour], -1, color, 1)
            cx, cy = int(c.center[0]), int(c.center[1])
            cv2.circle(frame, (cx, cy), 3, color, -1)
            cv2.putText(frame, f"{c.confidence:.2f}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def _draw_long_notes(self, frame: np.ndarray, notes: list[LongNoteCandidate]) -> None:
        for note in notes:
            color = (0, 165, 255)
            if note.body_contour is not None:
                cv2.drawContours(frame, [note.body_contour], -1, color, 1)
            cx, cy = int(note.head.center[0]), int(note.head.center[1])
            tx, ty = int(note.tail_position[0]), int(note.tail_position[1])
            cv2.line(frame, (cx, cy), (tx, ty), color, 2)
            cv2.putText(frame, f"LONG {note.confidence:.2f}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def _draw_tracks(self, frame: np.ndarray, tracks: list[TrackedNote]) -> None:
        for t in tracks:
            color = LANE_COLORS.get(t.lane, (200, 200, 200))
            cx, cy = int(t.position[0]), int(t.position[1])
            cv2.circle(frame, (cx, cy), 6, color, 2)
            cv2.putText(frame, f"#{t.id}", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            if t.velocity != 0:
                vx = int(t.velocity * 0.5)
                cv2.arrowedLine(frame, (cx, cy), (cx + vx, cy), color, 1)

    def _draw_events(
        self, frame: np.ndarray, events: list[KeyboardEvent], h: int,
    ) -> None:
        y0 = 30
        for i, ev in enumerate(events[:6]):
            color = (0, 255, 0) if ev.action == KeyAction.DOWN else (0, 0, 255)
            text = f"{ev.key} {ev.action.value}"
            cv2.putText(frame, text, (10, y0 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_key_state(
        self, frame: np.ndarray, key_state: dict[str, bool], h: int,
    ) -> None:
        x0 = frame.shape[1] - 120
        y0 = 30
        cv2.putText(frame, "Keys:", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for i, (key, is_down) in enumerate(sorted(key_state.items())):
            color = (0, 200, 0) if is_down else (80, 80, 80)
            status = "DOWN" if is_down else "UP"
            cv2.putText(frame, f"{key}: {status}", (x0, y0 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _draw_stats(
        self, frame: np.ndarray, state: OverlayState, w: int,
    ) -> None:
        y0 = frame.shape[0] - 60
        cv2.putText(frame, f"FPS: {state.fps:.0f}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Tracks: {len(state.tracks)}", (10, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Detect: {state.detection_latency_ms:.1f}ms", (10, y0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

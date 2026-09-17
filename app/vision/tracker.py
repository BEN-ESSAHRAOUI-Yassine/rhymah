from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from app.rhythm.note import NoteType, TrackedNote

logger = logging.getLogger(__name__)


@dataclass
class TrackConfig:
    max_distance: float = 80.0
    max_frames_missing: int = 5
    velocity_smoothing: float = 0.3
    initial_confidence: float = 0.9
    confidence_decay: float = 0.1
    min_confidence: float = 0.1


@dataclass
class _Detection:
    center: tuple[float, float]
    confidence: float
    note_type: NoteType
    lane: str


class Tracker:
    def __init__(self, config: TrackConfig | None = None) -> None:
        self._config = config or TrackConfig()
        self._tracks: dict[int, TrackedNote] = {}
        self._next_id = 1
        self._frame_count = 0

    @property
    def active_tracks(self) -> list[TrackedNote]:
        return [t for t in self._tracks.values()]

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def update(
        self,
        detections: list[_Detection],
        timestamp: float,
    ) -> list[TrackedNote]:
        self._frame_count += 1

        matched: dict[int, _Detection] = {}
        unmatched_dets = list(detections)

        for track_id, track in list(self._tracks.items()):
            best_det, best_idx = self._find_nearest(track, unmatched_dets)
            if best_det is not None:
                self._update_track(track, best_det, timestamp)
                matched[track_id] = best_det
                unmatched_dets.pop(best_idx)

        for track_id in list(self._tracks.keys()):
            if track_id not in matched:
                self._decay_track(track_id, timestamp)

        for det in unmatched_dets:
            self._create_track(det, timestamp)

        self._expire_tracks(timestamp)

        return self.active_tracks

    def _find_nearest(
        self,
        track: TrackedNote,
        detections: list[_Detection],
    ) -> tuple[_Detection | None, int]:
        best_det: _Detection | None = None
        best_idx = -1
        best_dist = float("inf")

        pred_x = track.position[0] + track.velocity * 0.016
        pred_y = track.position[1]

        for i, det in enumerate(detections):
            if det.lane != track.lane:
                continue

            dx = pred_x - det.center[0]
            dy = pred_y - det.center[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self._config.max_distance and dist < best_dist:
                best_dist = dist
                best_det = det
                best_idx = i

        return best_det, best_idx

    def _update_track(
        self,
        track: TrackedNote,
        det: _Detection,
        timestamp: float,
    ) -> None:
        prev_pos = track.position
        prev_time = track.last_seen

        track.previous_position = prev_pos
        track.position = det.center
        track.last_seen = timestamp
        track.confidence = min(1.0, det.confidence + 0.1)
        track.type = det.note_type

        dt = timestamp - prev_time
        if dt > 0:
            dx = track.position[0] - prev_pos[0]
            new_vel = dx / dt
            track.velocity = (
                self._config.velocity_smoothing * new_vel
                + (1 - self._config.velocity_smoothing) * track.velocity
            )

    def _decay_track(self, track_id: int, timestamp: float) -> None:
        track = self._tracks[track_id]
        track.confidence -= self._config.confidence_decay
        track.last_seen = timestamp

    def _create_track(self, det: _Detection, timestamp: float) -> None:
        track = TrackedNote(
            id=self._next_id,
            lane=det.lane,
            type=det.note_type,
            position=det.center,
            first_seen=timestamp,
            last_seen=timestamp,
            confidence=self._config.initial_confidence,
        )
        self._tracks[self._next_id] = track
        self._next_id += 1
        logger.debug("Created track %d in lane %s", track.id, det.lane)

    def _expire_tracks(self, timestamp: float) -> None:
        expired = []
        for track_id, track in self._tracks.items():
            age = timestamp - track.last_seen
            if age > self._config.max_frames_missing * 0.016:
                expired.append(track_id)
            elif track.confidence < self._config.min_confidence:
                expired.append(track_id)

        for track_id in expired:
            logger.debug("Expired track %d", track_id)
            del self._tracks[track_id]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0

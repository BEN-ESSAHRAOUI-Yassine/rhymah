from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

from app.calibration.profile import LaneTrajectory, Point
from app.rhythm.note import NoteType, TrackedNote

logger = logging.getLogger(__name__)


@dataclass
class TimingConfig:
    smoothing_window: int = 5
    min_samples: int = 3


@dataclass
class TimingEstimate:
    track_id: int
    lane: str
    note_type: NoteType
    progress: float
    velocity: float
    hit_time: float
    release_time: float | None = None
    confidence: float = 0.0
    samples_used: int = 0


class TimingEngine:
    def __init__(self, config: TimingConfig | None = None) -> None:
        self._config = config or TimingConfig()
        self._progress_history: dict[int, deque[tuple[float, float]]] = {}
        self._velocity_history: dict[int, deque[float]] = {}

    def predict(
        self,
        track: TrackedNote,
        trajectory: LaneTrajectory,
        current_time: float,
        note_duration_ms: int = 80,
    ) -> TimingEstimate:
        progress = self._calc_progress(track.position, trajectory)
        velocity = self._update_velocity(track.id, progress, current_time)

        hit_time = self._predict_hit_time(
            progress=progress,
            velocity=velocity,
            current_time=current_time,
        )

        release_time = None
        if track.type == NoteType.LONG:
            release_time = hit_time + 0.5

        confidence = self._estimate_confidence(track.id)

        return TimingEstimate(
            track_id=track.id,
            lane=track.lane,
            note_type=track.type,
            progress=progress,
            velocity=velocity,
            hit_time=hit_time,
            release_time=release_time,
            confidence=confidence,
            samples_used=len(self._progress_history.get(track.id, [])),
        )

    def _calc_progress(
        self,
        position: tuple[float, float],
        trajectory: LaneTrajectory,
    ) -> float:
        if not trajectory.points:
            return 0.0

        best_dist = float("inf")
        best_idx = 0

        for i, tp in enumerate(trajectory.points):
            dx = position[0] - tp.x
            dy = position[1] - tp.y
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        total = len(trajectory.points) - 1
        if total <= 0:
            return 0.0

        return best_idx / total

    def _update_velocity(
        self,
        track_id: int,
        progress: float,
        timestamp: float,
    ) -> float:
        if track_id not in self._progress_history:
            self._progress_history[track_id] = deque(maxlen=self._config.smoothing_window)
            self._velocity_history[track_id] = deque(maxlen=self._config.smoothing_window)

        history = self._progress_history[track_id]
        vel_history = self._velocity_history[track_id]

        history.append((timestamp, progress))

        if len(history) < 2:
            vel_history.append(0.0)
            return 0.0

        t1, p1 = history[-2]
        t2, p2 = history[-1]
        dt = t2 - t1

        if dt <= 0:
            vel_history.append(0.0)
            return 0.0

        velocity = (p2 - p1) / dt
        vel_history.append(velocity)

        return sum(vel_history) / len(vel_history)

    def _predict_hit_time(
        self,
        progress: float,
        velocity: float,
        current_time: float,
    ) -> float:
        remaining = 1.0 - progress
        if velocity <= 0:
            return current_time + 10.0

        time_to_hit = remaining / velocity
        return current_time + time_to_hit

    def _estimate_confidence(self, track_id: int) -> float:
        history = self._progress_history.get(track_id, [])
        n = len(history)
        if n < self._config.min_samples:
            return 0.3

        vel_history = self._velocity_history.get(track_id, [])
        if not vel_history:
            return 0.3

        velocities = list(vel_history)
        mean_vel = sum(velocities) / len(velocities)
        if mean_vel == 0:
            return 0.3

        variance = sum((v - mean_vel) ** 2 for v in velocities) / len(velocities)
        stability = max(0.0, 1.0 - variance * 100)
        sample_bonus = min(0.3, n * 0.05)

        return round(min(1.0, 0.4 + stability * 0.3 + sample_bonus), 3)

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._progress_history.clear()
            self._velocity_history.clear()
        else:
            self._progress_history.pop(track_id, None)
            self._velocity_history.pop(track_id, None)

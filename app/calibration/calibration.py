from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.calibration.profile import CalibrationProfile, ColorSample, LaneTrajectory, Point

logger = logging.getLogger(__name__)

LANE_KEYS = ["A", "S", "D", "J", "K", "L"]


def sample_color_from_region(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int = 10,
) -> ColorSample:
    h, w = frame.shape[:2]
    x, y = center

    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius)
    y2 = min(h, y + radius)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return ColorSample(h_min=0, h_max=180, s_min=0, s_max=255, v_min=0, v_max=255)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    h_mean = int(np.mean(h_channel))
    s_mean = int(np.mean(s_channel))
    v_mean = int(np.mean(v_channel))

    margin_h = 15
    margin_s = 60
    margin_v = 60

    return ColorSample(
        h_min=max(0, h_mean - margin_h),
        h_max=min(180, h_mean + margin_h),
        s_min=max(0, s_mean - margin_s),
        s_max=min(255, s_mean + margin_s),
        v_min=max(0, v_mean - margin_v),
        v_max=min(255, v_mean + margin_v),
    )


def create_color_mask(frame: np.ndarray, color: ColorSample) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([color.h_min, color.s_min, color.v_min], dtype=np.uint8)
    upper = np.array([color.h_max, color.s_max, color.v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def interpolate_trajectory(
    start: Point,
    end: Point,
    steps: int = 10,
) -> LaneTrajectory:
    points = []
    for i in range(steps + 1):
        t = i / steps
        x = start.x + (end.x - start.x) * t
        y = start.y + (end.y - start.y) * t
        points.append(Point(x=x, y=y))
    return LaneTrajectory(points=points)


def assign_lane_by_trajectory(
    point: Point,
    trajectories: dict[str, LaneTrajectory],
) -> tuple[str, float]:
    best_lane = "A"
    best_dist = float("inf")

    for lane, traj in trajectories.items():
        for tp in traj.points:
            dx = point.x - tp.x
            dy = point.y - tp.y
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_lane = lane

    confidence = max(0.0, 1.0 - (best_dist / 10000.0))
    return best_lane, confidence


def list_profiles(profile_dir: str | Path) -> list[str]:
    profile_dir = Path(profile_dir)
    if not profile_dir.exists():
        return []
    return sorted(p.stem for p in profile_dir.glob("*.yaml"))

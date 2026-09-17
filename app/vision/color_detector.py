from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.calibration.profile import ColorSample

logger = logging.getLogger(__name__)


@dataclass
class ColorRange:
    name: str
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int
    priority: int = 0


PRESET_COLORS: dict[str, ColorRange] = {
    "purple": ColorRange(
        name="purple",
        h_min=120, h_max=160,
        s_min=50, s_max=255,
        v_min=50, v_max=255,
        priority=0,
    ),
    "orange": ColorRange(
        name="orange",
        h_min=5, h_max=25,
        s_min=100, s_max=255,
        v_min=100, v_max=255,
        priority=1,
    ),
    "yellow": ColorRange(
        name="yellow",
        h_min=25, h_max=40,
        s_min=100, s_max=255,
        v_min=100, v_max=255,
        priority=2,
    ),
}


@dataclass
class ColorDetectorConfig:
    colors: list[str] = field(default_factory=lambda: ["purple"])
    merge_masks: bool = True


class ColorDetector:
    def __init__(self, config: ColorDetectorConfig | None = None) -> None:
        self._config = config or ColorDetectorConfig()
        self._ranges: list[ColorRange] = []
        self._custom_color: ColorSample | None = None
        self._load_presets()

    def _load_presets(self) -> None:
        self._ranges = []
        for name in self._config.colors:
            if name in PRESET_COLORS:
                self._ranges.append(PRESET_COLORS[name])
            else:
                logger.warning("Unknown color preset: %s", name)

    def set_custom_color(self, color: ColorSample) -> None:
        self._custom_color = color

    def clear_custom_color(self) -> None:
        self._custom_color = None

    def add_color(self, name: str, color_range: ColorRange) -> None:
        PRESET_COLORS[name] = color_range
        if name not in [r.name for r in self._ranges]:
            self._ranges.append(color_range)

    def remove_color(self, name: str) -> None:
        self._ranges = [r for r in self._ranges if r.name != name]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self._custom_color is not None:
            return self._mask_from_sample(hsv, self._custom_color)

        masks: list[np.ndarray] = []
        for color_range in self._ranges:
            lower = np.array([
                color_range.h_min,
                color_range.s_min,
                color_range.v_min,
            ], dtype=np.uint8)
            upper = np.array([
                color_range.h_max,
                color_range.s_max,
                color_range.v_max,
            ], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)

        if not masks:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        if self._config.merge_masks:
            combined = masks[0]
            for m in masks[1:]:
                combined = cv2.bitwise_or(combined, m)
            return combined

        return masks[0] if len(masks) == 1 else masks[0]

    def detect_per_color(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result: dict[str, np.ndarray] = {}

        for color_range in self._ranges:
            lower = np.array([
                color_range.h_min,
                color_range.s_min,
                color_range.v_min,
            ], dtype=np.uint8)
            upper = np.array([
                color_range.h_max,
                color_range.s_max,
                color_range.v_max,
            ], dtype=np.uint8)
            result[color_range.name] = cv2.inRange(hsv, lower, upper)

        return result

    def _mask_from_sample(self, hsv: np.ndarray, sample: ColorSample) -> np.ndarray:
        lower = np.array([sample.h_min, sample.s_min, sample.v_min], dtype=np.uint8)
        upper = np.array([sample.h_max, sample.s_max, sample.v_max], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    @property
    def active_colors(self) -> list[str]:
        return [r.name for r in self._ranges]

    def get_color_ranges(self) -> list[ColorRange]:
        return list(self._ranges)

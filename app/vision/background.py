from __future__ import annotations

import logging

import cv2
import numpy as np

from app.calibration.calibration import create_color_mask
from app.calibration.profile import ColorSample

logger = logging.getLogger(__name__)


class BackgroundModel:
    def __init__(self, history_size: int = 30, motion_threshold: int = 25) -> None:
        self._history_size = history_size
        self._motion_threshold = motion_threshold
        self._frames: list[np.ndarray] = []
        self._background: np.ndarray | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def history_size(self) -> int:
        return self._history_size

    def reset(self) -> None:
        self._frames.clear()
        self._background = None
        self._ready = False

    def add_frame(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._frames.append(gray)

        if len(self._frames) > self._history_size:
            self._frames.pop(0)

        if len(self._frames) >= self._history_size and not self._ready:
            self._build_background()
            self._ready = True
            logger.info("Background model ready with %d frames", self._history_size)

        return self._ready

    def _build_background(self) -> None:
        stack = np.stack(self._frames, axis=0)
        self._background = np.median(stack, axis=0).astype(np.uint8)
        logger.debug("Background model built: shape=%s", self._background.shape)

    def get_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        if not self._ready or self._background is None:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self._background)

        _, mask = cv2.threshold(diff, self._motion_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def filter_frame(
        self,
        frame: np.ndarray,
        color_sample: ColorSample | None = None,
    ) -> np.ndarray:
        motion_mask = self.get_motion_mask(frame)

        if color_sample is not None:
            color_mask = create_color_mask(frame, color_sample)
            combined = cv2.bitwise_and(motion_mask, color_mask)
        else:
            combined = motion_mask

        return combined

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mss
import numpy as np

from app.rhythm.note import Frame

logger = logging.getLogger(__name__)


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def to_mss_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class ScreenCapture:
    def __init__(self, region: CaptureRegion | None = None) -> None:
        self._sct = mss.mss()
        self._region = region
        self._sequence = 0
        self._fps_counter = _FPSCounter()

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    @property
    def region(self) -> CaptureRegion | None:
        return self._region

    @region.setter
    def region(self, value: CaptureRegion | None) -> None:
        self._region = value

    def grab(self) -> Frame | None:
        if self._region is None:
            logger.warning("No capture region set")
            return None

        timestamp = time.perf_counter()
        monitor = self._region.to_mss_dict()

        try:
            raw = self._sct.grab(monitor)
        except mss.ScreenShotError as e:
            logger.error("Screen capture failed: %s", e)
            return None

        img = np.array(raw, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self._sequence += 1
        self._fps_counter.tick()

        return Frame(image=img, timestamp=timestamp, sequence_number=self._sequence)

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> ScreenCapture:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _FPSCounter:
    def __init__(self, window: int = 60) -> None:
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> None:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

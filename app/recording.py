from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.calibration.profile import CalibrationProfile
from app.rhythm.note import Frame

logger = logging.getLogger(__name__)


@dataclass
class RecordingMeta:
    version: int = 1
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    duration_s: float = 0.0
    calibration: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "version": self.version,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_s": self.duration_s,
        }
        if self.calibration:
            d["calibration"] = self.calibration
        return d

    @classmethod
    def from_dict(cls, data: dict) -> RecordingMeta:
        return cls(
            version=data.get("version", 1),
            width=data.get("width", 0),
            height=data.get("height", 0),
            fps=data.get("fps", 0.0),
            total_frames=data.get("total_frames", 0),
            duration_s=data.get("duration_s", 0.0),
            calibration=data.get("calibration"),
        )


class Recorder:
    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir = self._output_dir / "frames"
        self._frames_dir.mkdir(exist_ok=True)
        self._frame_count = 0
        self._timestamps: list[float] = []
        self._start_time: float | None = None
        self._meta: RecordingMeta | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self, width: int, height: int, fps: float = 60.0) -> None:
        self._start_time = time.perf_counter()
        self._meta = RecordingMeta(width=width, height=height, fps=fps)
        logger.info("Recording started: %dx%d @ %.1f fps", width, height, fps)

    def capture_frame(self, frame: Frame) -> None:
        if self._start_time is None:
            return

        timestamp = frame.timestamp - self._start_time
        self._timestamps.append(timestamp)

        frame_path = self._frames_dir / f"frame_{self._frame_count:06d}.png"
        cv2.imwrite(str(frame_path), frame.image)

        self._frame_count += 1

        if self._frame_count % 100 == 0:
            logger.info("Recorded %d frames", self._frame_count)

    def save(self, calibration: CalibrationProfile | None = None) -> Path:
        if self._meta is None:
            raise ValueError("Recording not started")

        duration = self._timestamps[-1] if self._timestamps else 0.0
        self._meta.total_frames = self._frame_count
        self._meta.duration_s = duration
        if self._frame_count > 1 and duration > 0:
            self._meta.fps = (self._frame_count - 1) / duration

        if calibration:
            self._meta.calibration = calibration.to_dict()

        meta_path = self._output_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._meta.to_dict(), f, indent=2)

        ts_path = self._output_dir / "timestamps.json"
        with open(ts_path, "w") as f:
            json.dump(self._timestamps, f)

        logger.info(
            "Recording saved: %d frames, %.2fs, path=%s",
            self._frame_count, duration, self._output_dir,
        )
        return self._output_dir

    def reset(self) -> None:
        self._frame_count = 0
        self._timestamps.clear()
        self._start_time = None
        self._meta = None


class ReplayEngine:
    def __init__(self, recording_dir: str | Path) -> None:
        self._recording_dir = Path(recording_dir)
        self._meta: RecordingMeta | None = None
        self._timestamps: list[float] = []
        self._frame_index = 0

    @property
    def meta(self) -> RecordingMeta | None:
        return self._meta

    @property
    def total_frames(self) -> int:
        return len(self._timestamps)

    @property
    def position(self) -> int:
        return self._frame_index

    def load(self) -> bool:
        meta_path = self._recording_dir / "meta.json"
        ts_path = self._recording_dir / "timestamps.json"

        if not meta_path.exists() or not ts_path.exists():
            logger.error("Recording files not found: %s", self._recording_dir)
            return False

        with open(meta_path) as f:
            self._meta = RecordingMeta.from_dict(json.load(f))

        with open(ts_path) as f:
            self._timestamps = json.load(f)

        self._frame_index = 0
        logger.info(
            "Loaded recording: %d frames, %.2fs",
            self._meta.total_frames, self._meta.duration_s,
        )
        return True

    def load_calibration(self) -> CalibrationProfile | None:
        if self._meta is None or self._meta.calibration is None:
            return None
        return CalibrationProfile.from_dict(self._meta.calibration)

    def get_frame(self, index: int) -> Frame | None:
        if index < 0 or index >= len(self._timestamps):
            return None

        frame_path = self._recording_dir / "frames" / f"frame_{index:06d}.png"
        if not frame_path.exists():
            return None

        image = cv2.imread(str(frame_path))
        if image is None:
            return None

        return Frame(
            image=image,
            timestamp=self._timestamps[index],
            sequence_number=index + 1,
        )

    def get_next(self) -> Frame | None:
        frame = self.get_frame(self._frame_index)
        if frame is not None:
            self._frame_index += 1
        return frame

    def reset(self) -> None:
        self._frame_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        frame = self.get_next()
        if frame is None:
            raise StopIteration
        return frame

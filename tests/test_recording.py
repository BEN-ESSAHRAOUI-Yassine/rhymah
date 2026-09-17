from __future__ import annotations

from pathlib import Path

import numpy as np

from app.calibration.profile import CalibrationProfile, Point
from app.recording import RecordingMeta, Recorder, ReplayEngine
from app.rhythm.note import Frame


def _make_frame(seq: int = 1, ts: float = 0.0) -> Frame:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    return Frame(image=img, timestamp=ts, sequence_number=seq)


class TestRecordingMeta:
    def test_to_dict(self):
        meta = RecordingMeta(width=640, height=480, fps=60.0, total_frames=100)
        d = meta.to_dict()
        assert d["width"] == 640
        assert d["total_frames"] == 100

    def test_from_dict(self):
        data = {"width": 320, "height": 240, "fps": 30.0, "total_frames": 50, "duration_s": 1.67}
        meta = RecordingMeta.from_dict(data)
        assert meta.width == 320
        assert meta.total_frames == 50

    def test_calibration_included(self):
        meta = RecordingMeta(calibration={"name": "test"})
        d = meta.to_dict()
        assert d["calibration"]["name"] == "test"


class TestRecorder:
    def test_start(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(640, 480)
        assert rec.frame_count == 0

    def test_capture_frame(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        frame = _make_frame(ts=0.0)
        rec.capture_frame(frame)
        assert rec.frame_count == 1

    def test_save(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.capture_frame(_make_frame(ts=0.016))
        rec.save()

        assert (tmp_path / "rec" / "meta.json").exists()
        assert (tmp_path / "rec" / "timestamps.json").exists()
        assert (tmp_path / "rec" / "frames").exists()

    def test_save_with_calibration(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))

        cal = CalibrationProfile(name="test", roi=Point(0, 0))
        rec.save(calibration=cal)

        engine = ReplayEngine(tmp_path / "rec")
        engine.load()
        loaded_cal = engine.load_calibration()
        assert loaded_cal is not None
        assert loaded_cal.name == "test"

    def test_reset(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.reset()
        assert rec.frame_count == 0

    def test_frames_saved(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.save()

        frames_dir = tmp_path / "rec" / "frames"
        assert (frames_dir / "frame_000000.png").exists()


class TestReplayEngine:
    def test_load(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.capture_frame(_make_frame(ts=0.016))
        rec.save()

        engine = ReplayEngine(tmp_path / "rec")
        assert engine.load()
        assert engine.total_frames == 2

    def test_load_nonexistent(self, tmp_path: Path):
        engine = ReplayEngine(tmp_path / "nonexistent")
        assert not engine.load()

    def test_get_frame(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.save()

        engine = ReplayEngine(tmp_path / "rec")
        engine.load()
        frame = engine.get_frame(0)
        assert frame is not None
        assert frame.image.shape == (100, 100, 3)

    def test_get_next(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.capture_frame(_make_frame(ts=0.016))
        rec.save()

        engine = ReplayEngine(tmp_path / "rec")
        engine.load()

        f1 = engine.get_next()
        assert f1 is not None
        assert engine.position == 1

        f2 = engine.get_next()
        assert f2 is not None
        assert engine.position == 2

        f3 = engine.get_next()
        assert f3 is None

    def test_reset(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.save()

        engine = ReplayEngine(tmp_path / "rec")
        engine.load()
        engine.get_next()
        assert engine.position == 1
        engine.reset()
        assert engine.position == 0

    def test_iteration(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.capture_frame(_make_frame(ts=0.016))
        rec.save()

        engine = ReplayEngine(tmp_path / "rec")
        engine.load()

        frames = list(engine)
        assert len(frames) == 2

    def test_load_calibration_none(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.save()

        engine = ReplayEngine(tmp_path / "rec")
        engine.load()
        cal = engine.load_calibration()
        assert cal is None

    def test_meta_properties(self, tmp_path: Path):
        rec = Recorder(tmp_path / "rec")
        rec.start(100, 100)
        rec.capture_frame(_make_frame(ts=0.0))
        rec.save()

        engine = ReplayEngine(tmp_path / "rec")
        engine.load()
        assert engine.meta is not None
        assert engine.meta.width == 100

from __future__ import annotations

from app.rhythm.note import NoteType
from app.vision.tracker import Tracker, TrackConfig, _Detection


def _det(x: float = 100.0, y: float = 100.0, lane: str = "A", conf: float = 0.9) -> _Detection:
    return _Detection(center=(x, y), confidence=conf, note_type=NoteType.SHORT, lane=lane)


class TestTracker:
    def test_initial_state(self):
        t = Tracker()
        assert t.track_count == 0
        assert t.active_tracks == []

    def test_creates_track_on_first_detection(self):
        t = Tracker()
        t.update([_det()], timestamp=1.0)
        assert t.track_count == 1
        assert t.active_tracks[0].lane == "A"

    def test_same_track_persists(self):
        t = Tracker()
        t.update([_det(x=100)], timestamp=1.0)
        t.update([_det(x=102)], timestamp=1.016)
        assert t.track_count == 1
        assert t.active_tracks[0].id == 1

    def test_different_lanes_create_separate_tracks(self):
        t = Tracker()
        t.update([_det(lane="A"), _det(lane="S")], timestamp=1.0)
        assert t.track_count == 2

    def test_velocity_estimation(self):
        t = Tracker()
        t.update([_det(x=100)], timestamp=1.0)
        t.update([_det(x=110)], timestamp=1.016)
        track = t.active_tracks[0]
        assert track.velocity != 0.0

    def test_track_confidence_decay(self):
        cfg = TrackConfig(max_frames_missing=2, confidence_decay=0.2)
        t = Tracker(config=cfg)
        t.update([_det()], timestamp=1.0)
        assert t.track_count == 1

        t.update([], timestamp=1.016)
        assert t.track_count == 1
        track = t.active_tracks[0]
        assert track.confidence < 0.9

    def test_track_expires(self):
        cfg = TrackConfig(max_frames_missing=1, confidence_decay=0.5)
        t = Tracker(config=cfg)
        t.update([_det()], timestamp=1.0)
        t.update([], timestamp=1.02)
        t.update([], timestamp=1.05)
        assert t.track_count == 0

    def test_multiple_tracks_independent(self):
        t = Tracker()
        t.update([_det(lane="A"), _det(lane="D")], timestamp=1.0)
        t.update([_det(lane="A", x=101)], timestamp=1.016)
        assert t.track_count == 2

    def test_reset(self):
        t = Tracker()
        t.update([_det()], timestamp=1.0)
        assert t.track_count == 1
        t.reset()
        assert t.track_count == 0

    def test_new_detection_after_expire(self):
        cfg = TrackConfig(max_frames_missing=1, confidence_decay=0.5)
        t = Tracker(config=cfg)
        t.update([_det(x=100)], timestamp=1.0)
        t.update([], timestamp=1.02)
        t.update([], timestamp=1.05)
        t.update([_det(x=100)], timestamp=2.0)
        assert t.track_count == 1
        assert t.active_tracks[0].id == 2

    def test_custom_config(self):
        cfg = TrackConfig(max_distance=50, max_frames_missing=3)
        t = Tracker(config=cfg)
        assert t._config.max_distance == 50
        assert t._config.max_frames_missing == 3

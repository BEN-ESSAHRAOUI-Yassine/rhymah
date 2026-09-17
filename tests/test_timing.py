from __future__ import annotations

from app.calibration.profile import LaneTrajectory, Point
from app.rhythm.note import NoteType, TrackedNote
from app.rhythm.timing import TimingConfig, TimingEngine, TimingEstimate


def _trajectory() -> LaneTrajectory:
    return LaneTrajectory(points=[
        Point(x=100, y=0),
        Point(x=100, y=50),
        Point(x=100, y=100),
        Point(x=100, y=150),
        Point(x=100, y=200),
    ])


def _track(x: float = 100.0, y: float = 100.0, note_type: NoteType = NoteType.SHORT) -> TrackedNote:
    return TrackedNote(
        id=1,
        lane="A",
        type=note_type,
        position=(x, y),
        first_seen=1.0,
        last_seen=1.0,
        confidence=0.9,
    )


class TestTimingEngine:
    def test_initial_state(self):
        engine = TimingEngine()
        assert engine._progress_history == {}

    def test_calc_progress_start(self):
        engine = TimingEngine()
        traj = _trajectory()
        progress = engine._calc_progress((100, 0), traj)
        assert progress == 0.0

    def test_calc_progress_end(self):
        engine = TimingEngine()
        traj = _trajectory()
        progress = engine._calc_progress((100, 200), traj)
        assert progress == 1.0

    def test_calc_progress_middle(self):
        engine = TimingEngine()
        traj = _trajectory()
        progress = engine._calc_progress((100, 100), traj)
        assert 0.4 <= progress <= 0.6

    def test_predict_short_note(self):
        engine = TimingEngine()
        traj = _trajectory()
        track = _track(y=100, note_type=NoteType.SHORT)
        est = engine.predict(track, traj, current_time=1.0)
        assert isinstance(est, TimingEstimate)
        assert est.note_type == NoteType.SHORT
        assert est.hit_time > 1.0
        assert est.release_time is None

    def test_predict_long_note(self):
        engine = TimingEngine()
        traj = _trajectory()
        track = _track(y=100, note_type=NoteType.LONG)
        est = engine.predict(track, traj, current_time=1.0)
        assert est.note_type == NoteType.LONG
        assert est.release_time is not None
        assert est.release_time > est.hit_time

    def test_velocity_update(self):
        engine = TimingEngine()
        traj = _trajectory()

        track = _track(y=50)
        engine.predict(track, traj, current_time=1.0)

        track2 = _track(y=100)
        est = engine.predict(track2, traj, current_time=1.016)
        assert est.velocity != 0.0

    def test_smoothing(self):
        engine = TimingEngine(TimingConfig(smoothing_window=3))
        traj = _trajectory()

        for i in range(5):
            y = 40 + i * 30
            track = _track(y=float(y))
            engine.predict(track, traj, current_time=1.0 + i * 0.016)

        assert len(engine._velocity_history[1]) <= 3

    def test_confidence_increases_with_samples(self):
        engine = TimingEngine()
        traj = _trajectory()

        est1 = engine.predict(_track(y=50), traj, current_time=1.0)
        est2 = engine.predict(_track(y=100), traj, current_time=1.016)
        est3 = engine.predict(_track(y=150), traj, current_time=1.032)

        assert est3.confidence >= est1.confidence

    def test_reset_track(self):
        engine = TimingEngine()
        traj = _trajectory()
        engine.predict(_track(), traj, current_time=1.0)
        assert 1 in engine._progress_history

        engine.reset(track_id=1)
        assert 1 not in engine._progress_history

    def test_reset_all(self):
        engine = TimingEngine()
        traj = _trajectory()
        engine.predict(_track(), traj, current_time=1.0)
        engine.reset()
        assert len(engine._progress_history) == 0

    def test_empty_trajectory(self):
        engine = TimingEngine()
        traj = LaneTrajectory(points=[])
        progress = engine._calc_progress((100, 100), traj)
        assert progress == 0.0

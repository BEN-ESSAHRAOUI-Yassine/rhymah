from __future__ import annotations

from pathlib import Path

import numpy as np

from app.calibration.calibration import (
    assign_lane_by_trajectory,
    create_color_mask,
    interpolate_trajectory,
    sample_color_from_region,
)
from app.calibration.profile import (
    CalibrationProfile,
    ColorSample,
    LaneTrajectory,
    Point,
)


class TestPoint:
    def test_to_dict(self):
        p = Point(x=10.5, y=20.0)
        d = p.to_dict()
        assert d == {"x": 10.5, "y": 20.0}

    def test_from_dict(self):
        p = Point.from_dict({"x": 10.5, "y": 20.0})
        assert p.x == 10.5
        assert p.y == 20.0


class TestColorSample:
    def test_round_trip(self):
        cs = ColorSample(h_min=0, h_max=30, s_min=100, s_max=255, v_min=100, v_max=255)
        d = cs.to_dict()
        cs2 = ColorSample.from_dict(d)
        assert cs2.h_min == 0
        assert cs2.v_max == 255


class TestLaneTrajectory:
    def test_round_trip(self):
        traj = LaneTrajectory(points=[Point(1, 2), Point(3, 4)])
        d = traj.to_dict()
        traj2 = LaneTrajectory.from_dict(d)
        assert len(traj2.points) == 2
        assert traj2.points[0].x == 1


class TestCalibrationProfile:
    def test_default_profile(self):
        p = CalibrationProfile()
        assert p.name == "default"
        assert p.version == 1
        assert p.hit_points == {}

    def test_save_and_load(self, tmp_path: Path):
        profile = CalibrationProfile(
            name="test_profile",
            window_title="Game Window",
            hit_points={"A": Point(100, 200), "S": Point(150, 200)},
            note_color=ColorSample(h_min=0, h_max=30, s_min=100, s_max=255, v_min=100, v_max=255),
        )
        save_path = tmp_path / "test.yaml"
        profile.save(save_path)
        assert save_path.exists()

        loaded = CalibrationProfile.load(save_path)
        assert loaded.name == "test_profile"
        assert loaded.window_title == "Game Window"
        assert loaded.hit_points["A"].x == 100
        assert loaded.note_color is not None
        assert loaded.note_color.h_max == 30

    def test_to_dict_minimal(self):
        p = CalibrationProfile()
        d = p.to_dict()
        assert "name" in d
        assert "version" in d
        assert "roi" not in d

    def test_to_dict_full(self):
        p = CalibrationProfile(
            roi=Point(0, 0),
            roi_size=Point(640, 480),
            hit_points={"A": Point(100, 200)},
            lane_trajectories={"A": LaneTrajectory(points=[Point(100, 0), Point(100, 480)])},
            note_color=ColorSample(h_min=0, h_max=30, s_min=100, s_max=255, v_min=100, v_max=255),
        )
        d = p.to_dict()
        assert "roi" in d
        assert "hit_points" in d
        assert "lane_trajectories" in d
        assert "note_color" in d


class TestColorSampling:
    def test_sample_color_from_region(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (120, 200, 200)
        cs = sample_color_from_region(frame, center=(50, 50), radius=5)
        assert cs.h_min >= 0
        assert cs.h_max <= 180
        assert cs.s_min >= 0

    def test_create_color_mask(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cs = ColorSample(h_min=0, h_max=180, s_min=0, s_max=255, v_min=0, v_max=255)
        mask = create_color_mask(frame, cs)
        assert mask.shape == (100, 100)


class TestInterpolation:
    def test_interpolate_trajectory(self):
        start = Point(0, 0)
        end = Point(100, 100)
        traj = interpolate_trajectory(start, end, steps=5)
        assert len(traj.points) == 6
        assert traj.points[0].x == 0
        assert traj.points[-1].x == 100

    def test_single_step(self):
        traj = interpolate_trajectory(Point(0, 0), Point(10, 10), steps=1)
        assert len(traj.points) == 2


class TestLaneAssignment:
    def test_assign_lane(self):
        trajectories = {
            "A": LaneTrajectory(points=[Point(100, 200)]),
            "S": LaneTrajectory(points=[Point(200, 200)]),
            "D": LaneTrajectory(points=[Point(300, 200)]),
        }
        lane, conf = assign_lane_by_trajectory(Point(105, 200), trajectories)
        assert lane == "A"
        assert conf > 0.5

    def test_exact_match(self):
        trajectories = {"K": LaneTrajectory(points=[Point(500, 200)])}
        lane, conf = assign_lane_by_trajectory(Point(500, 200), trajectories)
        assert lane == "K"
        assert conf > 0.9

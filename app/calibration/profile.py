from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Point:
        return cls(x=d["x"], y=d["y"])


@dataclass
class LaneTrajectory:
    points: list[Point] = field(default_factory=list)

    def to_dict(self) -> list[dict[str, float]]:
        return [p.to_dict() for p in self.points]

    @classmethod
    def from_dict(cls, data: list[dict[str, float]]) -> LaneTrajectory:
        return cls(points=[Point.from_dict(p) for p in data])


@dataclass
class ColorSample:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def to_dict(self) -> dict[str, int]:
        return {
            "h_min": self.h_min, "h_max": self.h_max,
            "s_min": self.s_min, "s_max": self.s_max,
            "v_min": self.v_min, "v_max": self.v_max,
        }

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> ColorSample:
        return cls(**d)


@dataclass
class CalibrationProfile:
    name: str = "default"
    version: int = 1

    window_title: str = ""
    window_handle: int = 0

    roi: Point | None = None
    roi_size: Point | None = None

    hit_points: dict[str, Point] = field(default_factory=dict)
    lane_trajectories: dict[str, LaneTrajectory] = field(default_factory=dict)
    note_color: ColorSample | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "window_title": self.window_title,
            "window_handle": self.window_handle,
        }
        if self.roi:
            d["roi"] = self.roi.to_dict()
        if self.roi_size:
            d["roi_size"] = self.roi_size.to_dict()
        if self.hit_points:
            d["hit_points"] = {k: v.to_dict() for k, v in self.hit_points.items()}
        if self.lane_trajectories:
            d["lane_trajectories"] = {k: v.to_dict() for k, v in self.lane_trajectories.items()}
        if self.note_color:
            d["note_color"] = self.note_color.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationProfile:
        hit_points = {}
        if "hit_points" in data:
            for k, v in data["hit_points"].items():
                hit_points[k] = Point.from_dict(v)

        trajectories = {}
        if "lane_trajectories" in data:
            for k, v in data["lane_trajectories"].items():
                trajectories[k] = LaneTrajectory.from_dict(v)

        return cls(
            name=data.get("name", "default"),
            version=data.get("version", 1),
            window_title=data.get("window_title", ""),
            window_handle=data.get("window_handle", 0),
            roi=Point.from_dict(data["roi"]) if "roi" in data else None,
            roi_size=Point.from_dict(data["roi_size"]) if "roi_size" in data else None,
            hit_points=hit_points,
            lane_trajectories=trajectories,
            note_color=ColorSample.from_dict(data["note_color"]) if "note_color" in data else None,
        )

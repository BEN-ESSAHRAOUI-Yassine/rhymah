from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class NoteType(Enum):
    SHORT = "SHORT"
    LONG = "LONG"


class KeyAction(Enum):
    DOWN = "DOWN"
    UP = "UP"


class BarOrientation(Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    DIAGONAL = "DIAGONAL"


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    sequence_number: int


@dataclass
class Note:
    id: int
    lane: str
    type: NoteType
    detected_at: float
    hit_time: float = 0.0
    release_time: float = 0.0
    confidence: float = 0.0
    source_track_id: int | None = None


@dataclass
class TrackedNote:
    id: int
    lane: str
    type: NoteType
    position: tuple[float, float] = (0.0, 0.0)
    previous_position: tuple[float, float] | None = None
    velocity: float = 0.0
    acceleration: float = 0.0
    progress: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    confidence: float = 0.0
    hit_time: float = 0.0
    release_time: float = 0.0


@dataclass
class KeyboardEvent:
    timestamp: float
    key: str
    action: KeyAction
    note_id: int
    synchronization_group: int | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "key": self.key,
            "action": self.action.value,
            "note_id": self.note_id,
            "synchronization_group": self.synchronization_group,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KeyboardEvent:
        return cls(
            timestamp=data["timestamp"],
            key=data["key"],
            action=KeyAction(data["action"]),
            note_id=data.get("note_id", 0),
            synchronization_group=data.get("synchronization_group"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class VisualPrimitive:
    kind: str
    center: tuple[float, float]
    contour: np.ndarray | None = None
    length: float = 0.0
    width: float = 0.0
    orientation: BarOrientation = BarOrientation.VERTICAL
    confidence: float = 0.0
    lane: str = ""


@dataclass
class NoteEvent:
    id: int
    heads: list[VisualPrimitive] = field(default_factory=list)
    bars: list[VisualPrimitive] = field(default_factory=list)
    lanes: list[str] = field(default_factory=list)
    start_position: tuple[float, float] = (0.0, 0.0)
    end_position: tuple[float, float] = (0.0, 0.0)
    hit_time: float = 0.0
    release_time: float = 0.0
    confidence: float = 0.0
    note_type: NoteType = NoteType.SHORT

    @property
    def lane(self) -> str:
        return self.lanes[0] if self.lanes else ""

    @property
    def is_composite(self) -> bool:
        return len(self.lanes) > 1 or len(self.heads) > 1

    def add_head(self, primitive: VisualPrimitive) -> None:
        self.heads.append(primitive)
        if primitive.lane and primitive.lane not in self.lanes:
            self.lanes.append(primitive.lane)

    def add_bar(self, primitive: VisualPrimitive) -> None:
        self.bars.append(primitive)
        if primitive.lane and primitive.lane not in self.lanes:
            self.lanes.append(primitive.lane)

    def merge(self, other: NoteEvent) -> None:
        self.heads.extend(other.heads)
        self.bars.extend(other.bars)
        for lane in other.lanes:
            if lane not in self.lanes:
                self.lanes.append(lane)
        self.confidence = min(self.confidence, other.confidence)
        if other.hit_time and (not self.hit_time or other.hit_time < self.hit_time):
            self.hit_time = other.hit_time
        if other.release_time and other.release_time > self.release_time:
            self.release_time = other.release_time
        if len(self.lanes) > 1:
            self.note_type = NoteType.LONG

    @property
    def bar_orientations(self) -> list[BarOrientation]:
        return [b.orientation for b in self.bars]

    @property
    def has_horizontal_bar(self) -> bool:
        return any(b.orientation == BarOrientation.HORIZONTAL for b in self.bars)

    @property
    def has_vertical_bar(self) -> bool:
        return any(b.orientation == BarOrientation.VERTICAL for b in self.bars)

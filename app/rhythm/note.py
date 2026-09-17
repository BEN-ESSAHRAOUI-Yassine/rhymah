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

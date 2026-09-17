from __future__ import annotations

import pytest
import numpy as np

from app.rhythm.note import (
    Frame,
    KeyAction,
    KeyboardEvent,
    Note,
    NoteType,
    TrackedNote,
)


def test_note_type_enum():
    assert NoteType.SHORT.value == "SHORT"
    assert NoteType.LONG.value == "LONG"


def test_key_action_enum():
    assert KeyAction.DOWN.value == "DOWN"
    assert KeyAction.UP.value == "UP"


def test_frame_creation():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = Frame(image=img, timestamp=1.0, sequence_number=1)
    assert frame.timestamp == 1.0
    assert frame.sequence_number == 1
    assert frame.image.shape == (100, 100, 3)


def test_note_creation():
    note = Note(
        id=1,
        lane="A",
        type=NoteType.SHORT,
        detected_at=1.0,
        hit_time=1.5,
        release_time=1.58,
        confidence=0.95,
    )
    assert note.id == 1
    assert note.lane == "A"
    assert note.type == NoteType.SHORT
    assert note.confidence == 0.95


def test_note_long_type():
    note = Note(
        id=2,
        lane="S",
        type=NoteType.LONG,
        detected_at=2.0,
        hit_time=2.5,
        release_time=3.1,
        confidence=0.90,
    )
    assert note.type == NoteType.LONG
    assert note.release_time > note.hit_time


def test_tracked_note_creation():
    track = TrackedNote(
        id=10,
        lane="D",
        type=NoteType.SHORT,
        position=(100.0, 200.0),
        first_seen=1.0,
        last_seen=1.5,
        confidence=0.88,
    )
    assert track.id == 10
    assert track.position == (100.0, 200.0)
    assert track.velocity == 0.0


def test_tracked_note_defaults():
    track = TrackedNote(id=1, lane="K", type=NoteType.SHORT)
    assert track.previous_position is None
    assert track.velocity == 0.0
    assert track.acceleration == 0.0
    assert track.progress == 0.0


def test_keyboard_event_creation():
    event = KeyboardEvent(
        timestamp=10.250,
        key="D",
        action=KeyAction.DOWN,
        note_id=3,
        synchronization_group=1,
    )
    assert event.key == "D"
    assert event.action == KeyAction.DOWN
    assert event.synchronization_group == 1


def test_keyboard_event_no_sync_group():
    event = KeyboardEvent(
        timestamp=10.300,
        key="J",
        action=KeyAction.UP,
        note_id=3,
    )
    assert event.synchronization_group is None

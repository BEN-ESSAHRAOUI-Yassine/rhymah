from __future__ import annotations

import numpy as np
import pytest

from app.rhythm.note import (
    BarOrientation,
    KeyAction,
    KeyboardEvent,
    NoteEvent,
    NoteType,
    VisualPrimitive,
)


class TestBarOrientation:
    def test_vertical_value(self):
        assert BarOrientation.VERTICAL.value == "VERTICAL"

    def test_horizontal_value(self):
        assert BarOrientation.HORIZONTAL.value == "HORIZONTAL"

    def test_diagonal_value(self):
        assert BarOrientation.DIAGONAL.value == "DIAGONAL"


class TestVisualPrimitive:
    def test_creation(self):
        p = VisualPrimitive(
            kind="head",
            center=(100.0, 200.0),
            confidence=0.9,
            lane="A",
        )
        assert p.kind == "head"
        assert p.center == (100.0, 200.0)
        assert p.confidence == 0.9
        assert p.lane == "A"

    def test_bar_primitive(self):
        p = VisualPrimitive(
            kind="bar",
            center=(100.0, 200.0),
            length=120.0,
            width=20.0,
            orientation=BarOrientation.HORIZONTAL,
        )
        assert p.length == 120.0
        assert p.orientation == BarOrientation.HORIZONTAL


class TestNoteEvent:
    def test_creation(self):
        event = NoteEvent(id=1)
        assert event.id == 1
        assert event.heads == []
        assert event.bars == []
        assert event.lanes == []
        assert event.note_type == NoteType.SHORT

    def test_lane_property_empty(self):
        event = NoteEvent(id=1)
        assert event.lane == ""

    def test_lane_property_single(self):
        event = NoteEvent(id=1, lanes=["A"])
        assert event.lane == "A"

    def test_lane_property_multiple(self):
        event = NoteEvent(id=1, lanes=["A", "S"])
        assert event.lane == "A"

    def test_is_composite_single_head(self):
        head = VisualPrimitive(kind="head", center=(0, 0), lane="A")
        event = NoteEvent(id=1)
        event.add_head(head)
        assert not event.is_composite

    def test_is_composite_multi_head(self):
        h1 = VisualPrimitive(kind="head", center=(0, 0), lane="A")
        h2 = VisualPrimitive(kind="head", center=(100, 0), lane="S")
        event = NoteEvent(id=1)
        event.add_head(h1)
        event.add_head(h2)
        assert event.is_composite

    def test_is_composite_multi_lane(self):
        h1 = VisualPrimitive(kind="head", center=(0, 0), lane="A")
        bar = VisualPrimitive(kind="bar", center=(50, 0), lane="S", orientation=BarOrientation.HORIZONTAL)
        event = NoteEvent(id=1)
        event.add_head(h1)
        event.add_bar(bar)
        assert event.is_composite

    def test_add_head(self):
        event = NoteEvent(id=1)
        head = VisualPrimitive(kind="head", center=(10, 20), lane="D")
        event.add_head(head)
        assert len(event.heads) == 1
        assert "D" in event.lanes

    def test_add_head_no_duplicate_lane(self):
        event = NoteEvent(id=1, lanes=["D"])
        head = VisualPrimitive(kind="head", center=(10, 20), lane="D")
        event.add_head(head)
        assert event.lanes.count("D") == 1

    def test_add_bar(self):
        event = NoteEvent(id=1)
        bar = VisualPrimitive(kind="bar", center=(10, 20), lane="J")
        event.add_bar(bar)
        assert len(event.bars) == 1
        assert "J" in event.lanes

    def test_merge_combines_heads(self):
        e1 = NoteEvent(id=1)
        e1.add_head(VisualPrimitive(kind="head", center=(0, 0), lane="A"))
        e2 = NoteEvent(id=2)
        e2.add_head(VisualPrimitive(kind="head", center=(100, 0), lane="S"))
        e1.merge(e2)
        assert len(e1.heads) == 2
        assert "A" in e1.lanes
        assert "S" in e1.lanes

    def test_merge_combines_bars(self):
        e1 = NoteEvent(id=1)
        e1.add_bar(VisualPrimitive(kind="bar", center=(0, 0), lane="A"))
        e2 = NoteEvent(id=2)
        e2.add_bar(VisualPrimitive(kind="bar", center=(100, 0), lane="S"))
        e1.merge(e2)
        assert len(e1.bars) == 2

    def test_merge_takes_min_confidence(self):
        e1 = NoteEvent(id=1, confidence=0.9)
        e2 = NoteEvent(id=2, confidence=0.5)
        e1.merge(e2)
        assert e1.confidence == 0.5

    def test_merge_takes_earliest_hit_time(self):
        e1 = NoteEvent(id=1, hit_time=10.5)
        e2 = NoteEvent(id=2, hit_time=10.2)
        e1.merge(e2)
        assert e1.hit_time == 10.2

    def test_merge_takes_latest_release_time(self):
        e1 = NoteEvent(id=1, release_time=10.8)
        e2 = NoteEvent(id=2, release_time=11.2)
        e1.merge(e2)
        assert e1.release_time == 11.2

    def test_merge_becomes_long_on_multi_lane(self):
        e1 = NoteEvent(id=1, note_type=NoteType.SHORT)
        e1.add_head(VisualPrimitive(kind="head", center=(0, 0), lane="A"))
        e2 = NoteEvent(id=2, note_type=NoteType.SHORT)
        e2.add_head(VisualPrimitive(kind="head", center=(100, 0), lane="S"))
        e1.merge(e2)
        assert e1.note_type == NoteType.LONG

    def test_bar_orientations(self):
        e = NoteEvent(id=1)
        e.add_bar(VisualPrimitive(kind="bar", center=(0, 0), orientation=BarOrientation.HORIZONTAL))
        e.add_bar(VisualPrimitive(kind="bar", center=(0, 0), orientation=BarOrientation.VERTICAL))
        assert BarOrientation.HORIZONTAL in e.bar_orientations
        assert BarOrientation.VERTICAL in e.bar_orientations

    def test_has_horizontal_bar(self):
        e = NoteEvent(id=1)
        assert not e.has_horizontal_bar
        e.add_bar(VisualPrimitive(kind="bar", center=(0, 0), orientation=BarOrientation.HORIZONTAL))
        assert e.has_horizontal_bar

    def test_has_vertical_bar(self):
        e = NoteEvent(id=1)
        assert not e.has_vertical_bar
        e.add_bar(VisualPrimitive(kind="bar", center=(0, 0), orientation=BarOrientation.VERTICAL))
        assert e.has_vertical_bar

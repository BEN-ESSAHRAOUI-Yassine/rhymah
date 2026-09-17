from __future__ import annotations

import numpy as np

from app.rhythm.note import BarOrientation, NoteEvent, NoteType, VisualPrimitive
from app.vision.note_grouper import GrouperConfig, NoteGrouper


def _head(lane: str = "A", center: tuple[float, float] = (100.0, 100.0)) -> VisualPrimitive:
    return VisualPrimitive(kind="head", center=center, confidence=0.9, lane=lane)


def _bar(
    lane: str = "",
    center: tuple[float, float] = (100.0, 200.0),
    orientation: BarOrientation = BarOrientation.VERTICAL,
    length: float = 100.0,
) -> VisualPrimitive:
    return VisualPrimitive(
        kind="bar",
        center=center,
        length=length,
        width=20.0,
        orientation=orientation,
        confidence=0.8,
        lane=lane,
    )


class TestGrouperConfig:
    def test_defaults(self):
        cfg = GrouperConfig()
        assert cfg.max_head_bar_dist == 60.0
        assert cfg.max_head_head_dist == 80.0
        assert cfg.max_bar_bar_dist == 50.0
        assert cfg.min_group_confidence == 0.3
        assert cfg.horizontal_connect_dist == 100.0


class TestNoteGrouper:
    def test_empty_input(self):
        grouper = NoteGrouper()
        events = grouper.group([], [])
        assert events == []

    def test_single_head_no_bars(self):
        grouper = NoteGrouper()
        heads = [_head("A")]
        events = grouper.group(heads, [])
        assert len(events) == 1
        assert events[0].lane == "A"
        assert events[0].note_type == NoteType.SHORT

    def test_head_with_nearby_vertical_bar(self):
        grouper = NoteGrouper(GrouperConfig(max_head_bar_dist=100.0))
        heads = [_head("A", center=(100.0, 100.0))]
        bars = [_bar(center=(100.0, 180.0), orientation=BarOrientation.VERTICAL)]
        events = grouper.group(heads, bars)
        assert len(events) == 1
        assert len(events[0].heads) == 1
        assert len(events[0].bars) == 1

    def test_two_heads_with_horizontal_bar(self):
        grouper = NoteGrouper(GrouperConfig(max_head_bar_dist=100.0))
        heads = [
            _head("A", center=(80.0, 100.0)),
            _head("S", center=(200.0, 100.0)),
        ]
        bars = [
            _bar(
                center=(140.0, 100.0),
                orientation=BarOrientation.HORIZONTAL,
                length=140.0,
            ),
        ]
        events = grouper.group(heads, bars)
        assert len(events) == 1
        assert events[0].is_composite
        assert len(events[0].heads) == 2
        assert "A" in events[0].lanes
        assert "S" in events[0].lanes

    def test_heads_without_connecting_bar_not_grouped(self):
        grouper = NoteGrouper(GrouperConfig(max_head_head_dist=80.0, max_head_bar_dist=30.0))
        heads = [
            _head("A", center=(50.0, 100.0)),
            _head("L", center=(300.0, 100.0)),
        ]
        events = grouper.group(heads, [])
        assert len(events) == 2

    def test_bar_only_becomes_long_note(self):
        grouper = NoteGrouper()
        bars = [_bar(center=(100.0, 200.0), orientation=BarOrientation.VERTICAL)]
        events = grouper.group([], bars)
        assert len(events) == 1
        assert events[0].note_type == NoteType.LONG
        assert len(events[0].bars) == 1

    def test_event_ids_are_unique(self):
        grouper = NoteGrouper()
        heads = [_head("A"), _head("S"), _head("D")]
        events = grouper.group(heads, [])
        ids = [e.id for e in events]
        assert len(ids) == len(set(ids))

    def test_events_sorted_by_confidence(self):
        grouper = NoteGrouper()
        heads = [
            VisualPrimitive(kind="head", center=(0, 0), confidence=0.5, lane="A"),
            VisualPrimitive(kind="head", center=(100, 0), confidence=0.9, lane="S"),
            VisualPrimitive(kind="head", center=(200, 0), confidence=0.7, lane="D"),
        ]
        events = grouper.group(heads, [])
        confidences = [e.confidence for e in events]
        assert confidences == sorted(confidences, reverse=True)

    def test_start_end_positions_set(self):
        grouper = NoteGrouper(GrouperConfig(max_head_bar_dist=100.0))
        heads = [_head("A", center=(100.0, 80.0))]
        bars = [_bar(center=(100.0, 200.0), orientation=BarOrientation.VERTICAL)]
        events = grouper.group(heads, bars)
        assert len(events) == 1
        assert events[0].start_position == (100.0, 80.0)
        assert events[0].end_position == (100.0, 200.0)

    def test_reset(self):
        grouper = NoteGrouper()
        grouper.group([_head("A")], [])
        grouper.reset()
        events = grouper.group([_head("A")], [])
        assert events[0].id == 1

    def test_multi_lane_becomes_long(self):
        grouper = NoteGrouper(GrouperConfig(max_head_bar_dist=100.0))
        heads = [
            _head("A", center=(80.0, 100.0)),
            _head("S", center=(180.0, 100.0)),
        ]
        bars = [
            _bar(
                center=(130.0, 100.0),
                orientation=BarOrientation.HORIZONTAL,
                length=120.0,
            ),
        ]
        events = grouper.group(heads, bars)
        assert len(events) == 1
        assert events[0].note_type == NoteType.LONG

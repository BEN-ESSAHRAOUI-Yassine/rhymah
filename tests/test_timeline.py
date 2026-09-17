from __future__ import annotations

from app.rhythm.note import KeyAction, NoteType
from app.rhythm.timeline import Timeline, TimelineConfig
from app.rhythm.timing import TimingEstimate


def _short_estimate(track_id: int = 1, lane: str = "A", hit_time: float = 1.0) -> TimingEstimate:
    return TimingEstimate(
        track_id=track_id,
        lane=lane,
        note_type=NoteType.SHORT,
        progress=0.9,
        velocity=0.5,
        hit_time=hit_time,
        confidence=0.9,
    )


def _long_estimate(track_id: int = 2, lane: str = "S", hit_time: float = 1.0, release_time: float = 1.5) -> TimingEstimate:
    return TimingEstimate(
        track_id=track_id,
        lane=lane,
        note_type=NoteType.LONG,
        progress=0.9,
        velocity=0.5,
        hit_time=hit_time,
        release_time=release_time,
        confidence=0.9,
    )


LANE_KEYS = {"A": "A", "S": "S", "D": "D", "J": "J", "K": "K", "L": "L"}


class TestTimeline:
    def test_empty_timeline(self):
        tl = Timeline()
        assert tl.events == []

    def test_short_note_generates_down_up(self):
        tl = Timeline()
        tl.add_estimate(_short_estimate(hit_time=1.0), LANE_KEYS)
        events = tl.events
        assert len(events) == 2
        assert events[0].action == KeyAction.DOWN
        assert events[0].key == "A"
        assert events[1].action == KeyAction.UP
        assert events[1].key == "A"

    def test_short_note_release_after_hit(self):
        tl = Timeline()
        tl.add_estimate(_short_estimate(hit_time=1.0), LANE_KEYS)
        events = tl.events
        assert events[1].timestamp > events[0].timestamp

    def test_long_note_generates_down_up(self):
        tl = Timeline()
        tl.add_estimate(_long_estimate(hit_time=1.0, release_time=1.5), LANE_KEYS)
        events = tl.events
        assert len(events) == 2
        assert events[0].action == KeyAction.DOWN
        assert events[1].action == KeyAction.UP
        assert events[1].timestamp == 1.5

    def test_multiple_notes(self):
        tl = Timeline()
        tl.add_estimate(_short_estimate(track_id=1, lane="A", hit_time=1.0), LANE_KEYS)
        tl.add_estimate(_short_estimate(track_id=2, lane="D", hit_time=1.1), LANE_KEYS)
        events = tl.events
        assert len(events) == 4

    def test_events_sorted_by_timestamp(self):
        tl = Timeline()
        tl.add_estimate(_short_estimate(hit_time=2.0), LANE_KEYS)
        tl.add_estimate(_short_estimate(track_id=2, lane="D", hit_time=1.0), LANE_KEYS)
        events = tl.events
        assert events[0].timestamp <= events[1].timestamp

    def test_synchronize_groups_nearby(self):
        cfg = TimelineConfig(synchronization_window_ms=10)
        tl = Timeline(config=cfg)
        tl.add_estimate(_short_estimate(hit_time=1.0), LANE_KEYS)
        tl.add_estimate(_short_estimate(track_id=2, lane="D", hit_time=1.005), LANE_KEYS)
        synced = tl.synchronize()
        down_events = [e for e in synced if e.action == KeyAction.DOWN]
        assert down_events[0].synchronization_group == down_events[1].synchronization_group

    def test_synchronize_separates_distant(self):
        cfg = TimelineConfig(synchronization_window_ms=5)
        tl = Timeline(config=cfg)
        tl.add_estimate(_short_estimate(hit_time=1.0), LANE_KEYS)
        tl.add_estimate(_short_estimate(track_id=2, lane="D", hit_time=1.1), LANE_KEYS)
        synced = tl.synchronize()
        down_events = [e for e in synced if e.action == KeyAction.DOWN]
        assert down_events[0].synchronization_group != down_events[1].synchronization_group

    def test_clear(self):
        tl = Timeline()
        tl.add_estimate(_short_estimate(), LANE_KEYS)
        tl.clear()
        assert tl.events == []

    def test_lane_mapping(self):
        tl = Timeline()
        tl.add_estimate(_short_estimate(lane="K"), LANE_KEYS)
        events = tl.events
        assert events[0].key == "K"

    def test_long_note_holds_key(self):
        tl = Timeline()
        tl.add_estimate(_long_estimate(hit_time=1.0, release_time=2.0), LANE_KEYS)
        events = tl.events
        down = [e for e in events if e.action == KeyAction.DOWN]
        up = [e for e in events if e.action == KeyAction.UP]
        assert len(down) == 1
        assert len(up) == 1
        assert up[0].timestamp - down[0].timestamp == 1.0

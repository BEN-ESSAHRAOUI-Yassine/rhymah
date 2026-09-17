from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.rhythm.note import BarOrientation, NoteEvent, NoteType, VisualPrimitive

logger = logging.getLogger(__name__)


@dataclass
class GrouperConfig:
    max_head_bar_dist: float = 60.0
    max_head_head_dist: float = 80.0
    max_bar_bar_dist: float = 50.0
    min_group_confidence: float = 0.3
    horizontal_connect_dist: float = 100.0


class NoteGrouper:
    def __init__(self, config: GrouperConfig | None = None) -> None:
        self._config = config or GrouperConfig()
        self._next_id = 1

    def group(
        self,
        heads: list[VisualPrimitive],
        bars: list[VisualPrimitive],
    ) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        assigned_bars: set[int] = set()
        assigned_heads: set[int] = set()

        for i, head in enumerate(heads):
            if id(head) in assigned_heads:
                continue

            event = NoteEvent(
                id=self._next_id,
                confidence=head.confidence,
                note_type=NoteType.SHORT,
            )
            event.add_head(head)
            assigned_heads.add(id(head))

            self._connect_nearby_bars(head, bars, event, assigned_bars, assigned_heads, heads)
            self._connect_nearby_heads(head, heads, event, assigned_heads, assigned_bars, bars)

            if len(event.heads) > 1 or len(event.bars) > 0:
                self._refine_event(event)

            events.append(event)
            self._next_id += 1

        for i, bar in enumerate(bars):
            if id(bar) in assigned_bars:
                continue

            event = NoteEvent(
                id=self._next_id,
                confidence=bar.confidence,
                note_type=NoteType.LONG,
            )
            event.add_bar(bar)
            assigned_bars.add(id(bar))
            events.append(event)
            self._next_id += 1

        events.sort(key=lambda e: e.confidence, reverse=True)
        logger.debug("Grouped into %d note events", len(events))
        return events

    def _connect_nearby_bars(
        self,
        head: VisualPrimitive,
        bars: list[VisualPrimitive],
        event: NoteEvent,
        assigned_bars: set[int],
        assigned_heads: set[int],
        all_heads: list[VisualPrimitive],
    ) -> None:
        for bar in bars:
            if id(bar) in assigned_bars:
                continue

            dist = self._point_to_segment_dist(
                head.center,
                bar.center,
                bar.length,
                bar.orientation,
            )

            if dist <= self._config.max_head_bar_dist:
                event.add_bar(bar)
                assigned_bars.add(id(bar))

                if bar.orientation == BarOrientation.HORIZONTAL:
                    self._find_connected_heads_horizontal(
                        bar, all_heads, event, assigned_heads,
                    )

    def _connect_nearby_heads(
        self,
        head: VisualPrimitive,
        all_heads: list[VisualPrimitive],
        event: NoteEvent,
        assigned_heads: set[int],
        assigned_bars: set[int],
        bars: list[VisualPrimitive],
    ) -> None:
        for other in all_heads:
            if id(other) in assigned_heads or other is head:
                continue

            dist = math.dist(head.center, other.center)

            if dist <= self._config.max_head_head_dist:
                connecting_bar = self._find_connecting_bar(
                    head, other, bars, assigned_bars,
                )

                if connecting_bar is not None:
                    event.add_head(other)
                    assigned_heads.add(id(other))
                    event.add_bar(connecting_bar)
                    assigned_bars.add(id(connecting_bar))
            else:
                connecting_bar = self._find_connecting_bar(
                    head, other, bars, assigned_bars,
                )
                if connecting_bar is not None:
                    event.add_head(other)
                    assigned_heads.add(id(other))
                    event.add_bar(connecting_bar)
                    assigned_bars.add(id(connecting_bar))

    def _find_connecting_bar(
        self,
        head_a: VisualPrimitive,
        head_b: VisualPrimitive,
        bars: list[VisualPrimitive],
        assigned_bars: set[int],
    ) -> VisualPrimitive | None:
        mid_x = (head_a.center[0] + head_b.center[0]) / 2.0
        mid_y = (head_a.center[1] + head_b.center[1]) / 2.0

        best_bar: VisualPrimitive | None = None
        best_dist = float("inf")

        for bar in bars:
            if id(bar) in assigned_bars:
                continue

            if bar.orientation != BarOrientation.HORIZONTAL:
                continue

            bar_mid_x, bar_mid_y = bar.center
            dist_to_mid = math.dist((mid_x, mid_y), (bar_mid_x, bar_mid_y))

            if dist_to_mid < self._config.max_bar_bar_dist and dist_to_mid < best_dist:
                best_dist = dist_to_mid
                best_bar = bar

        return best_bar

    def _find_connected_heads_horizontal(
        self,
        bar: VisualPrimitive,
        all_heads: list[VisualPrimitive],
        event: NoteEvent,
        assigned_heads: set[int],
    ) -> None:
        half_len = bar.length / 2.0
        left_end = (bar.center[0] - half_len, bar.center[1])
        right_end = (bar.center[0] + half_len, bar.center[1])

        for other in all_heads:
            if id(other) in assigned_heads or other in event.heads:
                continue

            dist_left = math.dist(other.center, left_end)
            dist_right = math.dist(other.center, right_end)
            min_dist = min(dist_left, dist_right)

            if min_dist <= self._config.max_head_bar_dist:
                event.add_head(other)
                assigned_heads.add(id(other))

    def _point_to_segment_dist(
        self,
        point: tuple[float, float],
        segment_center: tuple[float, float],
        segment_length: float,
        orientation: BarOrientation,
    ) -> float:
        if orientation == BarOrientation.HORIZONTAL:
            half_len = segment_length / 2.0
            sx = segment_center[0] - half_len
            ex = segment_center[0] + half_len
            sy = segment_center[1]
            ey = segment_center[1]
        elif orientation == BarOrientation.VERTICAL:
            half_len = segment_length / 2.0
            sx = segment_center[0]
            ex = segment_center[0]
            sy = segment_center[1] - half_len
            ey = segment_center[1] + half_len
        else:
            return math.dist(point, segment_center)

        px, py = point
        dx = ex - sx
        dy = ey - sy
        len_sq = dx * dx + dy * dy

        if len_sq == 0:
            return math.dist(point, segment_center)

        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / len_sq))
        proj_x = sx + t * dx
        proj_y = sy + t * dy

        return math.dist(point, (proj_x, proj_y))

    def _refine_event(self, event: NoteEvent) -> None:
        if len(event.heads) > 1 or event.has_horizontal_bar:
            event.note_type = NoteType.LONG

        if event.heads:
            top_head = min(event.heads, key=lambda h: h.center[1])
            event.start_position = top_head.center

        if event.bars:
            bottom_bar = max(event.bars, key=lambda b: b.center[1])
            event.end_position = bottom_bar.center
        elif event.heads:
            bottom_head = max(event.heads, key=lambda h: h.center[1])
            event.end_position = bottom_head.center

    def reset(self) -> None:
        self._next_id = 1

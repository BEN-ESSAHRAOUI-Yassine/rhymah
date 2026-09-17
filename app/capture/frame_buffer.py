from __future__ import annotations

import threading
from collections import deque

from app.rhythm.note import Frame


class FrameBuffer:
    def __init__(self, capacity: int = 3) -> None:
        self._capacity = capacity
        self._buffer: deque[Frame] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, frame: Frame) -> None:
        with self._lock:
            self._buffer.append(frame)

    def latest(self) -> Frame | None:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def pop(self) -> Frame | None:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer.pop()

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __bool__(self) -> bool:
        return len(self) > 0

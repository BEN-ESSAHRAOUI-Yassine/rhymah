from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
from dataclasses import dataclass

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    handle: int
    title: str
    class_name: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    @property
    def is_visible(self) -> bool:
        return user32.IsWindowVisible(self.handle) != 0

    @property
    def client_rect(self) -> tuple[int, int, int, int]:
        rect = ctypes.wintypes.RECT()
        user32.GetClientRect(self.handle, ctypes.byref(rect))
        pt = ctypes.wintypes.POINT(0, 0)
        user32.ClientToScreen(self.handle, ctypes.byref(pt))
        return (pt.x, pt.y, pt.x + rect.right, pt.y + rect.bottom)

def enumerate_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def callback(hwnd: int, lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd) == 0:
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)
        class_name = class_buf.value

        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))

        windows.append(WindowInfo(
            handle=hwnd,
            title=title,
            class_name=class_name,
            rect=(rect.left, rect.top, rect.right, rect.bottom),
        ))
        return True

    user32.EnumWindows(callback, 0)
    return windows


def find_window_by_title(title_substring: str) -> WindowInfo | None:
    windows = enumerate_windows()
    title_lower = title_substring.lower()
    for w in windows:
        if title_lower in w.title.lower():
            return w
    return None

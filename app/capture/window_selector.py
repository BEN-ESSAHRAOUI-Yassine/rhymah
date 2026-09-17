from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
from dataclasses import dataclass

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

logger = logging.getLogger(__name__)

TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPTHREAD = 0x00000004


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

    def refresh_rect(self) -> None:
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(self.handle, ctypes.byref(rect))
        self.rect = (rect.left, rect.top, rect.right, rect.bottom)


class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.wintypes.DWORD),
        ("cntUsage", ctypes.wintypes.DWORD),
        ("th32ProcessID", ctypes.wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", ctypes.wintypes.DWORD),
        ("cntThreads", ctypes.wintypes.DWORD),
        ("th32ParentProcessID", ctypes.wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("szExeFile", ctypes.c_char * 260),
    ]


class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.wintypes.DWORD),
        ("cntUsage", ctypes.wintypes.DWORD),
        ("th32ThreadID", ctypes.wintypes.DWORD),
        ("th32OwnerProcessID", ctypes.wintypes.DWORD),
        ("tpBasePri", ctypes.c_long),
        ("tpDeltaPri", ctypes.c_long),
        ("dwFlags", ctypes.wintypes.DWORD),
    ]


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


def find_window_by_class(class_name: str) -> WindowInfo | None:
    hwnd = user32.FindWindowW(class_name, None)
    if hwnd == 0:
        return None

    length = user32.GetWindowTextLengthW(hwnd)
    if length > 0:
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
    else:
        title = ""

    actual_class_buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, actual_class_buf, 256)

    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))

    return WindowInfo(
        handle=hwnd,
        title=title,
        class_name=actual_class_buf.value,
        rect=(rect.left, rect.top, rect.right, rect.bottom),
    )


def find_window_by_process(process_name: str) -> WindowInfo | None:
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == -1:
        return None

    try:
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)

        pid = None
        if kernel32.Process32First(snapshot, ctypes.byref(entry)):
            while True:
                try:
                    name = entry.szExeFile.decode("utf-8", errors="ignore").lower()
                    if process_name.lower() in name:
                        pid = entry.th32ProcessID
                        break
                except Exception:
                    pass
                if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                    break

        if pid is None:
            return None

        return _find_window_by_pid(pid)
    finally:
        kernel32.CloseHandle(snapshot)


def _find_window_by_pid(pid: int) -> WindowInfo | None:
    thread_snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if thread_snapshot == -1:
        return None

    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(THREADENTRY32)

        found_hwnd: int | None = None

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def enum_callback(hwnd: int, lparam: int) -> bool:
            nonlocal found_hwnd
            if user32.IsWindowVisible(hwnd) != 0:
                found_hwnd = hwnd
                return False
            return True

        if kernel32.Thread32First(thread_snapshot, ctypes.byref(entry)):
            while True:
                if entry.th32OwnerProcessID == pid:
                    user32.EnumThreadWindows(entry.th32ThreadID, enum_callback, 0)
                    if found_hwnd is not None:
                        break
                if not kernel32.Thread32Next(thread_snapshot, ctypes.byref(entry)):
                    break

        if found_hwnd is None:
            return None

        length = user32.GetWindowTextLengthW(found_hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(found_hwnd, buf, length + 1)
            title = buf.value
        else:
            title = ""

        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(found_hwnd, class_buf, 256)

        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(found_hwnd, ctypes.byref(rect))

        return WindowInfo(
            handle=found_hwnd,
            title=title,
            class_name=class_buf.value,
            rect=(rect.left, rect.top, rect.right, rect.bottom),
        )
    finally:
        kernel32.CloseHandle(thread_snapshot)


KNOWN_GAME_WINDOWS: dict[str, dict[str, str]] = {
    "genshin_impact": {
        "class_name": "UnityWndClass",
        "process_name": "GenshinImpact.exe",
        "title substring": "Genshin",
    },
}


def auto_detect_game() -> WindowInfo | None:
    for game_key, info in KNOWN_GAME_WINDOWS.items():
        result = find_window_by_class(info["class_name"])
        if result is not None:
            logger.info("Auto-detected %s via class name", game_key)
            return result

    for game_key, info in KNOWN_GAME_WINDOWS.items():
        result = find_window_by_process(info["process_name"])
        if result is not None:
            logger.info("Auto-detected %s via process name", game_key)
            return result

    return None

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

VK_MAP = {
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "J": 0x4A,
    "K": 0x4B,
    "L": 0x4C,
    "F8": 0x77,
}


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT)]
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]


def _send_key(vk: int, flags: int) -> bool:
    scan_code = user32.MapVirtualKeyW(vk, 0)
    inp = _INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = vk
    inp.union.ki.wScan = scan_code
    inp.union.ki.dwFlags = flags
    inp.union.ki.time = 0
    inp.union.ki.dwExtraInfo = None

    result = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
    return result == 1


class KeyboardDriver:
    def __init__(self, dry_run: bool = True) -> None:
        self._dry_run = dry_run

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        self._dry_run = value

    def key_down(self, key: str) -> bool:
        vk = VK_MAP.get(key.upper())
        if vk is None:
            logger.error("Unknown key for key_down: %s", key)
            return False

        if self._dry_run:
            logger.debug("DRY RUN key_down: %s (vk=0x%02X)", key, vk)
            return True

        return _send_key(vk, 0)

    def key_up(self, key: str) -> bool:
        vk = VK_MAP.get(key.upper())
        if vk is None:
            logger.error("Unknown key for key_up: %s", key)
            return False

        if self._dry_run:
            logger.debug("DRY RUN key_up: %s (vk=0x%02X)", key, vk)
            return True

        return _send_key(vk, KEYEVENTF_KEYUP)

    def release_all(self) -> list[str]:
        released = []
        for key in ["A", "S", "D", "J", "K", "L"]:
            if self.key_up(key):
                released.append(key)
        return released

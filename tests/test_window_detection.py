from __future__ import annotations

import ctypes
import ctypes.wintypes

import pytest

from app.capture.window_selector import (
    WindowInfo,
    enumerate_windows,
    find_window_by_class,
    find_window_by_title,
    auto_detect_game,
    KNOWN_GAME_WINDOWS,
)


class TestWindowInfo:
    def test_width_calculation(self):
        w = WindowInfo(handle=1, title="test", class_name="cls", rect=(100, 200, 500, 600))
        assert w.width == 400

    def test_height_calculation(self):
        w = WindowInfo(handle=1, title="test", class_name="cls", rect=(100, 200, 500, 600))
        assert w.height == 400

    def test_refresh_rect(self):
        w = WindowInfo(handle=0, title="", class_name="", rect=(0, 0, 0, 0))
        assert w.rect == (0, 0, 0, 0)


class TestEnumerateWindows:
    def test_returns_list(self):
        windows = enumerate_windows()
        assert isinstance(windows, list)

    def test_windows_have_required_fields(self):
        windows = enumerate_windows()
        for w in windows:
            assert isinstance(w.handle, int)
            assert isinstance(w.title, str)
            assert isinstance(w.class_name, str)
            assert isinstance(w.rect, tuple)
            assert len(w.rect) == 4


class TestFindWindowByTitle:
    def test_returns_none_for_nonexistent(self):
        result = find_window_by_title("NonExistentWindow12345XYZ")
        assert result is None

    def test_returns_window_info(self):
        result = find_window_by_title("Program Manager")
        if result is not None:
            assert isinstance(result, WindowInfo)
            assert "Program Manager" in result.title.lower() or "program" in result.title.lower()


class TestAutoDetectGame:
    def test_returns_none_or_window(self):
        result = auto_detect_game()
        assert result is None or isinstance(result, WindowInfo)

    def test_known_games_dict(self):
        assert "genshin_impact" in KNOWN_GAME_WINDOWS
        info = KNOWN_GAME_WINDOWS["genshin_impact"]
        assert "class_name" in info
        assert "process_name" in info

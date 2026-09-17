from __future__ import annotations

from pathlib import Path

from app.config.loader import AppConfig, load_config


def test_load_default_config():
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.name == "rhythm-bot"
    assert config.timing.synchronization_window_ms == 8
    assert config.vision.minimum_note_confidence == 0.80
    assert config.keyboard.emergency_key == "F8"


def test_load_nonexistent_config_returns_defaults():
    config = load_config(config_path="/nonexistent/path.yaml")
    assert isinstance(config, AppConfig)
    assert config.name == "rhythm-bot"


def test_config_with_overrides():
    overrides = {"timing": {"synchronization_window_ms": 16}}
    config = load_config(user_overrides=overrides)
    assert isinstance(config, AppConfig)


def test_config_dataclass_fields():
    config = AppConfig()
    assert config.timing is not None
    assert config.vision is not None
    assert config.keyboard is not None
    assert config.scheduler is not None
    assert config.calibration is not None


def test_keyboard_default_lanes():
    config = AppConfig()
    expected = {"A": "A", "S": "S", "D": "D", "J": "J", "K": "K", "L": "L"}
    assert config.keyboard.lanes == expected

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TimingConfig:
    synchronization_window_ms: int = 8
    input_offset_ms: int = 0
    short_note_duration_ms: int = 80


@dataclass
class VisionConfig:
    minimum_note_confidence: float = 0.80
    background_frames: int = 30
    motion_threshold: int = 25


@dataclass
class KeyboardConfig:
    lanes: dict[str, str] = field(default_factory=lambda: {
        "A": "A",
        "S": "S",
        "D": "D",
        "J": "J",
        "K": "K",
        "L": "L",
    })
    emergency_key: str = "F8"


@dataclass
class SchedulerConfig:
    high_resolution_spin_us: int = 100
    timing_log_interval: int = 50


@dataclass
class CalibrationConfig:
    profile_dir: str = "config/profiles"


@dataclass
class AppConfig:
    name: str = "rhythm-bot"
    version: str = "0.1.0"
    timing: TimingConfig = field(default_factory=TimingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


def _merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(data: dict[str, Any]) -> AppConfig:
    return AppConfig(
        name=data.get("app", {}).get("name", "rhythm-bot"),
        version=data.get("app", {}).get("version", "0.1.0"),
        timing=TimingConfig(**data.get("timing", {})),
        vision=VisionConfig(**data.get("vision", {})),
        keyboard=KeyboardConfig(**data.get("keyboard", {})),
        scheduler=SchedulerConfig(**data.get("scheduler", {})),
        calibration=CalibrationConfig(**data.get("calibration", {})),
    )


def load_config(
    config_path: str | Path | None = None,
    user_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    base_data: dict[str, Any] = {}

    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                base_data = loaded

    if user_overrides:
        base_data = _merge(base_data, user_overrides)

    return _dict_to_config(base_data)

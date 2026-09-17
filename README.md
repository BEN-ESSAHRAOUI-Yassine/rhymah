# Rhythm Game Keyboard Automation

A Windows desktop application that observes rhythm-game gameplay, detects incoming notes, predicts hit/release times, and produces synchronized keyboard input.

## Overview

The system uses computer vision to detect short and long notes in a rhythm game, tracks them across frames, predicts when each note reaches its hit position, and sends precise keyboard events via Windows `SendInput`.

### Pipeline

```
Screen Capture
  → Background Filtering
  → Note Detection (short/long)
  → Lane Assignment
  → Temporal Tracking
  → Hit/Release Time Prediction
  → Event Timeline
  → Synchronization
  → Keyboard Scheduler
  → SendInput
```

### Note Types

- **Short note**: Circular note requiring a quick key press/release
- **Long note**: Circular head with elongated body; key held from head to tail

### Keyboard Layout

```
A  S  D      J  K  L
```

## Setup

### Requirements

- Windows 10/11
- Python 3.12+
- A rhythm game running in a window

### Installation

1. Clone the repository:

```bash
git clone https://github.com/youruser/rhymah.git
cd rhymah
```

2. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- `opencv-python` — Image processing and contour analysis
- `numpy` — Array operations
- `mss` — Fast screen capture
- `PySide6` — Desktop UI
- `PyYAML` — Configuration files
- `pytest` — Testing

## Usage

### Running Tests

```bash
python -m pytest tests/ -v
```

### Dry-Run Mode

Dry-run mode detects notes and shows predicted keyboard events without sending real input:

```bash
python -m app.main --log-level DEBUG
```

### Real Input Mode

Real keyboard input is only enabled explicitly through the application UI or by transitioning to `RUNNING` state. The system will never send real input accidentally.

### Emergency Stop

Press `F8` at any time to:
1. Stop all scheduling
2. Release all held keys
3. Enter emergency stop state

## Configuration

Configuration is stored in `app/config/config.yaml`:

```yaml
timing:
  synchronization_window_ms: 8
  input_offset_ms: 0
  short_note_duration_ms: 80

vision:
  minimum_note_confidence: 0.80
  background_frames: 30
  motion_threshold: 25

keyboard:
  lanes:
    A: A
    S: S
    D: D
    J: J
    K: K
    L: L
  emergency_key: F8
```

### Calibration Profiles

Calibration profiles are saved in `config/profiles/` as YAML files. Each profile stores:
- ROI (region of interest) relative to the game window
- Hit points for each lane
- Lane trajectories
- Note color (HSV ranges)

## Architecture

```
app/
├── main.py                 # Entry point
├── state.py                # Application state machine
├── automation.py           # State-gated controller
├── performance.py          # Performance metrics
├── recording.py            # Record/replay engine
├── capture/
│   ├── screen_capture.py   # MSS-based capture
│   └── frame_buffer.py     # Thread-safe frame buffer
├── vision/
│   ├── background.py       # Static background filtering
│   ├── note_detector.py    # Short-note detection
│   ├── note_classifier.py  # Long-note detection
│   └── tracker.py          # Temporal tracking
├── rhythm/
│   ├── note.py             # Domain models
│   ├── timing.py           # Hit/release prediction
│   ├── timeline.py         # Event generation
│   └── synchronizer.py     # Sync grouping
├── input/
│   ├── keyboard.py         # Windows SendInput driver
│   ├── key_state.py        # Key state machine
│   └── scheduler.py        # Priority-queue scheduler
├── calibration/
│   ├── profile.py          # Calibration data
│   └── calibration.py      # Calibration utilities
├── overlay/
│   └── debug_overlay.py    # Debug visualization
└── config/
    └── config.yaml         # Default configuration
```

### Key Design Principles

- **No hardcoded coordinates** — All values come from calibration
- **Vision never couples to keyboard** — Strict pipeline separation
- **Dry-run always available** — Never accidentally send real input
- **Long notes = one DOWN + one UP** — Not repeated presses
- **Monotonic timestamps** — Never use frame count as time
- **Emergency stop** — Always releases all held keys

## Application States

```
STOPPED → CALIBRATING → READY → DRY_RUN → RUNNING
                                    ↓
                              EMERGENCY_STOP → STOPPED
```

Real keyboard input is only permitted in `RUNNING` state.

## Testing

192 tests covering:
- Configuration loading
- Domain models
- State machine transitions
- Screen capture and frame buffering
- Background filtering
- Short and long note detection
- Temporal tracking
- Timing prediction
- Event timeline and synchronization
- Keyboard state machine
- Scheduler and emergency stop
- Recording and replay
- Performance metrics

```bash
python -m pytest tests/ -v
```

## License

MIT

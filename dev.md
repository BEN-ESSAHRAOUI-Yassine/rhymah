# dev.md — Implementation Plan & Progress

---

## Phase 1: Repository & Infrastructure

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Create folder structure | Done | `app/`, `tests/`, `config/profiles/` created |
| 1.2 | Create `requirements.txt` | Done | opencv-python, numpy, mss, PySide6, PyYAML, pytest |
| 1.3 | Config loader (`app/config/`) | Done | YAML defaults → user config → calibration profile |
| 1.4 | Logging setup | Done | Structured console + file, configurable level |
| 1.5 | Domain models (`app/rhythm/note.py`) | Done | NoteType, Note, TrackedNote, KeyboardEvent, Frame |
| 1.6 | Application state machine | Done | STOPPED → CALIBRATING → READY → DRY_RUN → RUNNING |
| 1.7 | Entry point (`app/main.py`) | Done | CLI args, init config/logging/UI shell |
| 1.8 | Phase 1 tests | Done | 23/23 tests pass |

## Phase 2: Screen Capture

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | ROI capture via MSS | Done | `CaptureRegion` + `ScreenCapture.grab()` |
| 2.2 | Frame timestamping | Done | `time.perf_counter()` per frame |
| 2.3 | FPS counter | Done | `_FPSCounter` sliding window |
| 2.4 | Frame buffer | Done | Thread-safe `deque`-backed ring buffer |
| 2.5 | Preview window | Pending | Live capture display for verification |
| 2.6 | Capture tests | Done | 11 tests: buffer, FPS, region, capture |

## Phase 3: Calibration

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | Window enumeration & selection | Pending | List visible windows, user selects target |
| 3.2 | ROI selection | Pending | Interactive crop on captured frame |
| 3.3 | Hit-point definition | Pending | Click 6 points for A/S/D/J/K/L hit positions |
| 3.4 | Lane trajectory representation | Done | `LaneTrajectory`, interpolation, assignment |
| 3.5 | Color sampling | Done | `sample_color_from_region()`, HSV mask |
| 3.6 | Profile save/load | Done | `CalibrationProfile.save/load`, YAML serialization |
| 3.7 | Calibration tests | Done | 14 tests: profile, color, trajectory, assignment |

## Phase 4: Static Background Model

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | Multi-frame capture | Done | Accumulate N frames |
| 4.2 | Background model build | Done | Median-based background model |
| 4.3 | Temporal differencing | Done | Motion mask via absdiff + threshold |
| 4.4 | Filtered output visualization | Done | `BackgroundModel.filter_frame()` |
| 4.5 | Background filter tests | Done | 10 tests: ready, motion, reset, filter |

## Phase 5: Short-Note Detector

| # | Task | Status | Notes |
|---|------|--------|-------|
| 5.1 | HSV color masking | Done | `create_color_mask()` in calibration |
| 5.2 | Contour extraction | Done | `cv2.findContours` + analysis |
| 5.3 | Geometry filtering | Done | Area, perimeter, circularity, convexity |
| 5.4 | Motion filtering | Done | Via `BackgroundModel.filter_frame()` |
| 5.5 | Lane assignment | Done | Trajectory-based assignment |
| 5.6 | Confidence scoring | Done | Multi-signal confidence [0,1] |
| 5.7 | Short-note tests | Done | 11 tests: geometry, rejection, assignment |

## Phase 6: Long-Note Detector

| # | Task | Status | Notes |
|---|------|--------|-------|
| 6.1 | Head detection | Done | Candidate from `ShortNoteDetector` |
| 6.2 | Body identification | Done | Elongated contour filtering |
| 6.3 | Head-body association | Done | Proximity + bounding box overlap |
| 6.4 | Geometry analysis | Done | Aspect ratio, body length/width |
| 6.5 | Orientation & tail | Done | `minAreaRect` angle + tail estimate |
| 6.6 | Classification | Done | LONG vs SHORT decision |
| 6.7 | Long-note tests | Done | 8 tests: elongation, head, tail |

## Phase 7: Temporal Tracker

| # | Task | Status | Notes |
|---|------|--------|-------|
| 7.1 | Track model | Done | `TrackedNote` with history |
| 7.2 | Nearest-neighbor association | Done | Predicted position + gating |
| 7.3 | Lane constraint | Done | Same-lane matching only |
| 7.4 | Track lifecycle | Done | Create, maintain, expire |
| 7.5 | Confidence decay | Done | Reduce on missed frames |
| 7.6 | Tracker tests | Done | 11 tests: ID, velocity, expiry |

## Phase 8: Timing Engine

| # | Task | Status | Notes |
|---|------|--------|-------|
| 8.1 | Lane progress calculation | Done | Progress along trajectory [0→1] |
| 8.2 | Velocity estimation | Done | Progress change / time delta |
| 8.3 | Smoothing | Done | Rolling deque average |
| 8.4 | Hit-time prediction | Done | Remaining / velocity |
| 8.5 | Release-time prediction | Done | Long-note tail arrival |
| 8.6 | Timing tests | Done | 12 tests: progress, velocity, smoothing |

## Phase 9: Event Timeline

| # | Task | Status | Notes |
|---|------|--------|-------|
| 9.1 | Note → KeyboardEvent conversion | Done | `Timeline.add_estimate()` |
| 9.2 | Long-note DOWN/UP | Done | Single DOWN + release_time UP |
| 9.3 | Synchronization grouping | Done | Configurable window grouping |
| 9.4 | Deterministic ordering | Done | Sort by timestamp + action |
| 9.5 | Timeline tests | Done | 11 tests: short, long, sync |

## Phase 10: Virtual Keyboard Simulation

| # | Task | Status | Notes |
|---|------|--------|-------|
| 10.1 | 6-key state model | Done | `KeyboardStateMachine` |
| 10.2 | Event execution (simulated) | Done | Apply events without SendInput |
| 10.3 | Dry-run output | Done | Scheduler dispatches + logs |
| 10.4 | Timing error metrics | Done | `SchedulerMetrics` |
| 10.5 | Simulation tests | Done | 10 tests: state, duplicates, log |

## Phase 11: Windows Keyboard Driver

| # | Task | Status | Notes |
|---|------|--------|-------|
| 11.1 | SendInput ctypes wrapper | Done | `_send_key()` via ctypes |
| 11.2 | `key_down(key)` | Done | Press + scan code |
| 11.3 | `key_up(key)` | Done | Release + KEYEVENTF_KEYUP |
| 11.4 | `release_all()` | Done | Release all 6 keys |
| 11.5 | Keyboard-state safety | Done | Duplicate DOWN/UP prevention |
| 11.6 | Driver tests | Done | 6 tests: dry-run, unknown key |

## Phase 12: Precision Scheduler

| # | Task | Status | Notes |
|---|------|--------|-------|
| 12.1 | Priority queue | Done | `heapq` ordered by timestamp |
| 12.2 | Coarse wait | Done | `time.sleep()` until near |
| 12.3 | High-resolution spin | Done | Busy-wait final microseconds |
| 12.4 | Timing metrics | Done | Mean/max error tracking |
| 12.5 | Cancellation | Done | `cancel_all()` clears queue |
| 12.6 | Scheduler tests | Done | 8 tests: schedule, cancel, dispatch |

## Phase 12b: Synchronizer

| # | Task | Status | Notes |
|---|------|--------|-------|
| 12b.1 | Event grouping | Done | `Synchronizer.group()` |
| 12b.2 | Configurable window | Done | Default 8ms |
| 12b.3 | Synchronizer tests | Done | 7 tests: grouping, reset |

## Phase 13: Real Input Integration

| # | Task | Status | Notes |
|---|------|--------|-------|
| 13.1 | State gate | Done | `AutomationController` with state checks |
| 13.2 | Emergency stop (F8) | Done | F8 key triggers `emergency_stop()` |
| 13.3 | Integration test | Done | Dry-run → real → stop cycle tested |
| 13.4 | Safety test | Done | 21 tests: state gate, emergency, dispatch |

## Phase 14: Replay & Regression Testing

| # | Task | Status | Notes |
|---|------|--------|-------|
| 14.1 | Recording format | Done | `RecordingMeta` JSON + timestamps + frames |
| 14.2 | Recorder | Done | `Recorder` saves frames + metadata |
| 14.3 | Replay engine | Done | `ReplayEngine` loads and iterates frames |
| 14.4 | Deterministic replay | Done | Same recording → same frames |
| 14.5 | Regression tests | Done | 17 tests: recorder, replay, calibration |

## Phase 15: Optimization

| # | Task | Status | Notes |
|---|------|--------|-------|
| 15.1 | Profiling | Done | `PerformanceTracker` with sliding window |
| 15.2 | Capture optimization | Done | FPS tracking built-in |
| 15.3 | Contour processing | Done | Allocations tracked via latency metrics |
| 15.4 | Tracker optimization | Done | Metrics available for tuning |
| 15.5 | Timing precision | Done | Scheduler error tracking |
| 15.6 | Performance benchmarks | Done | 12 tests: FPS, latency, metrics |

## Phase 16: Composite Note Events

| # | Task | Status | Notes |
|---|------|--------|-------|
| 16.1 | NoteEvent model | Done | Multi-lane, multi-head, multi-bar composite events |
| 16.2 | VisualPrimitive model | Done | Heads, bars with orientation |
| 16.3 | BarOrientation enum | Done | VERTICAL, HORIZONTAL, DIAGONAL |
| 16.4 | Horizontal bar detection | Done | `_classify_orientation()` + `detect_bars()` |
| 16.5 | Multi-lane note grouping | Done | `NoteGrouper` with connectivity analysis |
| 16.6 | Orange/yellow color support | Done | `ColorDetector` with preset HSV ranges |
| 16.7 | NoteEvent tests | Done | 20 tests: creation, merge, composite, orientation |
| 16.8 | Bar detection tests | Done | 15 tests: orientation, confidence, lane assignment |
| 16.9 | Note grouper tests | Done | 11 tests: grouping, connectivity, multi-lane |
| 16.10 | Color detector tests | Done | 14 tests: presets, detection, custom colors |

---

## Milestone 1 Complete (Phases 1–15) + Phase 16

**Total tests: 255**

| Phase | Tests |
|-------|-------|
| 1: Infrastructure | 23 |
| 2: Screen Capture | 11 |
| 3: Calibration | 14 |
| 4: Background Model | 10 |
| 5: Short-Note Detector | 11 |
| 6: Long-Note Detector | 8 |
| 7: Temporal Tracker | 11 |
| 8: Timing Engine | 12 |
| 9: Event Timeline | 11 |
| 10: Virtual Keyboard | 10 |
| 11: Keyboard Driver | 6 |
| 12: Scheduler | 8 |
| 12b: Synchronizer | 7 |
| 13: Real Input Integration | 21 |
| 14: Replay & Regression | 17 |
| 15: Optimization | 12 |
| 16: Composite Note Events | 60 |
| **Total** | **255** |

**Key modules:**
- `app/automation.py` — State-gated automation controller
- `app/recording.py` — Record/replay engine
- `app/performance.py` — Performance metrics tracker

**Deliverable:** Full pipeline from capture → vision → tracking → timing → scheduling → keyboard, with dry-run and real modes, emergency stop, recording/replay, and performance monitoring.

---

## Rules

- All environment values come from calibration, never hardcoded.
- Vision never couples directly to keyboard input.
- Dry-run mode available permanently.
- Tests required for every core feature.
- Monotonic timestamps, never frame count.
- Long notes = one DOWN + one UP, not repeated presses.
- Emergency stop = release all + stop.

# Rhythm Game Keyboard Automation --- Project Specification

## 1. Project Overview

Build a Windows desktop application that observes a rhythm-game gameplay
area, detects incoming notes, predicts when each note reaches its hit
position, and produces synchronized keyboard input.

The target interaction model has six keyboard lanes:

``` text
A   S   D       J   K   L
```

The game contains at least two note types:

-   **Short note**: a circular note requiring a quick key press/release.
-   **Long note**: a circular note head with an elongated same-color
    rectangular/bar body. It requires the corresponding key to remain
    held until the end of the note.

The game can also produce simultaneous notes, including:

-   multiple short notes at the same time;
-   multiple long notes at the same time;
-   short + long combinations;
-   simultaneous key-down and/or key-up events.

The application must therefore model **keyboard press/release events
with timestamps and durations**, rather than treating everything as a
generic mouse click.

## 2. Primary Engineering Goal

The system must be predictive, not reactive.

Do **not** implement:

``` text
detect note -> immediately press key
```

Instead implement:

``` text
Screen Capture
    -> Candidate Detection
    -> Static-Graphic Filtering
    -> Note Classification
    -> Lane Assignment
    -> Temporal Tracking
    -> Trajectory Estimation
    -> Hit-Time / Release-Time Prediction
    -> Event Timeline
    -> Synchronization
    -> High-Precision Keyboard Scheduler
    -> Keyboard Input
```

Detection, prediction, and input must remain separate modules.

## 3. Scope

### In scope

-   Windows desktop operation.
-   Screen-region capture.
-   Calibration of gameplay region and six lanes.
-   Detection of moving note candidates.
-   Short/long note classification.
-   Detection of same-color elongated long-note bodies.
-   Lane assignment.
-   Temporal tracking of notes across frames.
-   Hit-time prediction.
-   Long-note release-time prediction.
-   Simultaneous-note grouping.
-   Keyboard `KEY_DOWN` / `KEY_UP` events.
-   High-resolution event scheduling.
-   Dry-run/simulation mode.
-   Debug overlay.
-   Recording/replay analysis mode.
-   Configuration and calibration profiles.
-   Safety/emergency-stop handling.
-   Automated tests for core non-UI components.

### Explicitly out of scope for the first implementation

-   Machine-learning/YOLO detection.
-   OCR.
-   Game-memory reading.
-   Process injection.
-   DLL injection.
-   Kernel drivers.
-   Anti-cheat bypass.
-   Network manipulation.
-   Automatic game launching/logging in.
-   Any technique intended to circumvent game security.

Start with conventional screen capture, computer vision, geometry,
tracking, and normal Windows keyboard input.

## 4. Technology Stack

Use:

-   Python 3.12+
-   OpenCV
-   NumPy
-   MSS for screen capture
-   PySide6 for desktop UI/overlay
-   Windows `SendInput` through `ctypes` for keyboard events
-   `time.perf_counter()` and Windows high-resolution timing facilities
    where appropriate
-   PyYAML for configuration

Do not use PyAutoGUI for precision-critical final input.

## 5. Repository Structure

``` text
rhythm-bot/
├── app/
│   ├── main.py
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── screen_capture.py
│   │   └── frame_buffer.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── color_detector.py
│   │   ├── note_detector.py
│   │   ├── note_classifier.py
│   │   ├── lane_detector.py
│   │   ├── motion_detector.py
│   │   └── tracker.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   └── profile.py
│   ├── rhythm/
│   │   ├── __init__.py
│   │   ├── note.py
│   │   ├── trajectory.py
│   │   ├── timing.py
│   │   ├── timeline.py
│   │   └── synchronizer.py
│   ├── input/
│   │   ├── __init__.py
│   │   ├── keyboard.py
│   │   ├── key_state.py
│   │   └── scheduler.py
│   ├── overlay/
│   │   ├── __init__.py
│   │   └── debug_overlay.py
│   └── config/
│       └── config.yaml
├── tests/
├── recordings/
├── logs/
├── config/
│   └── profiles/
├── Project.md
├── AGENTS.md
├── requirements.txt
└── README.md
```

Keep modules small and testable.

## 6. Domain Model

### Note

Every note should have a stable internal representation:

``` python
Note:
    id
    lane
    type              # SHORT | LONG
    detected_at
    hit_time
    release_time
    confidence
    source_track_id
```

For short notes:

``` text
hit_time = press/release timing
release_time = hit_time + configured short duration
```

For long notes:

``` text
hit_time = key-down time
release_time = key-up time
```

The exact short-note duration must be configurable and should be
validated against observed game behavior.

### TrackedNote

``` python
TrackedNote:
    id
    lane
    type
    position
    previous_position
    velocity
    acceleration
    progress
    first_seen
    last_seen
    confidence
```

### Keyboard Event

``` python
KeyboardEvent:
    timestamp
    key
    action       # DOWN | UP
    note_id
    synchronization_group
```

## 7. Calibration System

Never hardcode coordinates from the supplied reference screenshot.

The calibration system must support:

1.  Selecting the gameplay ROI.
2.  Defining hit positions for A/S/D/J/K/L.
3.  Defining or learning each lane trajectory.
4.  Sampling the target note color.
5.  Saving/loading profiles.

Example profile:

``` yaml
name: default

keyboard:
  lanes:
    A: A
    S: S
    D: D
    J: J
    K: K
    L: L

timing:
  synchronization_window_ms: 8
  input_offset_ms: 0

vision:
  minimum_note_confidence: 0.80
```

Calibration data should be versioned so future changes do not silently
invalidate old profiles.

## 8. Screen Capture

Capture only the gameplay ROI.

Requirements:

-   Target at least 60 FPS.
-   Support higher capture rates when available.
-   Timestamp every frame using a monotonic high-resolution clock.
-   Never use frame number as the authoritative time.
-   Expose actual capture FPS in diagnostics.
-   Avoid unnecessary copies of image data.

Define:

``` python
Frame:
    image
    timestamp
    sequence_number
```

## 9. Vision Pipeline

The detector must not classify every purple pixel as a note.

The visual pipeline must combine:

``` text
color
+ shape
+ motion
+ expected lane geometry
+ temporal consistency
```

### 9.1 Color detection

Convert frames to HSV and create a configurable mask for the note color.

Support color calibration by sampling the note head/body.

Do not assume one hardcoded RGB value.

### 9.2 Static graphics filtering

The screenshot contains static purple lane/path graphics.

Build a static/background model from multiple frames.

Use temporal differences and/or background subtraction to suppress
persistent geometry.

The objective is:

``` text
static purple path -> ignore
moving purple note -> candidate
```

### 9.3 Short-note detection

Look for circular note heads using:

-   contours;
-   connected components;
-   area;
-   perimeter;
-   circularity;
-   approximate radius;
-   color;
-   motion;
-   proximity to an expected lane trajectory.

Do not depend exclusively on HoughCircles.

### 9.4 Long-note detection

A long note visually consists of a note head plus a same-color elongated
body/bar.

Use:

-   connected component/contour geometry;
-   bounding box;
-   aspect ratio;
-   contour area;
-   perimeter;
-   orientation;
-   body length;
-   body width;
-   circular head detection;
-   continuity between head and body;
-   temporal movement.

Do not use a single fixed rule such as `height > X`.

Classify long notes from geometry relative to their lane trajectory.

The internal result should contain:

``` text
type = LONG
hit_time = predicted head arrival
release_time = predicted tail/end arrival
```

### 9.5 Lane assignment

Do not rely only on global X coordinate because the paths are curved.

Represent six calibrated lane trajectories.

For each candidate, calculate its relationship to each trajectory and
choose the lane with the best valid fit.

The result must include a confidence score.

## 10. Temporal Tracking

Detection must be temporal.

A note detected in consecutive frames must retain the same track ID.

Initial implementation:

-   nearest-neighbor association;
-   predicted-position gating;
-   lane constraint;
-   type consistency;
-   confidence decay.

If necessary, introduce a Kalman filter later.

Tracking must estimate:

``` text
position
velocity
optional acceleration
progress along lane trajectory
```

Lost tracks should expire after a configurable timeout.

## 11. Trajectory and Timing Prediction

This is the central subsystem.

For each tracked note estimate:

``` text
current progress
current velocity
future progress
hit time
release time for long notes
```

Avoid assuming straight-line motion.

Use the calibrated lane trajectory and progress along that trajectory.

A simple first model may estimate time from progress change over
multiple frames:

``` text
progress_delta / time_delta
```

Then improve using smoothing and optional acceleration estimation.

Do not trust a single frame.

Maintain a rolling set of timing estimates and smooth them.

Example:

``` text
1.034 s
1.021 s
1.028 s
1.024 s

-> smoothed estimate
```

## 12. Long-Note Release Prediction

For a long note:

``` text
head -> hit position -> body continues -> tail/end
```

The system must schedule:

``` text
hit_time   -> KEY_DOWN
release_time -> KEY_UP
```

The key must remain held between those events.

Never repeatedly send `KEY_DOWN` while the note is held.

If the visual tail is temporarily ambiguous, maintain the track and
update the release prediction until confidence falls below the
configured safety threshold.

## 13. Synchronization

Notes whose predicted hit times are within a configurable tolerance must
be grouped.

Default:

``` yaml
timing:
  synchronization_window_ms: 8
```

Example:

``` text
D hit = 10.250
J hit = 10.253
```

becomes:

``` text
10.250:
    D DOWN
    J DOWN
```

Synchronization must also apply to release events when appropriate.

The event scheduler should dispatch events in a group as close together
as the operating system permits.

Do not artificially delay one key to process another.

## 14. Event Timeline

The detector must never directly control the keyboard.

Pipeline:

``` text
vision -> Note
Note -> Timing Prediction
Prediction -> Keyboard Events
Keyboard Events -> Synchronizer
Synchronizer -> Scheduler
Scheduler -> Keyboard Driver
```

Example:

``` text
10.200 A DOWN
10.250 D DOWN
10.250 J DOWN
10.330 A UP
10.800 D UP
10.800 J UP
```

Events must be ordered by timestamp.

## 15. Keyboard State Machine

Maintain state for:

``` text
A S D J K L
```

Each key is either:

``` text
UP
DOWN
```

Rules:

-   Never send duplicate DOWN events for an already-held key.
-   Never send UP for a key that is not held.
-   Long notes use one DOWN followed by one UP.
-   Application shutdown must release all held keys.
-   Emergency stop must immediately stop scheduling and release all held
    keys.

## 16. Windows Keyboard Driver

Use Windows `SendInput` through `ctypes`.

Expose a small interface:

``` python
keyboard.key_down("A")
keyboard.key_up("A")
keyboard.release_all()
```

Keep Windows-specific code isolated in `input/keyboard.py`.

Do not allow Windows API details to leak into vision/rhythm modules.

## 17. Precision Scheduler

Use a priority queue ordered by absolute monotonic timestamps.

For each event:

``` text
coarse wait
    ->
high-resolution final wait/spin
    ->
SendInput
```

Do not use ordinary `sleep()` as the only timing mechanism.

The scheduler must report:

``` text
scheduled timestamp
actual dispatch timestamp
timing error
```

Example:

``` text
scheduled: 10.250000
actual:    10.250800
error:     +0.800 ms
```

Collect these metrics during testing.

## 18. Timing Offset

Support:

``` yaml
timing:
  input_offset_ms: 0
```

The offset must be applied consistently to scheduled input timestamps.

Document the sign convention.

Do not hardcode an offset based on assumptions.

## 19. Dry-Run Mode

Real keyboard input must be disabled during development unless
explicitly enabled.

Dry-run output should look like:

``` text
00:01.240  A  SHORT
00:01.580  S  LONG   release=00:02.140
00:02.010  D+J SHORT
00:02.730  K+L LONG  release=00:03.410
```

Dry-run should visualize the same events that real mode would send.

## 20. Debug Overlay

Implement an optional transparent overlay showing:

-   gameplay ROI;
-   six lane trajectories;
-   hit positions;
-   detected candidates;
-   tracked note IDs;
-   note type;
-   lane;
-   confidence;
-   predicted hit time;
-   predicted release time;
-   synchronization groups;
-   scheduled keyboard events.

The overlay is a development/debugging tool, not part of the core
detector.

## 21. Recording and Replay

Implement a recording mode that stores:

-   gameplay ROI video or frame sequence;
-   timestamps;
-   calibration profile;
-   configuration version.

The detector must be runnable against a recording without sending real
keyboard input.

This is required for reproducible testing.

Development workflow:

``` text
record once
-> run detector
-> inspect overlay/log
-> modify algorithm
-> replay same recording
```

## 22. Confidence System

Every candidate and final note must have confidence.

Example:

``` text
A SHORT 0.97
S LONG  0.94
D SHORT 0.71
```

Default minimum:

``` yaml
vision:
  minimum_note_confidence: 0.80
```

Low-confidence candidates must not automatically generate real keyboard
input.

In dry-run mode, show low-confidence detections for debugging.

## 23. Fail-Safe Behavior

Automation must stop if:

-   capture is lost;
-   gameplay ROI disappears;
-   calibration is invalid;
-   game window/ROI becomes unavailable;
-   tracking becomes globally unreliable;
-   keyboard driver fails;
-   event queue becomes invalid;
-   excessive prediction errors occur.

Provide a global emergency-stop key:

``` text
F8
```

Emergency stop must:

1.  stop creating new events;
2.  stop scheduler dispatch;
3.  release every held key;
4.  switch to stopped state;
5.  require explicit restart.

## 24. Application States

Implement:

``` text
STOPPED
CALIBRATING
READY
DRY_RUN
RUNNING
ERROR
EMERGENCY_STOP
```

Transitions must be explicit.

Real input is permitted only in `RUNNING`.

## 25. UI Requirements

A minimal PySide6 UI should provide:

-   profile selector;
-   calibration button;
-   start dry-run;
-   start real mode;
-   stop;
-   emergency-stop indicator;
-   capture FPS;
-   detector FPS;
-   active tracked notes;
-   keyboard state;
-   timing error statistics;
-   debug overlay toggle.

Keep the UI separate from the core engine.

## 26. Testing Requirements

Write unit tests for:

### Vision

-   color mask behavior;
-   short-note geometry;
-   long-note geometry;
-   false-positive filtering;
-   lane assignment.

### Tracking

-   same note keeps same ID;
-   crossing/nearby candidates do not merge incorrectly;
-   lost notes expire correctly.

### Timing

-   predicted times are monotonic;
-   smoothing behaves correctly;
-   long-note duration is preserved;
-   synchronization groups are deterministic.

### Keyboard

-   DOWN/UP state transitions;
-   duplicate DOWN prevention;
-   duplicate UP prevention;
-   release-all behavior.

### Scheduler

-   event ordering;
-   simultaneous events;
-   cancellation;
-   emergency stop;
-   timing metrics.

## 27. Development Order

OpenCode must implement the project incrementally.

### Phase 1 --- Repository and infrastructure

Tasks:

-   create folder structure;
-   create virtual environment instructions;
-   create requirements.txt;
-   create configuration loader;
-   create logging;
-   create basic application entry point;
-   create tests.

Acceptance:

``` text
python -m pytest
```

works with the initial test suite.

### Phase 2 --- Screen capture

Tasks:

-   implement ROI capture;
-   timestamp frames;
-   expose FPS;
-   create preview window;
-   verify 60+ FPS where hardware allows.

Do not implement keyboard input.

### Phase 3 --- Calibration

Tasks:

-   ROI selection;
-   six hit points;
-   lane trajectory representation;
-   color sampling;
-   profile save/load.

Acceptance:

-   calibrated profile can be saved and restored exactly.

### Phase 4 --- Static background model

Tasks:

-   capture multiple frames;
-   build persistent/static mask;
-   visualize filtered output.

Acceptance:

-   static lane graphics are substantially suppressed without removing
    moving notes.

### Phase 5 --- Short-note detector

Tasks:

-   color candidate generation;
-   contour analysis;
-   circularity filtering;
-   motion filtering;
-   lane assignment;
-   confidence score.

Acceptance:

-   detector identifies short notes in supplied test recordings without
    keyboard input.

### Phase 6 --- Long-note detector

Tasks:

-   identify note head;
-   identify same-color elongated body;
-   calculate geometry;
-   associate head/body;
-   estimate tail position;
-   classify LONG.

Acceptance:

-   short and long notes are distinguished in recorded test cases.

### Phase 7 --- Temporal tracker

Tasks:

-   persistent track IDs;
-   position history;
-   velocity;
-   progress;
-   confidence;
-   track lifecycle.

Acceptance:

-   a single moving note does not become many different notes.

### Phase 8 --- Timing engine

Tasks:

-   lane-progress calculation;
-   velocity estimation;
-   prediction smoothing;
-   hit-time prediction;
-   long-note release prediction.

Acceptance:

-   predictions are displayed on the overlay and logs.

### Phase 9 --- Event timeline

Tasks:

-   convert Notes into keyboard events;
-   long-note DOWN/UP;
-   synchronization grouping;
-   deterministic ordering.

Acceptance:

-   dry-run output accurately describes expected input.

### Phase 10 --- Virtual keyboard simulation

Tasks:

-   show six-key state;
-   simulate event execution;
-   show event timing errors.

No real input.

### Phase 11 --- Windows keyboard driver

Tasks:

-   implement `SendInput`;
-   key-down;
-   key-up;
-   release-all;
-   keyboard-state safety.

### Phase 12 --- Precision scheduler

Tasks:

-   priority queue;
-   absolute timestamps;
-   high-resolution final wait;
-   timing metrics;
-   cancellation.

### Phase 13 --- Real input integration

Only enable after dry-run and scheduler tests pass.

Acceptance:

-   real input is behind an explicit setting;
-   emergency stop always works.

### Phase 14 --- Replay/testing improvements

Tasks:

-   recordings;
-   deterministic replay;
-   debug logs;
-   performance metrics;
-   regression test recordings.

### Phase 15 --- Optimization

Only after correctness:

-   reduce CPU usage;
-   reduce allocations;
-   optimize capture;
-   optimize contour processing;
-   improve tracker;
-   improve timing precision.

## 28. Performance Targets

Initial targets:

``` text
capture:       >= 60 FPS
vision:        <= 10 ms/frame target
prediction:    <= 2 ms target
scheduler:     sub-millisecond target where OS scheduling permits
```

These are engineering targets, not guarantees.

Always measure actual performance.

## 29. Logging

Create structured logs for:

-   frame rate;
-   detection count;
-   track count;
-   confidence;
-   predicted timing;
-   scheduled events;
-   actual dispatch;
-   timing error;
-   dropped frames;
-   emergency stops;
-   errors.

Do not log unnecessary sensitive information.

## 30. Acceptance Criteria

The project is considered functionally complete when:

1.  The gameplay ROI can be calibrated.
2.  Six lanes can be represented independently.
3.  Static purple graphics can be distinguished from moving note
    candidates.
4.  Short notes can be detected.
5.  Long notes with same-color rectangular bodies can be detected.
6.  Notes can be tracked across frames.
7.  Each note receives a lane assignment.
8.  Hit timing can be predicted.
9.  Long-note release timing can be predicted.
10. Simultaneous notes are grouped.
11. Events are represented as KEY_DOWN/KEY_UP.
12. Long notes keep their key held.
13. Dry-run mode produces a complete event timeline.
14. Real keyboard mode uses Windows SendInput.
15. Keyboard state is protected against duplicate/invalid events.
16. Emergency stop releases all held keys.
17. Recordings can be replayed without real input.
18. Core logic has automated tests.
19. Configuration/calibration is persistent.
20. No machine learning is required for the initial working
    implementation.

## 31. Engineering Rules

OpenCode must follow these rules:

-   Do not guess coordinates, colors, note dimensions, timing offsets,
    or trajectory shapes.
-   Derive environment-specific values through calibration or
    measurement.
-   Do not introduce ML unless classical CV demonstrably fails and the
    failure is documented.
-   Do not connect vision directly to keyboard input.
-   Do not use frame count as time.
-   Do not use `sleep()` as the only precision timing mechanism.
-   Do not hardcode the screenshot geometry.
-   Do not assume every purple object is a note.
-   Do not treat long notes as repeated short presses.
-   Do not merge simultaneous notes into one key.
-   Do not send real keyboard events during detector development.
-   Do not hide detection uncertainty.
-   Prefer deterministic algorithms.
-   Add tests before refactoring timing/state logic.
-   Keep platform-specific code isolated.
-   Preserve a dry-run mode permanently.

## 32. Required Initial Deliverable

The first implementation should NOT attempt the complete bot.

It must first deliver:

``` text
1. Screen capture
2. Calibration
3. Static-background filtering
4. Short-note candidate detection
5. Long-note candidate detection
6. Lane assignment
7. Debug overlay
8. Recording/replay
```

No real keyboard input in this milestone.

The next milestone can then add tracking and timing prediction.

## 33. Reference Visual

A reference screenshot is supplied with the project. It shows the
intended visual note system:

-   six keyboard lanes;
-   purple note heads;
-   long notes containing a same-color elongated rectangular/bar
    component;
-   curved lanes;
-   a visible hit area.

Treat the screenshot as a visual reference, not as a source of hardcoded
coordinates.

For robust development, collect a short gameplay recording and use it as
the primary test dataset.

## Target Window Management

The application MUST support selecting a specific target game window.

The system must NOT depend on a permanently hardcoded screen coordinate.

### Required behavior

1. Enumerate visible application windows.
2. Display a selectable list of window titles/process information.
3. Allow the user to select the target game window.
4. Store the selected window identifier for the current session.
5. Capture only the selected window's client/gameplay area.
6. Track the target window if it moves or changes position.
7. Recalculate the capture region when the window moves or resizes.
8. Detect when the target window is minimized, closed, inaccessible, or otherwise unavailable.
9. Stop automation safely when the target window is lost.
10. Never send real keyboard input if the target window is unavailable.

### Window-relative coordinates

All gameplay calibration must be stored relative to the target window/client area.

Do NOT store calibration as permanent absolute desktop coordinates.

Example:

```text
window client origin = (500, 300)

calibrated A hit point = (150, 420) relative to window

actual desktop position:
(650, 720)
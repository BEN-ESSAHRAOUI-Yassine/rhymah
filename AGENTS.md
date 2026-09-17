# AGENTS.md --- OpenCode Development Instructions

## Mission

You are implementing the Rhythm Game Keyboard Automation project
described in `Project.md`.

Your job is to build the system incrementally, test every subsystem, and
avoid guessing environment-specific values.

The application observes a rhythm-game screen, detects short/long notes,
predicts hit/release times, and eventually sends synchronized keyboard
events for:

```text
A S D J K L
```

## Read First

Before changing code:

1.  Read `Project.md` completely.
2.  Inspect the repository structure.
3.  Inspect existing tests.
4.  Inspect configuration files.
5.  Determine which phase is currently complete.
6.  Do not skip directly to real keyboard automation.

## Critical Architecture Rule

Never couple vision directly to keyboard input.

The required pipeline is:

```text
Capture
  -> Vision
  -> Tracking
  -> Timing
  -> Timeline
  -> Synchronization
  -> Scheduler
  -> Keyboard Driver
```

A detector must return data objects/events. It must never call the
keyboard driver.

## Development Method

Work phase-by-phase.

For every phase:

1.  State the implementation goal internally.
2.  Inspect relevant existing code.
3.  Implement the smallest coherent change.
4.  Add/update tests.
5.  Run tests.
6.  Run lint/type checks if configured.
7.  Run a local/manual diagnostic where applicable.
8.  Only then continue to the next phase.

Do not rewrite large portions of the project without a reason.

## Phase Order

Follow this exact high-level order:

```text
1. Repository/infrastructure
2. Screen capture
3. Calibration
4. Static-background filtering
5. Short-note detection
6. Long-note detection
7. Temporal tracking
8. Timing prediction
9. Event timeline
10. Virtual keyboard simulation
11. Windows SendInput
12. Precision scheduler
13. Real input integration
14. Replay/regression testing
15. Optimization
```

Real keyboard input is one of the final phases.

## Environment-Specific Values

Never guess:

- screen coordinates;
- lane coordinates;
- ROI dimensions;
- purple RGB values;
- HSV thresholds;
- note dimensions;
- long-note dimensions;
- timing offsets;
- synchronization tolerance;
- trajectory shape;
- game speed.

These must come from:

- calibration;
- recorded gameplay;
- measured observations;
- configurable parameters.

The supplied screenshot is a reference only.

## Vision Rules

Do not implement:

```python
if pixel_is_purple:
    note = True
```

Purple static lane graphics exist.

Use multiple signals:

```text
color
shape
motion
lane geometry
temporal persistence
```

Static graphics should be suppressed using temporal/background
information.

## Short Notes

Short notes should be detected using a combination of:

- contour;
- area;
- perimeter;
- circularity;
- color;
- motion;
- lane trajectory.

Do not rely solely on HoughCircles.

## Long Notes

Long notes have a circular head and an elongated same-color body/bar.

The classifier should consider:

- head;
- body;
- connectivity;
- aspect ratio;
- contour geometry;
- body length;
- body width;
- orientation;
- movement;
- lane trajectory.

Never classify long notes using only one hardcoded pixel dimension.

A long note means:

```text
KEY_DOWN at predicted head hit
KEY_UP at predicted tail/release
```

It does NOT mean repeated key presses.

## Lane Detection

The six lanes are:

```text
A S D J K L
```

The paths are curved.

Do not map a note to a lane using only its screen X coordinate.

Use calibrated lane trajectories.

## Tracking

A note seen in multiple frames must retain one track ID.

At minimum track:

```text
id
lane
type
position
velocity
progress
first_seen
last_seen
confidence
```

Use temporal association.

A first implementation may use nearest-neighbor matching with prediction
and gating.

Do not add a Kalman filter unless the simpler method is insufficient.

## Timing

Use monotonic high-resolution timestamps.

Use:

```python
time.perf_counter()
```

or a stronger Windows high-resolution timing source where appropriate.

Never use frame number as authoritative time.

Never make `time.sleep()` the only precision mechanism.

Timing prediction should use multiple observations and smoothing.

## Long-Note Timing

For:

```text
S LONG
```

produce:

```text
S DOWN
...
S UP
```

The key must remain logically held during the entire interval.

The state machine must prevent duplicate DOWN events.

## Synchronization

If multiple notes have predicted times within the configured
synchronization window, group them.

Example:

```text
D = 10.250
J = 10.253
```

should become a synchronized group.

The default synchronization window is configurable and must not be
treated as universally correct.

## Event Model

Use absolute timestamps.

Example:

```text
10.200 A DOWN
10.250 D DOWN
10.250 J DOWN
10.330 A UP
10.800 D UP
10.800 J UP
```

The scheduler consumes events.

The vision system does not.

## Keyboard Driver

Windows-specific keyboard code belongs in:

```text
app/input/keyboard.py
```

Use Windows `SendInput` through `ctypes`.

Expose a small interface:

```python
key_down(key)
key_up(key)
release_all()
```

Do not leak Windows API structures into unrelated modules.

## Keyboard Safety

Always maintain key state.

Rules:

- no duplicate DOWN;
- no invalid UP;
- long note = one DOWN + one UP;
- stop = release all;
- emergency stop = release all;
- application shutdown = release all.

The emergency key is:

```text
F8
```

It must stop scheduling and release held keys.

## Dry Run

Dry-run mode must remain available permanently.

In dry-run:

- no real keyboard input;
- show predicted events;
- show simulated key states;
- show timing metrics.

Never accidentally send real input from a test.

## Real Input Gate

Real keyboard input must require an explicit configuration/state.

Suggested behavior:

```text
STOPPED -> READY -> DRY_RUN -> RUNNING
```

Real input is only permitted in:

```text
RUNNING
```

The application must not enter RUNNING accidentally.

## Recording and Replay

Implement replay as early as practical.

A recording should preserve:

- frames/video;
- timestamps;
- calibration;
- relevant configuration.

The detector must be testable repeatedly on the same recording.

This is essential for regression testing.

## Debug Overlay

During development, expose:

- ROI;
- lanes;
- hit points;
- candidates;
- tracks;
- IDs;
- note type;
- confidence;
- predicted hit;
- predicted release;
- synchronization groups.

If detection is wrong, the developer must be able to see why.

## Confidence

Every note should have a confidence score.

Do not silently convert uncertain detections into real keyboard events.

If confidence is below the configured threshold:

- retain it for diagnostics;
- do not schedule real input.

## Error Handling

Stop automation if:

- capture is lost;
- calibration is invalid;
- vision becomes unreliable;
- scheduler fails;
- keyboard driver fails;
- event state becomes inconsistent.

Never continue sending keys after a fatal subsystem error.

## Testing

Every new core feature requires tests.

Prioritize tests for:

```text
Note model
Long/short classification
Lane assignment
Tracker
Timing predictor
Synchronization
Keyboard state machine
Event ordering
Scheduler cancellation
Emergency stop
Release-all
```

Run:

```text
pytest
```

after meaningful changes.

Do not declare success based only on compilation.

## Performance

Measure before optimizing.

Track:

```text
capture FPS
processing FPS
frame latency
detection latency
prediction latency
scheduler error
CPU usage
```

Do not prematurely introduce complex optimization.

## Dependencies

Keep dependencies minimal.

Preferred initial stack:

```text
opencv-python
numpy
mss
PySide6
PyYAML
pytest
```

Use additional dependencies only when justified.

Do not add an ML framework for the initial detector.

## Code Quality

Prefer:

- type hints;
- dataclasses for domain models;
- small functions;
- dependency injection for testable components;
- deterministic behavior;
- explicit state machines;
- structured logging;
- clear module boundaries.

Avoid:

- giant `main.py`;
- global mutable state;
- hidden background threads;
- direct keyboard calls from CV code;
- hardcoded coordinates;
- magic timing numbers.

## Threading

If using worker threads:

- define ownership of state;
- avoid unsafe shared mutable state;
- use queues/events for communication;
- never let a worker silently continue after shutdown;
- ensure keyboard scheduler shutdown releases all keys.

The UI thread must remain responsive.

## Configuration

Use YAML for user-adjustable parameters.

Separate:

```text
application defaults
user configuration
calibration profile
```

Do not bury configurable values in Python source.

## Logging

Log:

- detector results;
- tracking;
- prediction;
- scheduling;
- timing error;
- errors;
- state transitions.

Avoid excessive per-pixel/per-frame logs in normal mode.

Provide debug logging when needed.

## Do Not Do These Things

Do not:

- hardcode the supplied screenshot coordinates;
- assume one screen resolution;
- assume one exact purple RGB value;
- assume all purple objects are notes;
- press keys immediately upon detection;
- use PyAutoGUI for precision timing;
- use frame count as time;
- use only `sleep()` for timing;
- repeatedly press a long-note key;
- merge two simultaneous keys into one;
- skip dry-run;
- skip calibration;
- introduce YOLO before evaluating classical CV;
- add game-memory/process-injection techniques;
- bypass game security;
- hide low-confidence detections;
- claim accuracy without recorded test evidence.

## Current Implementation Strategy

When starting from an empty repository, implement only the first
milestone:

```text
1. Project structure
2. Config loader
3. Logging
4. Tests
5. Screen capture
6. Calibration
7. Static-background filtering
8. Candidate visualization
9. Short-note detection
10. Long-note detection
11. Lane assignment
12. Debug overlay
```

Do NOT implement real keyboard input in the first milestone.

The first milestone should end with a tool capable of visually showing:

```text
SHORT A
LONG S
SHORT D
SHORT J
LONG K
SHORT L
```

with confidence and track/candidate information where available.

## Completion Reporting

At the end of each implementation phase, report:

```text
Completed:
- ...

Tests:
- ...

Manual verification:
- ...

Known limitations:
- ...

Next phase:
- ...
```

Do not claim that the bot is accurate unless it has been tested against
actual recordings.

## Final Principle

Correctness comes before automation.

The project should first prove:

```text
"I can reliably understand what the game is showing."
```

Then prove:

```text
"I can predict when it will happen."
```

Then prove:

```text
"I can construct the correct keyboard timeline."
```

Only then enable:

```text
"I can send the keyboard timeline."
```

## Target Window Rule

The application must explicitly target a selected game window.

Do not implement the detector as a generic full-screen scanner.

The preferred flow is:

```text
Enumerate Windows
-> User Selects Game Window
-> Obtain Client Area
-> Capture Target Window
-> Detect Gameplay ROI
-> Run Vision Pipeline


**One further improvement:** I'd make the program automatically identify the **gameplay region inside the selected window** during calibration, rather than requiring you to manually configure screen coordinates. That would make the whole system much more robust.
```

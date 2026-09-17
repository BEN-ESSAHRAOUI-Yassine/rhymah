"""Demo: Run the full detection pipeline on synthetic frames."""
from __future__ import annotations

import time
import numpy as np
import cv2

from app.config.loader import load_config
from app.state import App, AppState
from app.calibration.profile import CalibrationProfile, LaneTrajectory, Point, ColorSample
from app.calibration.calibration import interpolate_trajectory
from app.vision.background import BackgroundModel
from app.vision.note_detector import ShortNoteDetector, DetectionConfig
from app.vision.note_classifier import LongNoteDetector
from app.vision.tracker import Tracker, TrackConfig, _Detection
from app.rhythm.timing import TimingEngine, TimingConfig
from app.rhythm.timeline import Timeline, TimelineConfig
from app.rhythm.synchronizer import Synchronizer, SyncConfig
from app.input.key_state import KeyboardStateMachine
from app.input.keyboard import KeyboardDriver
from app.input.scheduler import Scheduler
from app.automation import AutomationController
from app.performance import PerformanceTracker

LANES = ["A", "S", "D", "J", "K", "L"]
LANE_KEYS = {k: k for k in LANES}


def build_calibration() -> CalibrationProfile:
    trajectories = {}
    for i, lane in enumerate(LANES):
        x = 80 + i * 50
        trajectories[lane] = interpolate_trajectory(Point(x, 0), Point(x, 400), steps=20)

    return CalibrationProfile(
        name="demo",
        hit_points={lane: Point(80 + i * 50, 350) for i, lane in enumerate(LANES)},
        lane_trajectories=trajectories,
        note_color=ColorSample(h_min=140, h_max=170, s_min=80, s_max=255, v_min=80, v_max=255),
    )


def make_frame(
    background_color: int = 30,
    notes: list[tuple[int, int, str]] | None = None,
) -> np.ndarray:
    frame = np.full((400, 400, 3), background_color, dtype=np.uint8)

    for i, lane in enumerate(LANES):
        x = 80 + i * 50
        cv2.line(frame, (x, 0), (x, 400), (60, 30, 80), 2)

    if notes:
        for x, y, note_type in notes:
            if note_type == "SHORT":
                cv2.circle(frame, (x, y), 15, (180, 80, 200), -1)
            elif note_type == "LONG":
                cv2.circle(frame, (x, y), 15, (180, 80, 200), -1)
                cv2.rectangle(frame, (x - 8, y), (x + 8, y + 80), (180, 80, 200), -1)

    return frame


def run_demo():
    print("=" * 60)
    print("  Rhythm Game Keyboard Automation - Pipeline Demo")
    print("=" * 60)

    config = load_config()
    calibration = build_calibration()

    bg_model = BackgroundModel(history_size=10, motion_threshold=25)
    short_det = ShortNoteDetector(DetectionConfig(min_area=80, max_area=5000, min_circularity=0.3))
    long_det = LongNoteDetector(DetectionConfig(min_area=80, max_area=5000))
    tracker = Tracker(TrackConfig(max_distance=80, max_frames_missing=5))
    timing = TimingEngine(TimingConfig(smoothing_window=5))
    timeline = Timeline(TimelineConfig(synchronization_window_ms=8))
    sync = Synchronizer(SyncConfig(window_ms=8))
    key_state = KeyboardStateMachine()
    driver = KeyboardDriver(dry_run=True)
    scheduler = Scheduler(driver, key_state, dry_run=True)
    app = App()
    perf = PerformanceTracker()

    automation = AutomationController(app, driver, key_state, scheduler)

    print("\n[1] Building background model (no notes)...")
    for i in range(15):
        frame = make_frame(background_color=30, notes=None)
        bg_model.add_frame(frame)
    print(f"    Background ready: {bg_model.ready}")

    print("\n[2] Simulating gameplay with notes...")
    note_positions = [
        (80, 50, "SHORT"),
        (130, 30, "LONG"),
        (180, 60, "SHORT"),
        (280, 40, "SHORT"),
        (330, 55, "SHORT"),
    ]

    all_events = []
    for frame_num in range(20):
        ts = time.perf_counter()
        y_offset = frame_num * 15
        notes_at_frame = [(x, y + y_offset, t) for x, y, t in note_positions]
        frame_img = make_frame(notes=notes_at_frame)

        perf.tick_capture()

        t0 = time.perf_counter()
        mask = bg_model.filter_frame(frame_img, color_sample=calibration.note_color)
        t1 = time.perf_counter()
        perf.tick_detection((t1 - t0) * 1000)

        short_candidates = short_det.detect(mask, calibration.lane_trajectories)
        long_notes = long_det.detect(mask, short_candidates, calibration.lane_trajectories)

        detections = []
        for c in short_candidates:
            is_long = any(
                abs(c.center[0] - ln.head.center[0]) < 5 and abs(c.center[1] - ln.head.center[1]) < 5
                for ln in long_notes
            )
            if not is_long:
                detections.append(_Detection(
                    center=c.center,
                    confidence=c.confidence,
                    note_type=__import__("app.rhythm.note", fromlist=["NoteType"]).NoteType.SHORT,
                    lane=c.lane,
                ))

        for ln in long_notes:
            detections.append(_Detection(
                center=ln.head.center,
                confidence=ln.confidence,
                note_type=__import__("app.rhythm.note", fromlist=["NoteType"]).NoteType.LONG,
                lane=ln.lane,
            ))

        t2 = time.perf_counter()
        tracks = tracker.update(detections, ts)
        t3 = time.perf_counter()
        perf.tick_process((t3 - t2) * 1000)

        for track in tracks:
            if track.lane in calibration.lane_trajectories:
                traj = calibration.lane_trajectories[track.lane]
                t4 = time.perf_counter()
                est = timing.predict(track, traj, ts)
                t5 = time.perf_counter()
                perf.tick_prediction((t5 - t4) * 1000)
                timeline.add_estimate(est, LANE_KEYS)

    synced = sync.group(timeline.events)

    print(f"\n[3] Detection Results:")
    print(f"    Frames processed: 20")
    print(f"    Short candidates: {len(short_candidates)}")
    print(f"    Long notes: {len(long_notes)}")
    print(f"    Active tracks: {tracker.track_count}")

    print(f"\n[4] Event Timeline ({len(synced)} events):")
    for event in synced[:15]:
        sync_tag = f" [group {event.synchronization_group}]" if event.synchronization_group else ""
        print(f"    {event.timestamp:.3f}  {event.key:2s}  {event.action.value}{sync_tag}")
    if len(synced) > 15:
        print(f"    ... and {len(synced) - 15} more events")

    print(f"\n[5] Scheduling events...")
    automation.start_dry_run()
    scheduler.schedule(synced)
    time.sleep(0.5)
    automation.stop()

    print(f"\n[6] Performance Metrics:")
    snap = perf.snapshot()
    print(f"    Capture FPS: {snap.capture_fps:.1f}")
    print(f"    Detection latency: {snap.detection_latency_ms:.2f} ms")
    print(f"    Prediction latency: {snap.prediction_latency_ms:.2f} ms")

    print(f"\n[7] Key State:")
    for key, state in key_state.state.items():
        print(f"    {key}: {state.value}")

    print("\n" + "=" * 60)
    print("  Demo complete! All subsystems functional.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()

from __future__ import annotations

import json
import logging
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.capture.screen_capture import CaptureRegion, ScreenCapture
from app.capture.window_selector import WindowInfo, enumerate_windows, auto_detect_game
from app.calibration.calibration import create_color_mask, interpolate_trajectory
from app.calibration.profile import CalibrationProfile, Point
from app.config.loader import load_config
from app.input.keyboard import KeyboardDriver
from app.input.key_state import KeyboardStateMachine
from app.input.scheduler import Scheduler
from app.rhythm.note import KeyAction, KeyboardEvent, NoteType
from app.recording import Recorder, ReplayEngine
from app.state import App, AppState
from app.automation import AutomationController
from app.vision.background import BackgroundModel
from app.vision.note_detector import ShortNoteDetector, DetectionConfig
from app.vision.note_classifier import LongNoteCandidate, LongNoteDetector
from app.vision.tracker import Tracker, TrackConfig, _Detection
from app.rhythm.timing import TimingEngine, TimingConfig
from app.rhythm.timeline import Timeline, TimelineConfig
from app.rhythm.synchronizer import Synchronizer, SyncConfig
from app.overlay.debug_overlay import DebugOverlay, OverlayState
from app.performance import PerformanceTracker

logger = logging.getLogger(__name__)

LANES = ["A", "S", "D", "J", "K", "L"]
LANE_KEYS = {k: k for k in LANES}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Rhythm Game Keyboard Automation")
        self.setMinimumSize(1000, 650)

        self._config = load_config()
        self._app = App()
        self._driver = KeyboardDriver(dry_run=True)
        self._key_state = KeyboardStateMachine()
        self._scheduler = Scheduler(
            self._driver, self._key_state, dry_run=True,
            jitter_enabled=self._config.scheduler.jitter_enabled,
            jitter_min_ms=self._config.scheduler.jitter_min_ms,
            jitter_max_ms=self._config.scheduler.jitter_max_ms,
        )
        self._automation = AutomationController(
            self._app, self._driver, self._key_state, self._scheduler,
        )

        self._bg_model = BackgroundModel(
            history_size=self._config.vision.background_frames,
            motion_threshold=self._config.vision.motion_threshold,
        )
        self._short_det = ShortNoteDetector(DetectionConfig())
        self._long_det = LongNoteDetector(DetectionConfig())
        self._tracker = Tracker(TrackConfig())
        self._timing = TimingEngine(TimingConfig())
        self._timeline = Timeline(TimelineConfig(
            synchronization_window_ms=self._config.timing.synchronization_window_ms,
            short_note_duration_ms=self._config.timing.short_note_duration_ms,
        ))
        self._sync = Synchronizer(SyncConfig(
            window_ms=self._config.timing.synchronization_window_ms,
        ))
        self._overlay = DebugOverlay()
        self._perf = PerformanceTracker()

        self._capture: ScreenCapture | None = None
        self._selected_window: WindowInfo | None = None
        self._calibration: CalibrationProfile | None = None
        self._running = False
        self._frame_count = 0

        self._recorder: Recorder | None = None
        self._recording = False
        self._replay_engine: ReplayEngine | None = None
        self._replay_mode = False
        self._replay_paused = False
        self._replay_speed = 1.0
        self._analyzed_events: list[KeyboardEvent] = []
        self._consecutive_none_frames = 0
        self._max_none_frames = 30

        self._build_ui()
        self._refresh_windows()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._main_loop)
        self._timer_ms = 16  # ~60fps

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_panel = QVBoxLayout()

        # Window selector
        win_group = QGroupBox("Target Window")
        win_layout = QVBoxLayout()
        self._combo_windows = QComboBox()
        self._combo_windows.setMinimumWidth(250)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._refresh_windows)
        self._btn_select = QPushButton("Select Window")
        self._btn_select.clicked.connect(self._select_window)
        self._btn_auto_detect = QPushButton("Auto-Detect Game")
        self._btn_auto_detect.clicked.connect(self._auto_detect_game)
        self._btn_auto_detect.setStyleSheet("QPushButton { background-color: #2d5a27; color: white; }")
        win_layout.addWidget(self._combo_windows)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_select)
        win_layout.addLayout(btn_row)
        win_layout.addWidget(self._btn_auto_detect)
        win_group.setLayout(win_layout)
        left_panel.addWidget(win_group)

        # Calibration
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout()
        self._btn_calibrate = QPushButton("Calibrate")
        self._btn_calibrate.clicked.connect(self._start_calibration)
        self._btn_load_cal = QPushButton("Load Profile")
        self._btn_load_cal.clicked.connect(self._load_profile)
        self._btn_save_cal = QPushButton("Save Profile")
        self._btn_save_cal.clicked.connect(self._save_profile)
        cal_layout.addWidget(self._btn_calibrate)
        cal_layout.addWidget(self._btn_load_cal)
        cal_layout.addWidget(self._btn_save_cal)
        self._lbl_cal_status = QLabel("Not calibrated")
        cal_layout.addWidget(self._lbl_cal_status)
        cal_group.setLayout(cal_layout)
        left_panel.addWidget(cal_group)

        # Recording
        rec_group = QGroupBox("Recording (Safe Mode)")
        rec_layout = QVBoxLayout()
        self._btn_record = QPushButton("Start Recording")
        self._btn_record.clicked.connect(self._toggle_recording)
        self._btn_record.setStyleSheet("QPushButton { background-color: #8B0000; color: white; }")
        self._btn_load_rec = QPushButton("Load Recording")
        self._btn_load_rec.clicked.connect(self._load_recording)
        self._btn_analyze = QPushButton("Analyze Recording")
        self._btn_analyze.clicked.connect(self._analyze_recording)
        self._btn_analyze.setEnabled(False)
        self._btn_export = QPushButton("Export Timeline")
        self._btn_export.clicked.connect(self._export_timeline)
        self._btn_export.setEnabled(False)
        self._lbl_rec_status = QLabel("No recording")
        rec_layout.addWidget(self._btn_record)
        rec_layout.addWidget(self._btn_load_rec)
        rec_layout.addWidget(self._btn_analyze)
        rec_layout.addWidget(self._btn_export)
        rec_layout.addWidget(self._lbl_rec_status)
        rec_group.setLayout(rec_layout)
        left_panel.addWidget(rec_group)

        # Replay controls
        replay_group = QGroupBox("Replay Controls")
        replay_layout = QVBoxLayout()
        self._btn_replay_start = QPushButton("Start Replay (F9)")
        self._btn_replay_start.clicked.connect(self._start_replay)
        self._btn_replay_start.setEnabled(False)
        self._btn_replay_pause = QPushButton("Pause/Resume")
        self._btn_replay_pause.clicked.connect(self._toggle_replay_pause)
        self._btn_replay_pause.setEnabled(False)
        self._btn_replay_stop = QPushButton("Stop Replay")
        self._btn_replay_stop.clicked.connect(self._stop_replay)
        self._btn_replay_stop.setEnabled(False)
        self._lbl_replay_progress = QLabel("Replay: --")
        replay_layout.addWidget(self._btn_replay_start)
        replay_layout.addWidget(self._btn_replay_pause)
        replay_layout.addWidget(self._btn_replay_stop)
        replay_layout.addWidget(self._lbl_replay_progress)
        replay_group.setLayout(replay_layout)
        left_panel.addWidget(replay_group)

        # Controls
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()
        self._btn_dry_run = QPushButton("Start Dry Run")
        self._btn_dry_run.clicked.connect(self._toggle_dry_run)
        self._btn_real = QPushButton("Start Real Input")
        self._btn_real.clicked.connect(self._toggle_real)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_stop.setEnabled(False)
        self._btn_emergency = QPushButton("EMERGENCY STOP (F8)")
        self._btn_emergency.setStyleSheet("QPushButton { background-color: #cc0000; color: white; font-weight: bold; }")
        self._btn_emergency.clicked.connect(self._emergency_stop)
        ctrl_layout.addWidget(self._btn_dry_run)
        ctrl_layout.addWidget(self._btn_real)
        ctrl_layout.addWidget(self._btn_stop)
        ctrl_layout.addWidget(self._btn_emergency)
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)

        # Jitter toggle
        jitter_group = QGroupBox("Jitter")
        jitter_layout = QVBoxLayout()
        self._chk_jitter = QCheckBox("Enable timing jitter")
        self._chk_jitter.setChecked(self._config.scheduler.jitter_enabled)
        self._chk_jitter.toggled.connect(self._on_jitter_toggled)
        self._lbl_jitter_range = QLabel(
            f"Range: {self._config.scheduler.jitter_min_ms:.0f}-{self._config.scheduler.jitter_max_ms:.0f} ms"
        )
        jitter_layout.addWidget(self._chk_jitter)
        jitter_layout.addWidget(self._lbl_jitter_range)
        jitter_group.setLayout(jitter_layout)
        left_panel.addWidget(jitter_group)

        # Debug toggle
        self._btn_overlay = QPushButton("Toggle Overlay")
        self._btn_overlay.clicked.connect(lambda: setattr(self._overlay, 'visible', not self._overlay.visible))
        left_panel.addWidget(self._btn_overlay)

        # Status
        self._lbl_status = QLabel("State: STOPPED")
        self._lbl_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_panel.addWidget(self._lbl_status)

        self._lbl_fps = QLabel("FPS: 0")
        left_panel.addWidget(self._lbl_fps)
        self._lbl_tracks = QLabel("Tracks: 0")
        left_panel.addWidget(self._lbl_tracks)
        self._lbl_events = QLabel("Events: 0")
        left_panel.addWidget(self._lbl_events)

        left_panel.addStretch()

        # Video display
        self._video_label = QLabel("Select a window and calibrate to start")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setStyleSheet("background-color: #1a1a1a;")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(320)
        splitter.addWidget(left_widget)
        splitter.addWidget(self._video_label)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _refresh_windows(self) -> None:
        self._combo_windows.clear()
        windows = enumerate_windows()
        for w in windows:
            if w.width > 100 and w.height > 100:
                self._combo_windows.addItem(f"{w.title} ({w.width}x{w.height})", w.handle)

    def _auto_detect_game(self) -> None:
        self._lbl_status.setText("Searching for game...")
        QApplication.processEvents()

        window = auto_detect_game()
        if window is None:
            self._lbl_status.setText("Game not found. Try running the game in windowed mode.")
            QMessageBox.information(
                self, "Not Found",
                "Could not auto-detect the game.\n\n"
                "Make sure the game is running in windowed mode.\n"
                "Try: Refresh → manual selection from dropdown.",
            )
            return

        self._selected_window = window
        self._lbl_status.setText(f"Detected: {window.title[:40]}")
        logger.info("Auto-detected window: %s (%dx%d, class=%s)",
                     window.title, window.width, window.height, window.class_name)
        self._start_capture()

    def _select_window(self) -> None:
        idx = self._combo_windows.currentIndex()
        if idx < 0:
            return

        handle = self._combo_windows.currentData()
        windows = enumerate_windows()
        for w in windows:
            if w.handle == handle:
                self._selected_window = w
                self._lbl_status.setText(f"Selected: {w.title[:40]}")
                logger.info("Selected window: %s (%dx%d)", w.title, w.width, w.height)
                self._start_capture()
                break

    def _start_capture(self) -> None:
        if self._selected_window is None:
            return

        if self._capture:
            self._capture.close()

        left, top, right, bottom = self._selected_window.client_rect
        region = CaptureRegion(left=left, top=top, width=right - left, height=bottom - top)
        self._capture = ScreenCapture(region=region)
        self._bg_model.reset()
        self._frame_count = 0
        self._consecutive_none_frames = 0
        self._lbl_status.setText(f"Capturing: {self._selected_window.title[:30]}")
        logger.info("Capture started: %dx%d at (%d,%d)", region.width, region.height, left, top)

    def _start_calibration(self) -> None:
        if self._capture is None:
            QMessageBox.warning(self, "No Capture", "Select a window first.")
            return

        frame = self._capture.grab()
        if frame is None:
            QMessageBox.warning(self, "Capture Failed", "Could not capture frame.")
            return

        from app.ui.calibration_widget import CalibrationWidget
        self._cal_widget = CalibrationWidget()
        self._cal_widget.set_frame(frame.image)
        self._cal_widget.calibration_complete.connect(self._on_calibration_done)
        self._cal_widget.cancelled.connect(lambda: self._cal_widget.close())
        self._cal_widget.setWindowTitle("Calibration")
        self._cal_widget.resize(700, 600)
        self._cal_widget.show()

    def _on_calibration_done(self, profile: CalibrationProfile) -> None:
        self._calibration = profile
        self._lbl_cal_status.setText(f"Calibrated: {profile.name}")
        self._bg_model.reset()
        logger.info("Calibration complete: %s", profile.name)
        if self._cal_widget:
            self._cal_widget.close()

    def _load_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "config/profiles", "YAML files (*.yaml)"
        )
        if path:
            try:
                self._calibration = CalibrationProfile.load(path)
                self._lbl_cal_status.setText(f"Loaded: {self._calibration.name}")
                self._bg_model.reset()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _save_profile(self) -> None:
        if self._calibration is None:
            QMessageBox.warning(self, "No Calibration", "Calibrate first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "config/profiles/default.yaml", "YAML files (*.yaml)"
        )
        if path:
            try:
                self._calibration.save(path)
                self._lbl_cal_status.setText(f"Saved: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
            return

        if self._capture is None:
            QMessageBox.warning(self, "No Capture", "Select a window first.")
            return

        import os
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rec_dir = os.path.join("recordings", f"rec_{timestamp}")
        self._recorder = Recorder(rec_dir)
        frame = self._capture.grab()
        if frame is None:
            QMessageBox.warning(self, "Capture Failed", "Could not capture frame.")
            return
        self._recorder.start(frame.image.shape[1], frame.image.shape[0])
        self._recording = True
        self._btn_record.setText("Stop Recording")
        self._btn_record.setStyleSheet("QPushButton { background-color: #ff4444; color: white; }")
        self._lbl_rec_status.setText("Recording...")
        self._lbl_status.setText("RECORDING")
        self._lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
        logger.info("Recording started: %s", rec_dir)

    def _stop_recording(self) -> None:
        if not self._recording or self._recorder is None:
            return

        path = self._recorder.save(calibration=self._calibration)
        self._recording = False
        self._btn_record.setText("Start Recording")
        self._btn_record.setStyleSheet("QPushButton { background-color: #8B0000; color: white; }")
        self._lbl_rec_status.setText(f"Saved: {path.name} ({self._recorder.frame_count} frames)")
        self._lbl_status.setText(f"State: {self._app.state.value}")
        self._lbl_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._last_recording_path = str(path)
        self._btn_analyze.setEnabled(True)
        logger.info("Recording saved: %s", path)

    def _load_recording(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self, "Load Recording", "recordings",
        )
        if not dir_path:
            return

        engine = ReplayEngine(dir_path)
        if not engine.load():
            QMessageBox.critical(self, "Error", "Could not load recording.")
            return

        self._replay_engine = engine
        cal = engine.load_calibration()
        if cal:
            self._calibration = cal
            self._lbl_cal_status.setText(f"Loaded from recording: {cal.name}")
            self._bg_model.reset()

        self._lbl_rec_status.setText(
            f"Loaded: {engine.total_frames} frames, {engine.meta.duration_s:.1f}s"
        )
        self._btn_analyze.setEnabled(True)
        self._last_recording_path = dir_path
        logger.info("Recording loaded: %s", dir_path)

    def _analyze_recording(self) -> None:
        if self._replay_engine is None and not hasattr(self, '_last_recording_path'):
            QMessageBox.warning(self, "No Recording", "Load or record a video first.")
            return

        if self._replay_engine is None:
            self._replay_engine = ReplayEngine(self._last_recording_path)
            if not self._replay_engine.load():
                QMessageBox.critical(self, "Error", "Could not load recording.")
                return
            cal = self._replay_engine.load_calibration()
            if cal:
                self._calibration = cal
                self._bg_model.reset()

        if self._calibration is None:
            QMessageBox.warning(self, "No Calibration", "Calibrate or load a profile first.")
            return

        self._lbl_status.setText("Analyzing recording...")
        self._lbl_rec_status.setText("Analyzing...")
        QApplication.processEvents()

        self._analyzed_events.clear()
        self._tracker.reset()
        self._bg_model.reset()
        self._replay_engine.reset()

        frame_count = 0
        while True:
            frame = self._replay_engine.get_next()
            if frame is None:
                break

            if self._calibration.note_color:
                mask = self._bg_model.filter_frame(frame.image, self._calibration.note_color)
            else:
                mask = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

            short_candidates = self._short_det.detect(mask, self._calibration.lane_trajectories)
            long_notes = self._long_det.detect(mask, short_candidates, self._calibration.lane_trajectories)

            detections = []
            for c in short_candidates:
                is_long = any(
                    abs(c.center[0] - ln.head.center[0]) < 5 and abs(c.center[1] - ln.head.center[1]) < 5
                    for ln in long_notes
                )
                if not is_long:
                    detections.append(_Detection(
                        center=c.center, confidence=c.confidence,
                        note_type=NoteType.SHORT, lane=c.lane,
                    ))
            for ln in long_notes:
                detections.append(_Detection(
                    center=ln.head.center, confidence=ln.confidence,
                    note_type=NoteType.LONG, lane=ln.lane,
                ))

            tracks = self._tracker.update(detections, frame.timestamp)

            self._timeline.clear()
            for track in tracks:
                if track.lane in self._calibration.lane_trajectories:
                    traj = self._calibration.lane_trajectories[track.lane]
                    est = self._timing.predict(track, traj, frame.timestamp)
                    self._timeline.add_estimate(est, LANE_KEYS)

            synced = self._sync.group(self._timeline.events)
            self._analyzed_events.extend(synced)
            frame_count += 1

            if frame_count % 50 == 0:
                self._lbl_rec_status.setText(f"Analyzing: {frame_count} frames...")
                QApplication.processEvents()

        self._analyzed_events.sort(key=lambda e: e.timestamp)
        self._btn_export.setEnabled(True)
        self._btn_replay_start.setEnabled(True)
        self._lbl_rec_status.setText(
            f"Done: {len(self._analyzed_events)} events from {frame_count} frames"
        )
        self._lbl_status.setText("Analysis complete")
        logger.info("Analysis complete: %d events from %d frames",
                     len(self._analyzed_events), frame_count)

    def _export_timeline(self) -> None:
        if not self._analyzed_events:
            QMessageBox.warning(self, "No Events", "Analyze a recording first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Timeline", "timeline.json", "JSON files (*.json)"
        )
        if not path:
            return

        data = {
            "events": [e.to_dict() for e in self._analyzed_events],
            "total_events": len(self._analyzed_events),
            "duration_s": self._analyzed_events[-1].timestamp - self._analyzed_events[0].timestamp
                if self._analyzed_events else 0,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self._lbl_rec_status.setText(f"Exported: {Path(path).name}")
        logger.info("Timeline exported: %s", path)

    def _start_replay(self) -> None:
        if not self._analyzed_events:
            QMessageBox.warning(self, "No Events", "Analyze a recording first.")
            return

        reply = QMessageBox.question(
            self, "Start Replay",
            f"Will send {len(self._analyzed_events)} keyboard events.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._app.transition(AppState.READY)
        self._automation.start_real()
        self._replay_mode = True
        self._replay_paused = False
        self._replay_start_time = time.perf_counter()
        self._replay_event_index = 0
        self._running = True
        self._timer.start(self._timer_ms)
        self._update_buttons()
        self._lbl_status.setText("REPLAY: Running")
        logger.info("Replay started: %d events", len(self._analyzed_events))

    def _toggle_replay_pause(self) -> None:
        if not self._replay_mode:
            return
        self._replay_paused = not self._replay_paused
        if self._replay_paused:
            self._lbl_status.setText("REPLAY: Paused")
        else:
            self._lbl_status.setText("REPLAY: Running")

    def _stop_replay(self) -> None:
        self._replay_mode = False
        self._replay_paused = False
        self._stop()

    def _toggle_dry_run(self) -> None:
        if self._capture is None and not self._replay_mode:
            QMessageBox.warning(self, "No Capture", "Select a window first.")
            return
        if self._calibration is None:
            QMessageBox.warning(self, "No Calibration", "Calibrate first.")
            return

        if self._running:
            self._stop()
            return

        self._app.transition(AppState.READY)
        self._automation.start_dry_run()
        self._running = True
        self._timer.start(self._timer_ms)
        self._update_buttons()
        self._lbl_status.setText("Running: DRY RUN")

    def _toggle_real(self) -> None:
        if self._capture is None and not self._replay_mode:
            QMessageBox.warning(self, "No Capture", "Select a window first.")
            return
        if self._calibration is None:
            QMessageBox.warning(self, "No Calibration", "Calibrate first.")
            return

        reply = QMessageBox.question(
            self, "Real Input",
            "This will send REAL keyboard input to the game.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self._running:
            self._stop()
            return

        self._app.transition(AppState.READY)
        self._automation.start_real()
        self._running = True
        self._timer.start(self._timer_ms)
        self._update_buttons()
        self._lbl_status.setText("Running: REAL INPUT")

    def _stop(self) -> None:
        self._timer.stop()
        self._automation.stop()
        self._running = False
        self._replay_mode = False
        self._replay_paused = False
        self._update_buttons()
        self._lbl_status.setText(f"State: {self._app.state.value}")

    def _emergency_stop(self) -> None:
        self._timer.stop()
        self._automation.emergency_stop()
        self._running = False
        self._replay_mode = False
        self._replay_paused = False
        self._update_buttons()
        self._lbl_status.setText("EMERGENCY STOP")
        self._lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")

    def _on_jitter_toggled(self, checked: bool) -> None:
        self._scheduler.configure_jitter(
            enabled=checked,
            min_ms=self._config.scheduler.jitter_min_ms,
            max_ms=self._config.scheduler.jitter_max_ms,
        )

    def _update_buttons(self) -> None:
        active = self._running
        self._btn_dry_run.setEnabled(not active)
        self._btn_real.setEnabled(not active)
        self._btn_stop.setEnabled(active)
        self._btn_calibrate.setEnabled(not active)
        self._btn_replay_pause.setEnabled(active and self._replay_mode)
        self._btn_replay_stop.setEnabled(active and self._replay_mode)
        if not active:
            self._lbl_status.setStyleSheet("font-weight: bold; font-size: 14px;")

    def _main_loop(self) -> None:
        if self._replay_mode:
            self._replay_loop()
            return

        if self._capture is None or self._calibration is None:
            return

        t_start = time.perf_counter()
        self._perf.tick_capture()

        self._selected_window.refresh_rect()
        left, top, right, bottom = self._selected_window.client_rect
        new_w = right - left
        new_h = bottom - top
        if new_w > 0 and new_h > 0 and (new_w != self._capture._region.width or new_h != self._capture._region.height):
            self._capture._region = CaptureRegion(left=left, top=top, width=new_w, height=new_h)

        frame = self._capture.grab()
        if frame is None:
            self._consecutive_none_frames += 1
            if self._consecutive_none_frames >= self._max_none_frames:
                self._timer.stop()
                self._automation.stop()
                self._running = False
                self._update_buttons()
                self._lbl_status.setText("LOST: Window unavailable")
                self._lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
                QMessageBox.warning(
                    self, "Window Lost",
                    "The game window is no longer accessible.\nAutomation stopped.",
                )
            return

        self._consecutive_none_frames = 0

        if self._recording and self._recorder is not None:
            self._recorder.capture_frame(frame)

        self._process_frame(frame, t_start)

    def _replay_loop(self) -> None:
        if self._replay_paused:
            return

        if not self._analyzed_events or self._replay_event_index >= len(self._analyzed_events):
            self._stop_replay()
            self._lbl_status.setText("Replay complete")
            return

        elapsed = time.perf_counter() - self._replay_start_time

        while (self._replay_event_index < len(self._analyzed_events) and
               self._analyzed_events[self._replay_event_index].timestamp <= elapsed):
            event = self._analyzed_events[self._replay_event_index]
            self._scheduler.schedule([event])
            self._replay_event_index += 1

        progress = self._replay_event_index / len(self._analyzed_events) * 100
        self._lbl_replay_progress.setText(
            f"Replay: {self._replay_event_index}/{len(self._analyzed_events)} ({progress:.0f}%)"
        )

    def _process_frame(self, frame, t_start: float) -> None:
        t0 = time.perf_counter()
        mask = self._bg_model.filter_frame(frame.image, self._calibration.note_color)
        t1 = time.perf_counter()
        self._perf.tick_detection((t1 - t0) * 1000)

        short_candidates = self._short_det.detect(mask, self._calibration.lane_trajectories)
        long_notes = self._long_det.detect(mask, short_candidates, self._calibration.lane_trajectories)

        detections = []
        for c in short_candidates:
            is_long = any(
                abs(c.center[0] - ln.head.center[0]) < 5 and abs(c.center[1] - ln.head.center[1]) < 5
                for ln in long_notes
            )
            if not is_long:
                detections.append(_Detection(
                    center=c.center, confidence=c.confidence,
                    note_type=NoteType.SHORT, lane=c.lane,
                ))
        for ln in long_notes:
            detections.append(_Detection(
                center=ln.head.center, confidence=ln.confidence,
                note_type=NoteType.LONG, lane=ln.lane,
            ))

        t2 = time.perf_counter()
        tracks = self._tracker.update(detections, frame.timestamp)
        t3 = time.perf_counter()
        self._perf.tick_process((t3 - t2) * 1000)

        self._timeline.clear()
        for track in tracks:
            if track.lane in self._calibration.lane_trajectories:
                traj = self._calibration.lane_trajectories[track.lane]
                t4 = time.perf_counter()
                est = self._timing.predict(track, traj, frame.timestamp)
                t5 = time.perf_counter()
                self._perf.tick_prediction((t5 - t4) * 1000)
                self._timeline.add_estimate(est, LANE_KEYS)

        synced = self._sync.group(self._timeline.events)

        min_conf = self._config.vision.minimum_note_confidence
        filtered = [e for e in synced if e.confidence >= min_conf]

        if self._automation.is_active and filtered:
            self._scheduler.schedule(filtered)

        t_end = time.perf_counter()
        self._perf.tick_capture()

        overlay_state = OverlayState(
            candidates=short_candidates,
            long_notes=long_notes,
            tracks=tracks,
            events=filtered[:6],
            key_state={k: self._key_state.is_down(k) for k in LANES},
            fps=self._perf.capture_fps,
            detection_latency_ms=(t1 - t0) * 1000,
        )

        display = self._overlay.render(frame.image, self._calibration, overlay_state)
        self._display_frame(display)

        self._frame_count += 1
        if self._frame_count % 10 == 0:
            self._lbl_fps.setText(f"FPS: {self._perf.capture_fps:.0f}")
            self._lbl_tracks.setText(f"Tracks: {len(tracks)}")
            self._lbl_events.setText(f"Events: {len(filtered)}")

    def _display_frame(self, frame: np.ndarray) -> None:
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._automation.emergency_stop()
        if self._capture:
            self._capture.close()
        event.accept()

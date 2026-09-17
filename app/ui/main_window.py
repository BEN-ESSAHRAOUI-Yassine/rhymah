from __future__ import annotations

import logging
import time
import threading

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.capture.screen_capture import CaptureRegion, ScreenCapture
from app.capture.window_selector import WindowInfo, enumerate_windows
from app.calibration.calibration import create_color_mask, interpolate_trajectory
from app.calibration.profile import CalibrationProfile, Point
from app.config.loader import load_config
from app.input.keyboard import KeyboardDriver
from app.input.key_state import KeyboardStateMachine
from app.input.scheduler import Scheduler
from app.rhythm.note import KeyAction, NoteType
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
        self.setMinimumSize(900, 600)

        self._config = load_config()
        self._app = App()
        self._driver = KeyboardDriver(dry_run=True)
        self._key_state = KeyboardStateMachine()
        self._scheduler = Scheduler(self._driver, self._key_state, dry_run=True)
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
        right_panel = QVBoxLayout()

        # Window selector
        win_group = QGroupBox("Target Window")
        win_layout = QVBoxLayout()
        self._combo_windows = QComboBox()
        self._combo_windows.setMinimumWidth(250)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._refresh_windows)
        self._btn_select = QPushButton("Select Window")
        self._btn_select.clicked.connect(self._select_window)
        win_layout.addWidget(self._combo_windows)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_select)
        win_layout.addLayout(btn_row)
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
        left_widget.setMaximumWidth(300)
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
        from PySide6.QtWidgets import QFileDialog
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

        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "config/profiles/default.yaml", "YAML files (*.yaml)"
        )
        if path:
            try:
                self._calibration.save(path)
                self._lbl_cal_status.setText(f"Saved: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _toggle_dry_run(self) -> None:
        if self._capture is None:
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
        if self._capture is None:
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
        self._update_buttons()
        self._lbl_status.setText(f"State: {self._app.state.value}")

    def _emergency_stop(self) -> None:
        self._timer.stop()
        self._automation.emergency_stop()
        self._running = False
        self._update_buttons()
        self._lbl_status.setText("EMERGENCY STOP")
        self._lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")

    def _update_buttons(self) -> None:
        active = self._running
        self._btn_dry_run.setEnabled(not active)
        self._btn_real.setEnabled(not active)
        self._btn_stop.setEnabled(active)
        self._btn_calibrate.setEnabled(not active)
        if not active:
            self._lbl_status.setStyleSheet("font-weight: bold; font-size: 14px;")

    def _main_loop(self) -> None:
        if self._capture is None or self._calibration is None:
            return

        t_start = time.perf_counter()
        self._perf.tick_capture()

        frame = self._capture.grab()
        if frame is None:
            return

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
                    center=c.center,
                    confidence=c.confidence,
                    note_type=NoteType.SHORT,
                    lane=c.lane,
                ))
        for ln in long_notes:
            detections.append(_Detection(
                center=ln.head.center,
                confidence=ln.confidence,
                note_type=NoteType.LONG,
                lane=ln.lane,
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

        if self._automation.is_active and synced:
            self._scheduler.schedule(synced)

        t_end = time.perf_counter()
        self._perf.tick_capture()

        overlay_state = OverlayState(
            candidates=short_candidates,
            long_notes=long_notes,
            tracks=tracks,
            events=synced[:6],
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
            self._lbl_events.setText(f"Events: {len(synced)}")

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

from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.calibration.calibration import interpolate_trajectory, sample_color_from_region
from app.calibration.profile import CalibrationProfile, ColorSample, Point

logger = logging.getLogger(__name__)

LANE_KEYS = ["A", "S", "D", "J", "K", "L"]


class CalibStep(Enum):
    NONE = "none"
    ROI_TOP_LEFT = "roi_top_left"
    ROI_BOTTOM_RIGHT = "roi_bottom_right"
    HIT_POINTS = "hit_points"
    COLOR_SAMPLE = "color_sample"
    DONE = "done"


class CalibrationWidget(QWidget):
    calibration_complete = Signal(CalibrationProfile)
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._profile = CalibrationProfile(name="default")
        self._step = CalibStep.NONE
        self._hit_point_index = 0
        self._roi_p1: tuple[int, int] | None = None

        self._image_label = QLabel("No frame loaded")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(400, 400)
        self._image_label.mousePressEvent = self._on_click

        self._status_label = QLabel("Step: Not started")
        self._btn_start = QPushButton("Start Calibration (ROI)")
        self._btn_color = QPushButton("Sample Color")
        self._btn_done = QPushButton("Done")
        self._btn_cancel = QPushButton("Cancel")

        self._btn_start.clicked.connect(self._start_roi)
        self._btn_color.clicked.connect(self._start_color)
        self._btn_done.clicked.connect(self._finish)
        self._btn_cancel.clicked.connect(self.cancelled.emit)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self._btn_start)
        btn_layout.addWidget(self._btn_color)
        btn_layout.addWidget(self._btn_done)
        btn_layout.addWidget(self._btn_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status_label)
        layout.addWidget(self._image_label)
        layout.addLayout(btn_layout)

        self._update_status()

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame.copy()
        self._display_frame()

    def _display_frame(self) -> None:
        if self._frame is None:
            return
        display = self._frame.copy()
        self._draw_annotations(display)

        h, w, ch = display.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def _draw_annotations(self, frame: np.ndarray) -> None:
        cal = self._profile

        if cal.roi and cal.roi_size:
            x, y = int(cal.roi.x), int(cal.roi.y)
            w, h = int(cal.roi_size.x), int(cal.roi_size.y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        for i, lane in enumerate(LANE_KEYS):
            if lane in cal.hit_points:
                pt = cal.hit_points[lane]
                cv2.drawMarker(frame, (int(pt.x), int(pt.y)), (0, 255, 0),
                               cv2.MARKER_CROSS, 16, 2)
                cv2.putText(frame, lane, (int(pt.x) - 5, int(pt.y) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for lane, traj in cal.lane_trajectories.items():
            pts = [(int(p.x), int(p.y)) for p in traj.points]
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j + 1], (200, 200, 200), 1)

        if cal.note_color:
            c = cal.note_color
            cv2.putText(frame, f"Color: H={c.h_min}-{c.h_max} S={c.s_min}-{c.s_max}",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def _start_roi(self) -> None:
        self._step = CalibStep.ROI_TOP_LEFT
        self._roi_p1 = None
        self._update_status()

    def _start_color(self) -> None:
        if self._frame is None:
            return
        self._step = CalibStep.COLOR_SAMPLE
        self._update_status()

    def _finish(self) -> None:
        self._step = CalibStep.DONE
        self.calibration_complete.emit(self._profile)

    def _on_click(self, event) -> None:
        if self._frame is None:
            return

        label = self._image_label
        pixmap = label.pixmap()
        if pixmap is None:
            return

        scale_x = self._frame.shape[1] / pixmap.width()
        scale_y = self._frame.shape[0] / pixmap.height()

        x = int(event.position().x() * scale_x)
        y = int(event.position().y() * scale_y)

        x = max(0, min(x, self._frame.shape[1] - 1))
        y = max(0, min(y, self._frame.shape[0] - 1))

        if self._step == CalibStep.ROI_TOP_LEFT:
            self._roi_p1 = (x, y)
            self._step = CalibStep.ROI_BOTTOM_RIGHT
            self._status_label.setText(f"ROI Step 2: Click bottom-right corner (first: {x}, {y})")
            return

        if self._step == CalibStep.ROI_BOTTOM_RIGHT and self._roi_p1 is not None:
            rx = min(self._roi_p1[0], x)
            ry = min(self._roi_p1[1], y)
            rw = abs(x - self._roi_p1[0])
            rh = abs(y - self._roi_p1[1])

            self._profile.roi = Point(x=float(rx), y=float(ry))
            self._profile.roi_size = Point(x=float(rw), y=float(rh))

            for i, lane in enumerate(LANE_KEYS):
                lx = rx + int((i + 0.5) * rw / len(LANE_KEYS))
                ly = ry + int(rh * 0.85)
                self._profile.hit_points[lane] = Point(x=float(lx), y=float(ly))

            for lane in LANE_KEYS:
                if lane in self._profile.hit_points:
                    hp = self._profile.hit_points[lane]
                    self._profile.lane_trajectories[lane] = interpolate_trajectory(
                        Point(x=hp.x, y=float(ry)),
                        Point(x=hp.x, y=float(ry + rh)),
                        steps=20,
                    )

            self._step = CalibStep.HIT_POINTS
            self._hit_point_index = 0
            self._update_status()
            self._display_frame()
            return

        if self._step == CalibStep.HIT_POINTS:
            if self._hit_point_index < len(LANE_KEYS):
                lane = LANE_KEYS[self._hit_point_index]
                self._profile.hit_points[lane] = Point(x=float(x), y=float(y))

                if self._profile.roi and self._profile.roi_size:
                    self._profile.lane_trajectories[lane] = interpolate_trajectory(
                        Point(x=float(x), y=self._profile.roi.y),
                        Point(x=float(x), y=self._profile.roi.y + self._profile.roi_size.y),
                        steps=20,
                    )

                self._hit_point_index += 1
                self._update_status()
                self._display_frame()
            return

        if self._step == CalibStep.COLOR_SAMPLE:
            cs = sample_color_from_region(self._frame, (x, y), radius=15)
            self._profile.note_color = cs
            self._step = CalibStep.HIT_POINTS
            self._hit_point_index = 0
            self._update_status()
            self._display_frame()
            return

    def _update_status(self) -> None:
        if self._step == CalibStep.NONE:
            self._status_label.setText("Click 'Start Calibration' then click ROI top-left corner")
        elif self._step == CalibStep.ROI_TOP_LEFT:
            self._status_label.setText("Step 1: Click top-left corner of gameplay area")
        elif self._step == CalibStep.ROI_BOTTOM_RIGHT:
            self._status_label.setText("Step 2: Click bottom-right corner of gameplay area")
        elif self._step == CalibStep.HIT_POINTS:
            remaining = LANE_KEYS[self._hit_point_index:]
            self._status_label.setText(
                f"Step 3: Click hit points for: {', '.join(remaining)}"
            )
        elif self._step == CalibStep.COLOR_SAMPLE:
            self._status_label.setText("Click on a note head to sample its color")
        elif self._step == CalibStep.DONE:
            self._status_label.setText("Calibration complete!")


from PySide6.QtWidgets import QHBoxLayout

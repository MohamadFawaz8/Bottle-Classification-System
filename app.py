"""
app.py  —  Bottle Classification Project  |  PyQt5 Desktop Application
Run with:  python app.py

Displays REAL YOLO model results via pipeline.py.
Supports: image upload · video upload · live webcam
Glassmorphism UI with tabbed interface and results tracking
"""

import sys
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import shutil
import os
import tempfile
from functools import partial

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_MPL_CONFIG_DIR = os.path.join(_APP_DIR, ".mpl_config")
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)

from pipeline import (
    run_pipeline,
    reset_tracking_state,
)  # ← REAL model outputs only - MUST import before PyQt5
from counter import ImageCounter, VideoCounter, WebcamCounter

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSizePolicy,
    QGraphicsDropShadowEffect,
    QTabWidget,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QHeaderView,
    QAbstractItemView,
    QSlider,
    QDialog,
)
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSlot, pyqtSignal, QSize
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QFont,
    QColor,
    QPainter,
    QLinearGradient,
    QPalette,
    QIcon,
)

# ─────────────────────────────────────────────────────────────────────────────
#  STYLE - GLASSMORPHISM
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg": "#0b1f3a",
    "glass": "rgba(15, 34, 63, 0.88)",
    "border": "rgba(255, 255, 255, 0.08)",
    "accent_cyan": "#0ea5e9",
    "accent_emerald": "#10b981",
    "accent_yellow": "#fbbf24",
    "accent_red": "#ef4444",
    "text": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "glow_cyan": "rgba(14, 165, 233, 0.16)",
    "glow_emerald": "rgba(16, 185, 129, 0.15)",
}

STYLESHEET = f"""
/* ── Application base ── */
QMainWindow, QWidget {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', sans-serif;
}}

QTabWidget::pane {{
    border: none;
    background-color: transparent;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {PALETTE['text_secondary']};
    padding: 8px 30px;
    border: none;
    font-weight: 600;
    min-width: 120px;
}}

QTabBar::tab:selected {{
    color: white;
    background-color: rgba(14, 165, 233, 0.15);
    border-bottom: 2px solid {PALETTE['accent_cyan']};
}}

QTabBar::tab:hover {{
    color: white;
}}

/* ── Header bar ── */
#HeaderBar {{
    background-color: {PALETTE['glass']};
    border: 1px solid {PALETTE['border']};
    border-radius: 24px;
    padding: 30px;

}}

#HeaderTitle {{
    color: {PALETTE['accent_cyan']};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}}

#HeaderMain {{
    color: white;
    font-size: 28px;
    font-weight: 600;
}}

/* ── Buttons ── */
QPushButton {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(14, 165, 233, 0.1),
        stop:1 rgba(59, 130, 246, 0.1)
    );
    color: {PALETTE['text']};
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 24px;
    padding: 10px 24px;
    font-size: 12px;
    font-weight: 600;
    min-width: 120px;
}}

QPushButton:hover {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(14, 165, 233, 0.2),
        stop:1 rgba(59, 130, 246, 0.2)
    );
}}

QPushButton:pressed {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(14, 165, 233, 0.3),
        stop:1 rgba(59, 130, 246, 0.3)
    );
}}

QPushButton:disabled {{
    color: {PALETTE['text_secondary']};
    border: 1px solid {PALETTE['border']};
}}

#StopBtn {{
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
}}

#StopBtn:hover {{
    background: rgba(239, 68, 68, 0.2);
}}

/* ── Glass Cards ── */
#GlassCard {{
    background-color: {PALETTE['glass']};
    border: 1px solid {PALETTE['border']};
    border-radius: 24px;
}}

/* ── Display frame ── */
#DisplayFrame {{
    background-color: rgba(15, 23, 42, 0.6);
    border: 1px solid {PALETTE['border']};
    border-radius: 24px;
}}

#DisplayLabel {{
    color: {PALETTE['text_secondary']};
    font-size: 14px;
}}

/* ── Result text ── */
#ResultText {{
    color: {PALETTE['text']};
    font-size: 12px;
    font-family: 'Consolas', 'JetBrains Mono', monospace;
    background-color: transparent;
    border: none;
}}

/* ── Table ── */
QTableWidget {{
    background-color: transparent;
    alternate-background-color: rgba(255, 255, 255, 0.02);
    gridline-color: {PALETTE['border']};
    border: none;
}}

QHeaderView::section {{
    background-color: rgba(15, 23, 42, 0.4);
    color: {PALETTE['text_secondary']};
    padding: 4px;
    border: none;
    gap: 20px;
}}

QTableWidget::item {{
    padding: 4px;
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']};
}}

/* ── Scroll bars ── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
}}

QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: rgba(255, 255, 255, 0.2);
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
}}

QScrollBar::handle:horizontal {{
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background: rgba(255, 255, 255, 0.2);
}}

QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical,
QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {{
    background: none;
}}

/* ── Text input ── */
QTextEdit {{
    background-color: rgba(15, 23, 42, 0.4);
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 8px;
}}
"""

DISPLAY_W = 960
DISPLAY_H = 540
TIMER_MS = 30


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def bgr_to_pixmap(bgr_frame, max_w=DISPLAY_W, max_h=DISPLAY_H, allow_upscale=False):
    """Convert a BGR numpy frame → scaled QPixmap."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # Preserve aspect ratio
    scale = min(max_w / w, max_h / h)
    if scale < 1.0:
        rgb = cv2.resize(
            rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    elif allow_upscale and scale > 1.0:
        rgb = cv2.resize(
            rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
    h2, w2 = rgb.shape[:2]
    qimg = QImage(rgb.data, w2, h2, w2 * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def make_shadow(radius=20, color="#00C6FF", opacity=80):
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(radius)
    fx.setColor(QColor(color))
    fx.setOffset(0, 0)
    fx.setColor(
        QColor(color[0], color[1], color[2], opacity)
        if isinstance(color, tuple)
        else QColor(color)
    )
    return fx


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT STORAGE & STATS
# ─────────────────────────────────────────────────────────────────────────────


class RunResult:
    """Stores a single detection run with metadata."""

    def __init__(
        self,
        source_type,
        start_timestamp,
        end_timestamp,
        total_objects,
        processing_time,
        detections,
        frame_or_video_path,
        result_text=None,
    ):
        self.source_type = source_type
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.detections = list(detections or [])
        self.total_objects = len(self.detections)
        self.processing_time = processing_time
        self.frame_or_video_path = (
            frame_or_video_path  # Frame for image, path to video for video/webcam
        )
        self.result_text = result_text or build_run_result_text(
            source_type, self.detections
        )
        self.deformity_counts = defaultdict(int)
        self.fullness_counts = defaultdict(int)
        self.capacity_counts = defaultdict(int)
        # capacity of bottles that are non-deformed AND full
        self.non_deformed_full_capacity_counts = defaultdict(int)
        # capacity of bottles that are deformed OR not full
        self.deformed_capacity_counts = defaultdict(int)

        # Parse detections for category counts
        if self.detections:
            for det in self.detections:
                deformity = det.get("deformity")
                fullness = det.get("fullness")
                capacity = det.get("capacity")
                if _is_valid_stat_value(deformity):
                    self.deformity_counts[deformity] += 1
                if _is_valid_stat_value(fullness):
                    self.fullness_counts[fullness] += 1
                if _is_valid_stat_value(capacity):
                    self.capacity_counts[capacity] += 1

                # Determine non-deformed / full classification
                deformity_str = (
                    str(deformity or "")
                    .strip()
                    .lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                )
                fullness_str = (
                    str(fullness or "")
                    .strip()
                    .lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                )
                is_non_deformed = deformity_str in {
                    "non_deformed",
                    "not_deformed",
                    "normal",
                    "ok",
                }
                # Only the exact class "full" qualifies — half_full, three_quarters_full, etc. do NOT
                is_full = fullness_str == "full"

                if _is_valid_stat_value(capacity):
                    if is_non_deformed and is_full:
                        self.non_deformed_full_capacity_counts[capacity] += 1
                    else:
                        self.deformed_capacity_counts[capacity] += 1


def _is_valid_stat_value(value):
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in {"unknown", "none", "n/a"}


def is_near_center(box, frame_w, frame_h):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return frame_w * 0.3 < cx < frame_w * 0.7 and frame_h * 0.3 < cy < frame_h * 0.7


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - intersection
    return intersection / float(max(1, union))


def vertical_overlap_ratio(box1, box2):
    top = max(box1[1], box2[1])
    bottom = min(box1[3], box2[3])
    overlap = max(0, bottom - top)
    min_height = max(1, min(box1[3] - box1[1], box2[3] - box2[1]))
    return overlap / float(min_height)


def center_distance(box1, box2):
    c1x = (box1[0] + box1[2]) / 2.0
    c1y = (box1[1] + box1[3]) / 2.0
    c2x = (box2[0] + box2[2]) / 2.0
    c2y = (box2[1] + box2[3]) / 2.0
    return float(np.hypot(c1x - c2x, c1y - c2y))


def same_bottle_candidate(box1, box2, iou_threshold=0.55, distance_ratio=0.65):
    if len(box1) != 4 or len(box2) != 4:
        return False
    if calculate_iou(box1, box2) >= iou_threshold:
        return True

    vertical_overlap = vertical_overlap_ratio(box1, box2)
    width1 = max(1, box1[2] - box1[0])
    width2 = max(1, box2[2] - box2[0])
    min_width = float(min(width1, width2))
    max_distance = max(18.0, min_width * distance_ratio)

    return vertical_overlap >= 0.82 and center_distance(box1, box2) <= max_distance


def obvious_duplicate_bottle(box1, box2):
    if len(box1) != 4 or len(box2) != 4:
        return False

    iou = calculate_iou(box1, box2)
    overlap = vertical_overlap_ratio(box1, box2)
    distance = center_distance(box1, box2)
    width1 = max(1, box1[2] - box1[0])
    width2 = max(1, box2[2] - box2[0])
    min_width = float(min(width1, width2))
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    area_ratio = max(area1, area2) / float(min(area1, area2))

    return iou >= 0.62 or (
        overlap >= 0.88
        and distance <= max(22.0, min_width * 0.32)
        and area_ratio <= 1.75
    )


def build_run_result_text(source_type, detections):
    detections = list(detections or [])
    title = f"{source_type.capitalize()} processed. Unique bottles counted after verification: {len(detections)}"
    if not detections:
        return title

    lines = [title]
    ordered = sorted(
        detections,
        key=lambda item: (
            item.get("display_id")
            if item.get("display_id") is not None
            else item.get("track_id", 999999)
        ),
    )
    for index, detection in enumerate(ordered, 1):
        bottle_number = detection.get("display_id", index)
        lines.append(
            f"Bottle {bottle_number}: "
            f"Condition={detection.get('deformity') or '—'} | "
            f"Capacity={detection.get('capacity') or '—'} | "
            f"Fullness={detection.get('fullness') or '—'}"
        )
    return "\n".join(lines)


def create_pie_chart(data_dict, title):
    labels = list(data_dict.keys()) or ["No Data"]
    sizes = list(data_dict.values()) or [1]

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_facecolor("#0f223f")
    ax.set_facecolor("#0f223f")
    colors = ["#0ea5e9", "#10b981", "#fbbf24", "#ef4444", "#38bdf8", "#22c55e"]
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(labels)],
        labeldistance=1.08,
        pctdistance=0.72,
        textprops={"color": "#e2e8f0", "fontsize": 10, "fontweight": "semibold"},
        wedgeprops={"linewidth": 1.0, "edgecolor": "#0b1f3a"},
    )
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=16)
    ax.set_aspect("equal")
    fig.tight_layout(pad=1.2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  BOTTLE STORE  (imported from counter.py)
# ─────────────────────────────────────────────────────────────────────────────
from counter import BottleStore


# ─────────────────────────────────────────────────────────────────────────────
#  WEBCAM INFERENCE WORKER  (runs in a background QThread)
# ─────────────────────────────────────────────────────────────────────────────


class WebcamWorker(QObject):
    """
    Runs frame capture + YOLO inference in a dedicated thread so the Qt main
    thread (and therefore the UI) is never blocked.

    Strategy
    --------
    * A tight loop grabs the latest camera frame and runs run_pipeline().
    * If inference is slower than the camera, intermediate frames are dropped
      (we always process the *latest* available frame, never queue up stale ones).
    * Results are emitted via a signal; the main thread connects a slot to
      receive them and update the display.
    """

    # annotated_frame (numpy), result_text (str), detections (list)
    result_ready = pyqtSignal(object, str, object)
    # status text, colour hex
    status_update = pyqtSignal(str, str)

    def __init__(self, cap, webcam_counter, palette):
        super().__init__()
        self._cap = cap
        self._counter = webcam_counter
        self._palette = palette
        self._running = True

    def stop(self):
        self._running = False

    @pyqtSlot()
    def run(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            # Drain any buffered frames so we always process the freshest one
            for _ in range(2):
                self._cap.grab()
            ret, frame = self._cap.retrieve()
            if not ret:
                break

            # Downscale to 640 wide for consistent inference speed
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(
                    frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA
                )

            annotated, result_text, detections = run_pipeline(
                frame, self._counter, use_webcam=True
            )

            visible_count = len(detections or [])
            count = self._counter.get_count()
            self.status_update.emit(
                f"Webcam Live • Visible: {visible_count} • Unique: {count}",
                self._palette["accent_yellow"],
            )
            self.result_ready.emit(annotated, result_text, detections)

        self.status_update.emit("Ready", self._palette["accent_cyan"])


class BottleInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bottle Classification System")
        self.setMinimumSize(1800, 900)

        self._cap = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._video_playback_step)
        self._current_video_capture = None
        self._current_video_path = None
        self._current_video_fps = 30
        self._current_video_frame = 0
        self._current_video_frame_count = 0
        self._mode = "idle"  # idle | image | video | webcam

        # Initialize counters
        self._image_counter = ImageCounter()
        self._video_counter = VideoCounter()
        self._webcam_counter = None
        self._bottle_store = BottleStore()
        self._webcam_store = BottleStore()

        # Results and stats
        self._run_results = []
        self._stats_total = 0
        self._stats_unique = 0
        self._stats_runs = 0
        self._current_result_index = 0
        self._selected_run_index = None
        self._result_display_frame = None
        self._result_zoom = 1.0

        # Current run
        self._current_run_start = None
        self._last_annotated = None

        # Webcam worker thread (inference runs off the main thread)
        self._webcam_thread = None
        self._webcam_worker = None

        self._build_ui()
        self.setStyleSheet(STYLESHEET)

    # ─── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # ── Header ─────────────────────────────────────────────────────────
        header = self._build_header()
        root_layout.addWidget(header)

        # ── Tab Widget ─────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)

        self._tab_dashboard = self._build_tab_dashboard()
        self._tab_results = self._build_tab_results()
        self._tab_history = self._build_tab_history()
        self._tab_total_stats = self._build_tab_total_stats()
        self._tab_stats = self._build_tab_stats()
        self._tab_instructions = self._build_tab_instructions()

        self._tabs.addTab(self._tab_dashboard, "Dashboard")
        self._tabs.addTab(self._tab_results, "Results")
        self._tabs.addTab(self._tab_history, "Run History")
        self._tabs.addTab(self._tab_total_stats, "Total Stats")
        self._tabs.addTab(self._tab_stats, "Stats")
        self._tabs.addTab(self._tab_instructions, "Instructions")

        root_layout.addWidget(self._tabs, stretch=1)

    def _build_header(self):
        """Build the header section."""
        header = QWidget(objectName="HeaderBar")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 15, 20, 15)
        header_layout.setSpacing(8)

        title_label = QLabel("BOTTLE CLASSIFICATION SYSTEM", objectName="HeaderTitle")
        main_label = QLabel(
            "Automated Bottle Detection & Analysis", objectName="HeaderMain"
        )
        desc_label = QLabel(
            "Premium bottle analytics for image, video, and webcam workflows",
            objectName="HeaderTitle",
        )
        desc_label.setStyleSheet(f"color: {PALETTE['text_secondary']};")

        header_layout.addWidget(title_label)
        header_layout.addWidget(main_label)
        header_layout.addWidget(desc_label)

        return header

    def _build_tab_dashboard(self):
        """Build the Dashboard tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # ── Upload controls ────────────────────────────────────────────────
        controls_frame = QWidget(objectName="GlassCard")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(20, 15, 20, 15)
        controls_layout.setSpacing(10)

        self._btn_image = QPushButton("📤  Upload Image")
        self._btn_video = QPushButton("🎬  Upload Video")
        self._btn_webcam = QPushButton("📹  Open Webcam")
        self._btn_stop = QPushButton("⏹  Stop")
        self._btn_stop.setObjectName("StopBtn")
        self._btn_stop.setEnabled(False)

        for btn in (self._btn_image, self._btn_video, self._btn_webcam, self._btn_stop):
            controls_layout.addWidget(btn)

        controls_layout.addStretch()

        self._status_label = QLabel("READY")
        self._status_label.setStyleSheet(
            f"color: {PALETTE['accent_cyan']}; font-weight: 600; font-size: 11px;"
        )
        controls_layout.addWidget(self._status_label)

        layout.addWidget(controls_frame)

        # ── Display area ───────────────────────────────────────────────────
        display_frame = QWidget(objectName="DisplayFrame")
        display_layout = QVBoxLayout(display_frame)
        display_layout.setContentsMargins(0, 0, 0, 0)

        self._display = QLabel(objectName="DisplayLabel")
        self._display.setAlignment(Qt.AlignCenter)
        # Allow the label to expand fully in both directions to fill the frame
        self._display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._display.setMinimumSize(1, 1)  # remove any floor so it grows freely
        self._show_placeholder()
        display_layout.addWidget(self._display)

        layout.addWidget(display_frame, stretch=1)

        # ── Result output panel ────────────────────────────────────────────
        result_frame = QWidget(objectName="GlassCard")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(16, 12, 16, 12)
        result_layout.setSpacing(4)
        result_layout.setAlignment(Qt.AlignTop)

        result_label = QLabel("MODEL OUTPUT", objectName="HeaderTitle")
        self._result_text = QTextEdit(objectName="ResultText")
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(80)
        self._result_text.setText("—")

        result_layout.addWidget(result_label)
        result_layout.addWidget(self._result_text)

        layout.addWidget(result_frame)

        # ── Connections ───────────────────────────────────────────────────
        self._btn_image.clicked.connect(self._open_image)
        self._btn_video.clicked.connect(self._open_video)
        self._btn_webcam.clicked.connect(self._open_webcam)
        self._btn_stop.clicked.connect(self._stop)

        return widget

    def _build_tab_results(self):
        """Build the Results tab."""
        widget = QScrollArea()
        widget.setWidgetResizable(True)
        widget.setStyleSheet("border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # ── Run info section ────────────────────────────────────────────────
        info_frame = QWidget(objectName="GlassCard")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(16, 12, 16, 12)

        info_title = QLabel("RUN INFORMATION", objectName="HeaderTitle")
        info_grid_widget = QWidget()
        info_grid = QHBoxLayout(info_grid_widget)
        info_grid.setContentsMargins(0, 0, 0, 0)
        info_grid.setSpacing(10)

        self._result_timestamp = QLabel("—")
        self._result_objects = QLabel("0")
        self._result_objects.setStyleSheet(
            f"color: {PALETTE['accent_cyan']}; font-size: 16px; font-weight: 600;"
        )
        self._result_time = QLabel("—")
        self._result_source = QLabel("—")
        self._result_end_timestamp = QLabel("—")

        for label_text, label_widget in [
            ("Start Timestamp", self._result_timestamp),
            ("End Timestamp", self._result_end_timestamp),
            ("Total Objects", self._result_objects),
            ("Time (ms)", self._result_time),
            ("Source", self._result_source),
        ]:
            card = QWidget()
            card.setStyleSheet(
                f"background-color: rgba(15, 23, 42, 0.4); border-radius: 12px; padding: 8px;"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(4)
            card_layout.addWidget(QLabel(label_text, objectName="HeaderTitle"))
            card_layout.addWidget(label_widget)
            info_grid.addWidget(card)

        info_layout.addWidget(info_title)
        info_layout.addWidget(info_grid_widget)
        layout.addWidget(info_frame)

        # ── Image and detections grid ──────────────────────────────────────
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        # Buttons panel (left side, vertical)
        buttons_panel = QWidget()
        buttons_panel_layout = QVBoxLayout(buttons_panel)
        buttons_panel_layout.setContentsMargins(0, 0, 0, 0)
        buttons_panel_layout.setSpacing(12)

        self._btn_save_image = QPushButton("💾 Save Image")
        self._btn_save_video = QPushButton("💾 Save Video")
        self._btn_next = QPushButton("➡️ Next")
        self._btn_save_image.setEnabled(False)
        self._btn_save_video.setEnabled(False)
        self._btn_next.setEnabled(False)

        for btn in [self._btn_save_image, self._btn_save_video, self._btn_next]:
            btn.setMinimumWidth(100)
            buttons_panel_layout.addWidget(btn)
        buttons_panel_layout.addStretch()
        content_layout.addWidget(buttons_panel, stretch=0)

        # Image
        image_frame = QWidget(objectName="GlassCard")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(16, 12, 16, 12)

        image_title = QLabel("ANNOTATED OUTPUT", objectName="HeaderTitle")
        self._btn_zoom_in = QPushButton("🔍 +")
        self._btn_zoom_out = QPushButton("🔍 -")
        self._btn_zoom_in.setEnabled(False)
        self._btn_zoom_out.setEnabled(False)

        zoom_controls = QWidget()
        zoom_controls_layout = QHBoxLayout(zoom_controls)
        zoom_controls_layout.setContentsMargins(0, 0, 0, 0)
        zoom_controls_layout.setSpacing(8)
        zoom_controls_layout.addWidget(self._btn_zoom_in)
        zoom_controls_layout.addWidget(self._btn_zoom_out)
        zoom_controls_layout.addStretch()

        self._result_image = QLabel(objectName="DisplayLabel")
        self._result_image.setAlignment(Qt.AlignCenter)
        self._result_image.setStyleSheet(
            f"background-color: rgba(15, 23, 42, 0.4); border-radius: 12px;"
        )
        self._result_image.setMinimumHeight(DISPLAY_H - 50)
        self._result_image.setText("No results yet")
        self._result_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._result_image_scroll = QScrollArea()
        self._result_image_scroll.setWidgetResizable(False)
        self._result_image_scroll.setWidget(self._result_image)
        self._result_image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._result_image_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._result_image_scroll.setStyleSheet("border: none;")
        self._result_image_scroll.setMinimumHeight(DISPLAY_H - 50)

        # Video player preview
        self._result_video_display = QLabel(objectName="DisplayLabel")
        self._result_video_display.setAlignment(Qt.AlignCenter)
        self._result_video_display.setStyleSheet(
            f"background-color: rgba(15, 23, 42, 0.4); border-radius: 12px;"
        )
        self._result_video_display.setMinimumHeight(DISPLAY_H - 50)
        self._result_video_display.setText("No video loaded")
        self._result_video_display.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self._result_video_display.hide()

        self._video_slider = QSlider(Qt.Horizontal)
        self._video_slider.setRange(0, 0)
        self._video_slider.setEnabled(False)
        self._video_time_label = QLabel("00:00 / 00:00")
        self._video_time_label.setStyleSheet("color: #cbd5e1; font-size: 12px;")

        # Video controls
        self._btn_play_pause = QPushButton("▶ Play")
        self._btn_rewind = QPushButton("⏪ Rewind")
        self._btn_play_pause.setEnabled(False)
        self._btn_rewind.setEnabled(False)

        video_controls = QWidget()
        video_controls_layout = QVBoxLayout(video_controls)
        video_controls_layout.setContentsMargins(0, 0, 0, 0)
        video_controls_layout.setSpacing(8)

        video_button_row = QWidget()
        video_button_row_layout = QHBoxLayout(video_button_row)
        video_button_row_layout.setContentsMargins(0, 0, 0, 0)
        video_button_row_layout.setSpacing(8)
        video_button_row_layout.addWidget(self._btn_rewind)
        video_button_row_layout.addWidget(self._btn_play_pause)
        video_button_row_layout.addStretch()
        video_button_row_layout.addWidget(self._video_time_label)

        video_controls_layout.addWidget(video_button_row)
        video_controls_layout.addWidget(self._video_slider)
        video_controls.hide()
        self._video_controls = video_controls

        image_layout.addWidget(image_title)
        image_layout.addWidget(zoom_controls)
        image_layout.addWidget(self._result_image_scroll, stretch=1)
        image_layout.addWidget(video_controls)
        image_layout.addWidget(self._result_video_display, stretch=1)

        content_layout.addWidget(image_frame, stretch=2)

        # Detections table
        detections_frame = QWidget(objectName="GlassCard")
        detections_layout = QVBoxLayout(detections_frame)
        detections_layout.setContentsMargins(16, 12, 16, 12)

        detections_title = QLabel("BOTTLE STATS", objectName="HeaderTitle")
        self._result_count_label = QLabel("Total Bottles: 0")
        self._result_count_label.setStyleSheet(
            f"color: {PALETTE['text_secondary']}; font-size: 12px;"
        )
        self._result_detections_table = QTableWidget()
        self._result_detections_table.setColumnCount(4)
        self._result_detections_table.setHorizontalHeaderLabels(
            ["Bottle #", "Deformity", "Fullness", "Capacity"]
        )
        self._result_detections_table.horizontalHeader().setStretchLastSection(True)

        detections_layout.addWidget(detections_title)
        detections_layout.addWidget(self._result_count_label)
        detections_layout.addWidget(self._result_detections_table, stretch=1)

        content_layout.addWidget(detections_frame, stretch=1)

        layout.addWidget(content_widget, stretch=1)

        # Connections
        self._btn_save_image.clicked.connect(self._save_image)
        self._btn_save_video.clicked.connect(self._save_video)
        self._btn_next.clicked.connect(self._next_result)
        self._btn_zoom_in.clicked.connect(self._zoom_in_result)
        self._btn_zoom_out.clicked.connect(self._zoom_out_result)
        self._btn_play_pause.clicked.connect(self._play_pause_video)
        self._btn_rewind.clicked.connect(self._rewind_video)
        self._video_slider.sliderMoved.connect(self._seek_video)

        widget.setWidget(container)
        return widget

    def _build_tab_history(self):
        """Build the Run History tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        history_frame = QWidget(objectName="GlassCard")
        history_layout = QVBoxLayout(history_frame)
        history_layout.setContentsMargins(16, 12, 16, 12)
        history_layout.setSpacing(10)

        history_title = QLabel("RUN HISTORY", objectName="HeaderTitle")
        self._history_table = QTableWidget()
        self._history_table.setColumnCount(6)
        self._history_table.setHorizontalHeaderLabels(
            ["Start Time", "End Time", "Type", "Result", "Stats", "Save"]
        )
        self._history_table.horizontalHeader().setStretchLastSection(False)
        self._history_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._history_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self._history_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self._history_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self._history_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeToContents
        )
        self._history_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeToContents
        )
        self._history_table.setAlternatingRowColors(True)
        self._history_table.setMinimumHeight(400)
        self._history_table.verticalHeader().setDefaultSectionSize(42)
        self._history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        history_layout.addWidget(history_title)
        history_layout.addWidget(self._history_table, stretch=1)
        layout.addWidget(history_frame, stretch=1)

        return widget

    def _build_tab_stats(self):
        """Build the Stats tab (per selected run)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Per-run charts ─────────────────────────────────────────────────
        charts_row = QWidget()
        charts_layout = QHBoxLayout(charts_row)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(16)

        self._chart_canvases = {}
        self._chart_buttons = {}
        for key, title in [
            ("deformity", "Deformity"),
            ("non_deformed_full", "Non-Deformed Bottles"),
            ("deformed", "Deformed Bottles"),
        ]:
            chart_card = QWidget(objectName="GlassCard")
            chart_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            chart_card.setMinimumHeight(430)
            chart_card_layout = QVBoxLayout(chart_card)
            chart_card_layout.setContentsMargins(18, 14, 18, 14)
            chart_card_layout.setSpacing(12)
            chart_card_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

            title_label = QLabel(title.upper(), objectName="HeaderTitle")
            title_label.setAlignment(Qt.AlignCenter)
            chart_card_layout.addWidget(title_label)
            figure = create_pie_chart({}, title)
            canvas = FigureCanvas(figure)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setMinimumHeight(310)
            chart_card_layout.addWidget(canvas, alignment=Qt.AlignCenter, stretch=1)

            button = QPushButton("More Details")
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.clicked.connect(partial(self._show_chart_details, key))
            chart_card_layout.addWidget(button, alignment=Qt.AlignHCenter)

            self._chart_canvases[key] = canvas
            self._chart_buttons[key] = button
            charts_layout.addWidget(chart_card, stretch=1)

        layout.addWidget(charts_row, stretch=1)

        # ── Selected run summary ───────────────────────────────────────────
        recent_frame = QWidget(objectName="GlassCard")
        recent_layout = QVBoxLayout(recent_frame)
        recent_layout.setContentsMargins(16, 12, 16, 12)
        recent_layout.setSpacing(8)

        recent_title = QLabel("SELECTED RUN SUMMARY", objectName="HeaderTitle")
        self._stats_runs_table = QTableWidget()
        self._stats_runs_table.setColumnCount(6)
        self._stats_runs_table.setHorizontalHeaderLabels(
            ["Date", "Start Time", "End Time", "Type", "Objects", "Time (ms)"]
        )
        self._stats_runs_table.horizontalHeader().setStretchLastSection(True)
        self._stats_runs_table.setAlternatingRowColors(True)
        self._stats_runs_table.setMinimumHeight(100)
        self._stats_runs_table.setMaximumHeight(130)
        self._stats_runs_table.verticalHeader().setDefaultSectionSize(38)

        recent_layout.addWidget(recent_title)
        recent_layout.addWidget(self._stats_runs_table)

        layout.addWidget(recent_frame)

        return widget

    def _build_tab_total_stats(self):
        """Build the Total Stats tab — aggregated across all runs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Aggregated charts ──────────────────────────────────────────────
        charts_row = QWidget()
        charts_layout = QHBoxLayout(charts_row)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(16)

        self._total_chart_canvases = {}
        self._total_chart_buttons = {}
        for key, title in [
            ("deformity", "Deformity"),
            ("non_deformed_full", "Non-Deformed Bottles"),
            ("deformed", "Deformed Bottles"),
        ]:
            chart_card = QWidget(objectName="GlassCard")
            chart_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            chart_card.setMinimumHeight(430)
            chart_card_layout = QVBoxLayout(chart_card)
            chart_card_layout.setContentsMargins(18, 14, 18, 14)
            chart_card_layout.setSpacing(12)
            chart_card_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

            title_label = QLabel(title.upper(), objectName="HeaderTitle")
            title_label.setAlignment(Qt.AlignCenter)
            chart_card_layout.addWidget(title_label)
            figure = create_pie_chart({}, title)
            canvas = FigureCanvas(figure)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setMinimumHeight(310)
            chart_card_layout.addWidget(canvas, alignment=Qt.AlignCenter, stretch=1)

            button = QPushButton("More Details")
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.clicked.connect(partial(self._show_total_chart_details, key))
            chart_card_layout.addWidget(button, alignment=Qt.AlignHCenter)

            self._total_chart_canvases[key] = canvas
            self._total_chart_buttons[key] = button
            charts_layout.addWidget(chart_card, stretch=1)

        layout.addWidget(charts_row, stretch=1)

        # ── All runs summary ───────────────────────────────────────────────
        summary_frame = QWidget(objectName="GlassCard")
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(16, 12, 16, 12)
        summary_layout.setSpacing(8)

        summary_title = QLabel("ALL RUNS SUMMARY", objectName="HeaderTitle")
        self._total_stats_runs_table = QTableWidget()
        self._total_stats_runs_table.setColumnCount(6)
        self._total_stats_runs_table.setHorizontalHeaderLabels(
            ["Date", "Start Time", "End Time", "Type", "Objects", "Time (ms)"]
        )
        self._total_stats_runs_table.horizontalHeader().setStretchLastSection(True)
        self._total_stats_runs_table.setAlternatingRowColors(True)
        self._total_stats_runs_table.setMinimumHeight(100)
        self._total_stats_runs_table.setMaximumHeight(130)
        self._total_stats_runs_table.verticalHeader().setDefaultSectionSize(38)

        # Total object count label
        self._total_stats_count_label = QLabel("Total Objects Across All Runs: 0")
        self._total_stats_count_label.setStyleSheet(
            f"color: {PALETTE['accent_cyan']}; font-size: 13px; font-weight: 600;"
        )

        summary_layout.addWidget(summary_title)
        summary_layout.addWidget(self._total_stats_count_label)
        summary_layout.addWidget(self._total_stats_runs_table)

        layout.addWidget(summary_frame)

        return widget

    def _build_tab_instructions(self):
        """Build the Instructions tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(24)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(24)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        intro_title = QLabel("How to Use Bottle Classification System")
        intro_title.setObjectName("HeaderMain")
        intro_title.setStyleSheet(
            f"font-size: 32px; font-weight: 700; color: {PALETTE['accent_cyan']};"
        )
        intro_label = QLabel(
            "This advanced AI-powered system provides comprehensive bottle analysis through automated detection and classification. Follow the steps below to get started with processing your images, videos, or live webcam feeds."
        )
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet(
            f"color: {PALETTE['text']}; font-size: 16px; line-height: 1.5;"
        )

        header_layout.addWidget(intro_title)
        header_layout.addWidget(intro_label)
        content_layout.addWidget(header)

        # Quick Start Section
        quick_start_title = QLabel("Quick Start Guide", objectName="HeaderTitle")
        quick_start_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        content_layout.addWidget(quick_start_title)

        card_row = QWidget()
        card_layout = QHBoxLayout(card_row)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(20)

        for title, desc in [
            (
                "📤 Upload Media",
                "Select images or videos from your device for batch processing. The system will automatically detect bottles and classify their deformity, fullness, and capacity using advanced YOLO models.",
            ),
            (
                "📹 Live Webcam",
                "Activate your camera for real-time bottle detection. Monitor live feeds with instant analytics, overlays, and continuous counting capabilities.",
            ),
            (
                "📊 Analyze Results",
                "Review detailed detection results, annotated outputs, and comprehensive statistics. Export processed media and save analysis reports for further use.",
            ),
        ]:
            card = QWidget(objectName="GlassCard")
            card_layout_inner = QVBoxLayout(card)
            card_layout_inner.setContentsMargins(20, 16, 20, 16)
            card_layout_inner.setSpacing(12)
            title_label = QLabel(title)
            title_label.setObjectName("HeaderTitle")
            title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet(
                f"color: {PALETTE['text_secondary']}; font-size: 14px; line-height: 1.4;"
            )
            card_layout_inner.addWidget(title_label)
            card_layout_inner.addWidget(desc_label)
            card_layout.addWidget(card)

        content_layout.addWidget(card_row)

        # Best Practices Section
        best_practices_title = QLabel("Best Practices & Tips", objectName="HeaderTitle")
        best_practices_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        content_layout.addWidget(best_practices_title)

        tips_row = QWidget()
        tips_layout = QHBoxLayout(tips_row)
        tips_layout.setContentsMargins(0, 0, 0, 0)
        tips_layout.setSpacing(20)

        for tip_icon, tips_list in [
            (
                "🎯 Detection Optimization",
                [
                    "Position bottles centrally in the frame for optimal detection accuracy.",
                    "Ensure adequate lighting to minimize false positives and improve confidence scores.",
                    "Maintain consistent distance and angle for best results across multiple bottles.",
                ],
            ),
            (
                "💾 Data Management",
                [
                    "Regularly save annotated images and videos from the Run History tab.",
                    "Review model statistics to understand detection patterns and system performance.",
                    "Export results for integration with other analysis tools or reporting systems.",
                ],
            ),
        ]:
            tip_card = QWidget(objectName="GlassCard")
            tip_layout_inner = QVBoxLayout(tip_card)
            tip_layout_inner.setContentsMargins(20, 16, 20, 16)
            tip_layout_inner.setSpacing(12)
            tip_title = QLabel(tip_icon)
            tip_title.setObjectName("HeaderTitle")
            tip_title.setStyleSheet("font-size: 16px; font-weight: 600;")
            tip_layout_inner.addWidget(tip_title)
            for tip in tips_list:
                tip_label = QLabel("• " + tip)
                tip_label.setWordWrap(True)
                tip_label.setStyleSheet(
                    f"color: {PALETTE['text_secondary']}; font-size: 14px; line-height: 1.4; border-left: 3px solid rgba(14, 165, 233, 0.3); padding-left: 15px;"
                )
                tip_layout_inner.addWidget(tip_label)
            tips_layout.addWidget(tip_card)

        content_layout.addWidget(tips_row)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

        return widget

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _show_placeholder(self):
        self._display.clear()
        self._display.setText(
            "📤 Upload an image or video, or open your webcam to begin detection."
        )

    def _lock_buttons(self, running=True):
        self._btn_image.setEnabled(not running)
        self._btn_video.setEnabled(not running)
        self._btn_webcam.setEnabled(not running)
        self._btn_stop.setEnabled(running)

    def _update_status(self, text, color=None):
        self._status_label.setText(text.upper())
        c = color or PALETTE["accent_cyan"]
        self._status_label.setStyleSheet(
            f"color: {c}; font-weight: 600; font-size: 11px;"
        )

    def _set_result_text(self, text):
        self._result_text.setText(text if text else "—")

    def _show_frame(self, bgr_frame):
        # Use the actual rendered size of the display label so the feed fills the box
        w = self._display.width()
        h = self._display.height()
        pix = bgr_to_pixmap(
            bgr_frame,
            w if w > 100 else DISPLAY_W,
            h if h > 100 else DISPLAY_H,
            allow_upscale=True,
        )
        self._display.setPixmap(pix)

    def _refresh_result_image(self):
        if self._result_display_frame is None:
            return
        max_w = int(DISPLAY_W * self._result_zoom)
        max_h = int((DISPLAY_H - 50) * self._result_zoom)
        pix = bgr_to_pixmap(
            self._result_display_frame, max_w, max_h, allow_upscale=True
        )
        self._result_image.setPixmap(pix)
        self._result_image.resize(pix.size())
        self._result_image_scroll.horizontalScrollBar().setValue(
            self._result_image_scroll.horizontalScrollBar().maximum() // 2
        )
        self._result_image_scroll.verticalScrollBar().setValue(
            self._result_image_scroll.verticalScrollBar().maximum() // 2
        )

    def _set_zoom_buttons(self):
        if self._result_display_frame is None:
            self._btn_zoom_in.setEnabled(False)
            self._btn_zoom_out.setEnabled(False)
            return
        self._btn_zoom_in.setEnabled(self._result_zoom < 3.0)
        self._btn_zoom_out.setEnabled(self._result_zoom > 0.5)

    @pyqtSlot()
    def _zoom_in_result(self):
        if self._result_display_frame is None:
            return
        self._result_zoom = min(self._result_zoom + 0.2, 3.0)
        self._refresh_result_image()
        self._set_zoom_buttons()

    @pyqtSlot()
    def _zoom_out_result(self):
        if self._result_display_frame is None:
            return
        self._result_zoom = max(self._result_zoom - 0.2, 0.5)
        self._refresh_result_image()
        self._set_zoom_buttons()

    def _reset_video_player(self):
        self._video_timer.stop()
        if self._current_video_capture is not None:
            self._current_video_capture.release()
            self._current_video_capture = None
        self._current_video_path = None
        self._current_video_fps = 30
        self._current_video_frame = 0
        self._current_video_frame_count = 0
        self._video_slider.setRange(0, 0)
        self._video_slider.setEnabled(False)
        self._video_time_label.setText("00:00 / 00:00")
        self._result_video_display.setText("No video loaded")
        self._result_video_display.clear()

    def _get_video_duration_ms(self):
        if self._current_video_fps and self._current_video_frame_count:
            return int(
                (self._current_video_frame_count / self._current_video_fps) * 1000
            )
        return 0

    def _format_time(self, position, duration):
        def fmt(ms):
            seconds = int(ms / 1000)
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{minutes:02}:{seconds:02}"

        return f"{fmt(position)} / {fmt(duration)}"

    def _display_video_frame(self, bgr_frame):
        pix = bgr_to_pixmap(bgr_frame, DISPLAY_W, DISPLAY_H - 50, allow_upscale=True)
        self._result_video_display.setPixmap(pix)
        self._result_video_display.resize(pix.size())

    def _load_video_preview(self, path):
        self._reset_video_player()
        self._current_video_path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._result_video_display.setText("Unable to load video.")
            return

        self._current_video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._current_video_fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)) or 30)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self._result_video_display.setText("Unable to load video frame.")
            return

        self._display_video_frame(frame)
        self._video_slider.setRange(0, max(0, self._current_video_frame_count - 1))
        self._video_slider.setEnabled(self._current_video_frame_count > 0)
        duration = self._get_video_duration_ms()
        self._video_time_label.setText(self._format_time(0, duration))

    @pyqtSlot()
    def _play_pause_video(self):
        if not self._current_video_path:
            return

        if self._video_timer.isActive():
            self._video_timer.stop()
            self._btn_play_pause.setText("▶ Play")
            return

        if self._current_video_capture is None:
            self._current_video_capture = cv2.VideoCapture(self._current_video_path)
            if not self._current_video_capture.isOpened():
                self._result_video_display.setText("Unable to open video.")
                self._current_video_capture = None
                return
            self._current_video_capture.set(
                cv2.CAP_PROP_POS_FRAMES, self._current_video_frame
            )

        self._btn_play_pause.setText("⏸ Pause")
        interval = max(1, int(1000 / self._current_video_fps))
        self._video_timer.start(interval)

    @pyqtSlot()
    def _video_playback_step(self):
        if self._current_video_capture is None:
            return

        ret, frame = self._current_video_capture.read()
        if not ret:
            self._video_timer.stop()
            self._btn_play_pause.setText("▶ Play")
            return

        self._display_video_frame(frame)
        self._current_video_frame += 1
        self._video_slider.setValue(self._current_video_frame)
        duration = self._get_video_duration_ms()
        position_ms = int(
            (self._current_video_frame / max(1, self._current_video_frame_count))
            * duration
        )
        self._video_time_label.setText(self._format_time(position_ms, duration))

        if self._current_video_frame >= self._current_video_frame_count - 1:
            self._video_timer.stop()
            self._btn_play_pause.setText("▶ Play")

    @pyqtSlot(int)
    def _seek_video(self, position):
        if not self._current_video_path:
            return

        self._video_timer.stop()
        self._btn_play_pause.setText("▶ Play")
        self._current_video_frame = position
        if self._current_video_capture is not None:
            self._current_video_capture.release()
            self._current_video_capture = None

        cap = cv2.VideoCapture(self._current_video_path)
        if not cap.isOpened():
            self._result_video_display.setText("Unable to seek video.")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._display_video_frame(frame)
            duration = self._get_video_duration_ms()
            position_ms = int(
                (position / max(1, self._current_video_frame_count)) * duration
            )
            self._video_time_label.setText(self._format_time(position_ms, duration))
        self._video_slider.setValue(position)

    @pyqtSlot()
    def _rewind_video(self):
        self._seek_video(0)

    def _update_history_display(self):
        self._history_table.setRowCount(0)
        for idx, result in enumerate(reversed(self._run_results)):
            actual_index = len(self._run_results) - 1 - idx
            row = idx
            self._history_table.insertRow(row)
            for column, text in enumerate(
                [
                    result.start_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    (
                        result.end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        if result.end_timestamp
                        else "—"
                    ),
                    result.source_type.upper(),
                    f"{result.total_objects} unique bottles",
                ]
            ):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._history_table.setItem(row, column, item)

            open_btn = QPushButton("Open Stats")
            open_btn.clicked.connect(partial(self._open_history_run, actual_index))
            open_btn.setMinimumWidth(80)
            open_btn.setFixedHeight(28)
            open_btn.setStyleSheet("font-size: 12px; padding: 4px 8px; margin: 0px;")
            open_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            open_cell_widget = QWidget()
            open_cell_layout = QHBoxLayout(open_cell_widget)
            open_cell_layout.setContentsMargins(0, 0, 0, 0)
            open_cell_layout.setAlignment(Qt.AlignCenter)
            open_cell_layout.addWidget(open_btn)
            self._history_table.setCellWidget(row, 4, open_cell_widget)

            save_btn = QPushButton("Save")
            save_btn.setEnabled(
                isinstance(result.frame_or_video_path, np.ndarray)
                or (
                    isinstance(result.frame_or_video_path, str)
                    and os.path.exists(result.frame_or_video_path)
                )
            )
            save_btn.clicked.connect(partial(self._save_history_item, actual_index))
            save_btn.setMinimumWidth(80)
            save_btn.setFixedHeight(28)
            save_btn.setStyleSheet("font-size: 12px; padding: 4px 8px; margin: 0px;")
            save_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            save_cell_widget = QWidget()
            save_cell_layout = QHBoxLayout(save_cell_widget)
            save_cell_layout.setContentsMargins(0, 0, 0, 0)
            save_cell_layout.setAlignment(Qt.AlignCenter)
            save_cell_layout.addWidget(save_btn)
            self._history_table.setCellWidget(row, 5, save_cell_widget)
            self._history_table.setRowHeight(row, 42)

    def _open_history_run(self, index):
        self._select_run(index, tab_index=4)

    def _save_history_item(self, index):
        if index < 0 or index >= len(self._run_results):
            return
        result = self._run_results[index]
        if isinstance(result.frame_or_video_path, np.ndarray):
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Annotated Image", "", "Images (*.png *.jpg)"
            )
            if path:
                cv2.imwrite(path, result.frame_or_video_path)
        elif isinstance(result.frame_or_video_path, str) and os.path.exists(
            result.frame_or_video_path
        ):
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Annotated Video", "", "Videos (*.mp4)"
            )
            if path:
                shutil.copy(result.frame_or_video_path, path)

    def _add_run_result(self, result: RunResult):
        """Add a result to the history and update UI."""
        self._run_results.append(result)
        self._stats_runs = len(self._run_results)
        self._selected_run_index = len(self._run_results) - 1
        self._update_results_display(result)
        self._update_stats_display(result)
        self._update_history_display()

    def _select_run(self, index, tab_index=None):
        if index < 0 or index >= len(self._run_results):
            return
        self._selected_run_index = index
        result = self._run_results[index]
        self._update_results_display(result)
        self._update_stats_display(result)
        if tab_index is not None:
            self._tabs.setCurrentIndex(tab_index)

    def _update_results_display(self, result: RunResult):
        """Update the results tab with the latest run."""
        # Timestamp
        self._result_timestamp.setText(
            result.start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        self._result_end_timestamp.setText(
            result.end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if result.end_timestamp
            else "—"
        )
        self._result_objects.setText(str(result.total_objects))
        self._result_time.setText(f"{result.processing_time:.2f}")
        self._result_source.setText(result.source_type.upper())

        # Image/Video display
        self._result_display_frame = (
            result.frame_or_video_path
            if isinstance(result.frame_or_video_path, np.ndarray)
            else None
        )
        self._result_zoom = 1.0
        self._set_zoom_buttons()

        if self._result_display_frame is not None:
            # Show image
            self._refresh_result_image()
            self._result_image_scroll.show()
            self._result_video_display.hide()
            self._video_controls.hide()
            self._btn_play_pause.setEnabled(False)
            self._btn_rewind.setEnabled(False)
            self._btn_save_image.setEnabled(True)
            self._btn_save_video.setEnabled(False)
        elif isinstance(result.frame_or_video_path, str) and os.path.exists(
            result.frame_or_video_path
        ):
            self._result_image_scroll.hide()
            self._result_video_display.show()
            self._video_controls.show()
            self._load_video_preview(result.frame_or_video_path)
            self._btn_play_pause.setEnabled(True)
            self._btn_rewind.setEnabled(True)
            self._btn_play_pause.setText("▶ Play")
            self._btn_save_image.setEnabled(False)
            self._btn_save_video.setEnabled(True)
        else:
            # No media
            self._result_image.clear()
            self._result_image.setText("No media available")
            self._result_image_scroll.show()
            self._result_video_display.hide()
            self._video_controls.hide()
            self._btn_play_pause.setEnabled(False)
            self._btn_rewind.setEnabled(False)
            self._btn_save_image.setEnabled(False)
            self._btn_save_video.setEnabled(False)

        self._btn_next.setEnabled(len(self._run_results) >= 1)
        self._set_result_text(result.result_text)

        # Detection table
        self._result_detections_table.setRowCount(0)
        self._result_count_label.setText(
            f"Total Bottles: {len(result.detections) if result.detections else 0}"
        )
        if result.detections:
            for idx, det in enumerate(result.detections):
                self._result_detections_table.insertRow(idx)
                bottle_number = det.get("display_id", idx + 1)
                self._result_detections_table.setItem(
                    idx, 0, QTableWidgetItem(str(bottle_number))
                )
                deformity = det.get("deformity") or "—"
                fullness = det.get("fullness") or "—"
                capacity = det.get("capacity") or "—"
                self._result_detections_table.setItem(
                    idx, 1, QTableWidgetItem(deformity)
                )
                self._result_detections_table.setItem(
                    idx, 2, QTableWidgetItem(fullness)
                )
                self._result_detections_table.setItem(
                    idx, 3, QTableWidgetItem(capacity)
                )

    def _update_stats_display(self, result=None):
        """Update the stats tab for the selected run only."""
        if result is None:
            if self._selected_run_index is None or not self._run_results:
                self._stats_runs_table.setRowCount(0)
                for key in self._chart_canvases:
                    self._refresh_chart(key, {}, key.title())
                return
            result = self._run_results[self._selected_run_index]

        self._refresh_chart("deformity", dict(result.deformity_counts), "Deformity")
        self._refresh_chart(
            "non_deformed_full",
            dict(result.non_deformed_full_capacity_counts),
            "Non-Deformed Bottles",
        )
        self._refresh_chart(
            "deformed", dict(result.deformed_capacity_counts), "Deformed Bottles"
        )

        self._stats_runs_table.setRowCount(1)
        date_str = result.start_timestamp.strftime("%d/%m/%Y")
        self._stats_runs_table.setItem(0, 0, QTableWidgetItem(date_str))
        self._stats_runs_table.setItem(
            0, 1, QTableWidgetItem(result.start_timestamp.strftime("%H:%M:%S"))
        )
        end_str = (
            result.end_timestamp.strftime("%H:%M:%S") if result.end_timestamp else "—"
        )
        self._stats_runs_table.setItem(0, 2, QTableWidgetItem(end_str))
        self._stats_runs_table.setItem(
            0, 3, QTableWidgetItem(result.source_type.upper())
        )
        self._stats_runs_table.setItem(
            0, 4, QTableWidgetItem(str(result.total_objects))
        )
        self._stats_runs_table.setItem(
            0, 5, QTableWidgetItem(f"{result.processing_time:.2f}")
        )

        # Also refresh total stats tab
        self._update_total_stats_display()

    def _refresh_chart(self, key, data, title):
        old_canvas = self._chart_canvases[key]
        parent_layout = old_canvas.parentWidget().layout()
        parent_layout.removeWidget(old_canvas)
        old_canvas.setParent(None)
        old_canvas.figure.clf()
        plt.close(old_canvas.figure)

        canvas = FigureCanvas(create_pie_chart(data, title))
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumHeight(310)
        parent_layout.insertWidget(1, canvas, stretch=1)
        self._chart_canvases[key] = canvas

    def _refresh_total_chart(self, key, data, title):
        old_canvas = self._total_chart_canvases[key]
        parent_layout = old_canvas.parentWidget().layout()
        parent_layout.removeWidget(old_canvas)
        old_canvas.setParent(None)
        old_canvas.figure.clf()
        plt.close(old_canvas.figure)

        canvas = FigureCanvas(create_pie_chart(data, title))
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumHeight(310)
        parent_layout.insertWidget(1, canvas, stretch=1)
        self._total_chart_canvases[key] = canvas

    def _update_total_stats_display(self):
        """Aggregate all runs and update the Total Stats tab."""
        if not self._run_results:
            self._total_stats_runs_table.setRowCount(0)
            self._total_stats_count_label.setText("Total Objects Across All Runs: 0")
            for key, title in [
                ("deformity", "Deformity"),
                ("non_deformed_full", "Non-Deformed Bottles"),
                ("deformed", "Deformed Bottles"),
            ]:
                self._refresh_total_chart(key, {}, title)
            return

        combined_deformity = defaultdict(int)
        combined_non_deformed_full = defaultdict(int)
        combined_deformed = defaultdict(int)
        total_objects = 0

        for r in self._run_results:
            total_objects += r.total_objects
            for k, v in r.deformity_counts.items():
                combined_deformity[k] += v
            for k, v in r.non_deformed_full_capacity_counts.items():
                combined_non_deformed_full[k] += v
            for k, v in r.deformed_capacity_counts.items():
                combined_deformed[k] += v

        self._refresh_total_chart("deformity", dict(combined_deformity), "Deformity")
        self._refresh_total_chart(
            "non_deformed_full",
            dict(combined_non_deformed_full),
            "Non-Deformed Bottles",
        )
        self._refresh_total_chart(
            "deformed", dict(combined_deformed), "Deformed Bottles"
        )

        self._total_stats_count_label.setText(
            f"Total Objects Across All Runs: {total_objects}"
        )

        # Populate runs table (all runs)
        self._total_stats_runs_table.setRowCount(len(self._run_results))
        for row, r in enumerate(reversed(self._run_results)):
            self._total_stats_runs_table.setItem(
                row, 0, QTableWidgetItem(r.start_timestamp.strftime("%d/%m/%Y"))
            )
            self._total_stats_runs_table.setItem(
                row, 1, QTableWidgetItem(r.start_timestamp.strftime("%H:%M:%S"))
            )
            end_str = r.end_timestamp.strftime("%H:%M:%S") if r.end_timestamp else "—"
            self._total_stats_runs_table.setItem(row, 2, QTableWidgetItem(end_str))
            self._total_stats_runs_table.setItem(
                row, 3, QTableWidgetItem(r.source_type.upper())
            )
            self._total_stats_runs_table.setItem(
                row, 4, QTableWidgetItem(str(r.total_objects))
            )
            self._total_stats_runs_table.setItem(
                row, 5, QTableWidgetItem(f"{r.processing_time:.2f}")
            )

    def _show_chart_details(self, key):
        if self._selected_run_index is None or not self._run_results:
            return

        result = self._run_results[self._selected_run_index]
        title_map = {
            "deformity": ("Deformity Details", result.deformity_counts),
            "non_deformed_full": (
                "Non-Deformed Bottles — Capacity Details",
                result.non_deformed_full_capacity_counts,
            ),
            "deformed": (
                "Deformed Bottles — Capacity Details",
                result.deformed_capacity_counts,
            ),
        }
        title, counts = title_map[key]

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(420, 320)
        layout = QVBoxLayout(dialog)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Class", "Count"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.setRowCount(len(counts))

        for row, (label, count) in enumerate(
            sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ):
            table.setItem(row, 0, QTableWidgetItem(str(label)))
            table.setItem(row, 1, QTableWidgetItem(str(count)))

        layout.addWidget(table)
        dialog.exec_()

    def _show_total_chart_details(self, key):
        if not self._run_results:
            return

        combined = defaultdict(int)
        for r in self._run_results:
            source = {
                "deformity": r.deformity_counts,
                "non_deformed_full": r.non_deformed_full_capacity_counts,
                "deformed": r.deformed_capacity_counts,
            }[key]
            for k, v in source.items():
                combined[k] += v

        title_map = {
            "deformity": "All Runs — Deformity Details",
            "non_deformed_full": "All Runs — Non-Deformed Bottles Capacity",
            "deformed": "All Runs — Deformed Bottles Capacity",
        }

        dialog = QDialog(self)
        dialog.setWindowTitle(title_map[key])
        dialog.resize(420, 320)
        layout = QVBoxLayout(dialog)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Class", "Count"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.setRowCount(len(combined))

        for row, (label, count) in enumerate(
            sorted(combined.items(), key=lambda item: item[1], reverse=True)
        ):
            table.setItem(row, 0, QTableWidgetItem(str(label)))
            table.setItem(row, 1, QTableWidgetItem(str(count)))

        layout.addWidget(table)
        dialog.exec_()

    # ─── Image ────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            self._set_result_text("ERROR: Could not read image.")
            return

        self._stop()
        self._mode = "image"
        self._update_status("Processing", PALETTE["accent_cyan"])
        QApplication.processEvents()

        # Process with image counter
        self._image_counter.reset()
        start_time = datetime.now()
        annotated, result_text, detections = run_pipeline(frame, self._image_counter)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        self._show_frame(annotated)
        self._set_result_text(result_text)
        self._update_status("Image Ready", PALETTE["accent_emerald"])

        # Store result
        result = RunResult(
            "image",
            start_time,
            start_time,
            len(detections) if detections else 0,
            processing_time,
            detections or [],
            annotated,
            result_text=result_text,
        )
        self._add_run_result(result)

        # Switch to results tab
        self._tabs.setCurrentIndex(1)

    # ─── Video ────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)"
        )
        if not path:
            return

        self._stop()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._set_result_text("ERROR: Could not open video.")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video path (temporary file so annotated video is previewable without autosaving to the original folder)
        output_path = os.path.join(os.getcwd(), "annotated_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self._video_counter.reset()
        self._video_counter.fps = float(fps or 30)
        reset_tracking_state()
        self._mode = "video"
        self._lock_buttons(True)
        self._update_status("Processing Video", PALETTE["accent_cyan"])
        QApplication.processEvents()

        start_time = datetime.now()
        last_frame = None

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            self._update_status(
                f"Processing Video: {frame_count}/{total_frames}",
                PALETTE["accent_cyan"],
            )
            QApplication.processEvents()

            # Process frame
            annotated, result_text, detections = run_pipeline(
                frame, self._video_counter
            )
            out.write(annotated)
            last_frame = annotated

        cap.release()
        out.release()

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        unique_detections = self._video_counter.finalize()
        unique_bottles = len(unique_detections)
        final_result_text = build_run_result_text("video", unique_detections)
        self._show_frame(last_frame)
        self._set_result_text(final_result_text)
        self._update_status("Video Complete", PALETTE["accent_emerald"])
        self._lock_buttons(False)

        # Store result
        result = RunResult(
            "video",
            start_time,
            datetime.now(),
            unique_bottles,
            processing_time,
            unique_detections,
            output_path,
            result_text=final_result_text,
        )
        self._add_run_result(result)

        # Switch to results tab
        self._tabs.setCurrentIndex(1)

    # ─── Webcam ───────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _open_webcam(self):
        self._stop()
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._set_result_text("ERROR: No webcam found.")
            self._cap = None
            return

        # Request resolution / fps from the camera driver
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or 30
        self._webcam_counter = WebcamCounter(frame_width)
        self._webcam_counter.fps = float(frame_fps or 30)
        reset_tracking_state()

        self._mode = "webcam"
        self._current_run_start = datetime.now()
        self._lock_buttons(True)
        self._update_status("Webcam Live", PALETTE["accent_yellow"])

        # Launch inference on a background thread so the UI stays responsive
        self._webcam_worker = WebcamWorker(self._cap, self._webcam_counter, PALETTE)
        self._webcam_thread = QThread()
        self._webcam_worker.moveToThread(self._webcam_thread)
        self._webcam_thread.started.connect(self._webcam_worker.run)
        self._webcam_worker.result_ready.connect(self._on_webcam_result)
        self._webcam_worker.status_update.connect(self._update_status)
        self._webcam_thread.start()

    # ─── Stop ─────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _stop(self):
        self._timer.stop()

        # Stop the webcam inference worker thread (if running)
        if self._webcam_worker is not None:
            self._webcam_worker.stop()
            self._webcam_worker = None
        if self._webcam_thread is not None:
            self._webcam_thread.quit()
            self._webcam_thread.wait(2000)
            self._webcam_thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._reset_video_player()
        if (
            self._mode == "webcam"
            and self._current_run_start
            and self._webcam_counter is not None
        ):
            # Add webcam run result
            unique_detections = self._webcam_counter.finalize()
            total_det = len(unique_detections)
            final_result_text = build_run_result_text("webcam", unique_detections)
            self._set_result_text(final_result_text)
            result = RunResult(
                "webcam",
                self._current_run_start,
                datetime.now(),
                total_det,
                0,
                unique_detections,
                self._last_annotated,
                result_text=final_result_text,
            )
            self._add_run_result(result)
            self._current_run_start = None
            self._last_annotated = None
            self._webcam_counter.reset()
        self._mode = "idle"
        self._lock_buttons(False)
        self._update_status("Ready")

    # ─── Timer tick (video playback only — webcam uses _on_webcam_result) ──────

    @pyqtSlot()
    def _on_timer(self):
        """Used only for video file playback (not webcam)."""
        if self._cap is None:
            self._stop()
            return

        ret, frame = self._cap.read()
        if not ret:
            if self._mode == "video":
                self._update_status("Complete", PALETTE["accent_emerald"])
            self._stop()
            return

        # Downscale for speed
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(
                frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA
            )

        annotated, result_text, detections = run_pipeline(
            frame,
            self._webcam_counter,
            use_webcam=True,
        )

        self._show_frame(annotated)
        self._set_result_text(result_text)
        self._last_annotated = annotated

    # ─── Webcam result slot (called from WebcamWorker signal) ─────────────────

    @pyqtSlot(object, str, object)
    def _on_webcam_result(self, annotated, result_text, detections):
        """Receives inference results from the background webcam worker thread."""
        self._show_frame(annotated)
        self._set_result_text(result_text)
        self._last_annotated = annotated

    @pyqtSlot()
    def _save_image(self):
        if not self._run_results:
            return
        result = self._run_results[-1]
        if result.source_type != "image":
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated Image", "", "Images (*.png *.jpg)"
        )
        if path:
            cv2.imwrite(path, result.frame_or_video_path)

    @pyqtSlot()
    def _save_video(self):
        if not self._run_results:
            return
        result = self._run_results[-1]
        if (
            result.source_type not in ["video", "webcam"]
            or not result.frame_or_video_path
            or not os.path.exists(result.frame_or_video_path)
        ):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated Video", "", "Videos (*.mp4)"
        )
        if path:
            shutil.copy(result.frame_or_video_path, path)

    @pyqtSlot()
    def _next_result(self):
        # Return to dashboard and clear display so user can upload a new image
        self._reset_video_player()
        self._tabs.setCurrentIndex(0)
        self._show_placeholder()
        self._set_result_text("")
        self._result_display_frame = None
        self._result_zoom = 1.0
        self._set_zoom_buttons()
        self._btn_save_image.setEnabled(False)
        self._btn_save_video.setEnabled(False)
        self._lock_buttons(False)
        self._update_status("Ready")

    def closeEvent(self, event):
        self._stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    pal = QPalette()
    dark = QColor(11, 31, 58)
    pal.setColor(QPalette.Window, dark)
    pal.setColor(QPalette.WindowText, QColor(226, 232, 240))
    pal.setColor(QPalette.Base, QColor(15, 23, 42))
    pal.setColor(QPalette.AlternateBase, dark)
    pal.setColor(QPalette.ToolTipBase, QColor(15, 23, 42))
    pal.setColor(QPalette.ToolTipText, QColor(226, 232, 240))
    pal.setColor(QPalette.Text, QColor(226, 232, 240))
    pal.setColor(QPalette.Button, QColor(30, 41, 59))
    pal.setColor(QPalette.ButtonText, QColor(226, 232, 240))
    pal.setColor(QPalette.Highlight, QColor(14, 165, 233))
    pal.setColor(QPalette.HighlightedText, QColor(11, 31, 58))
    app.setPalette(pal)

    win = BottleInspector()
    win.show()
    sys.exit(app.exec_())

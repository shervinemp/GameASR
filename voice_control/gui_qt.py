"""Always-on-top transparent overlay using PyQt6."""
import math
import time
from queue import Empty, Queue

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen
from PyQt6.QtWidgets import QWidget, QApplication


class MicOverlay(QWidget):

    _SIZE = 60
    _COLORS = {
        "idle":      QColor("#6b7280"),
        "listening": QColor("#22c55e"),
        "think":     QColor("#eab308"),
        "speak":     QColor("#3b82f6"),
        "muted":     QColor("#ef4444"),
        "error":     QColor("#ef4444"),
        "unavail":   QColor("#374151"),
    }

    def __init__(self, pipeline):
        super().__init__()
        self._pipeline = pipeline
        self._queue = Queue()
        self._rms = 0.0
        self._target_rms = 0.0
        self._state = "idle"
        self._muted = False
        self._radius = 10.0
        self._anim = 0
        self._drag_start = None

        self.setWindowTitle("Mic")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self._SIZE, self._SIZE)

        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - self._SIZE - 20, screen.height() - self._SIZE - 80)

        pipeline.events.on("vad:level", self._on_level, async_=True)
        pipeline.events.on("pipeline:error", lambda e: self._queue.put(("state", "error")))
        pipeline.events.on("pipeline:state", lambda s: self._queue.put(("state", s)))

        self._timer = QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(33)

    def _on_level(self, rms, prob):
        self._queue.put(("rms", rms))
        if prob > 0.3:
            self._queue.put(("state", "listening"))

    def _toggle_mute(self):
        if self._pipeline.asr and hasattr(self._pipeline.asr, "_is_muted"):
            if self._pipeline.asr._is_muted.is_set():
                self._pipeline.asr.enable()
                self._muted = False
            else:
                self._pipeline.asr.disable_w_passthrough()
                self._muted = True

    def _poll(self):
        try:
            while True:
                k, v = self._queue.get_nowait()
                if k == "rms":
                    self._target_rms = min(1.0, v * 10)
                elif k == "state":
                    self._state = v
        except Empty:
            pass
        self._rms += (self._target_rms - self._rms) * 0.3
        self._anim += 1
        self.update()

    def mousePressEvent(self, e):
        self._drag_start = (e.globalPosition().x(), e.globalPosition().y(), self.x(), self.y())

    def mouseMoveEvent(self, e):
        if self._drag_start is None:
            return
        dx = e.globalPosition().x() - self._drag_start[0]
        dy = e.globalPosition().y() - self._drag_start[1]
        self.move(int(self._drag_start[2] + dx), int(self._drag_start[3] + dy))

    def mouseReleaseEvent(self, e):
        if self._drag_start is None:
            return
        dx = abs(e.globalPosition().x() - self._drag_start[0])
        dy = abs(e.globalPosition().y() - self._drag_start[1])
        if dx < 5 and dy < 5:
            self._toggle_mute()
        self._drag_start = None

    def paintEvent(self, event):
        t = self._anim * 0.033
        cx = cy = self._SIZE // 2
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        state = self._state
        s = self._pipeline.status["asr"]
        if self._muted or s == "muted":
            state = "muted"
        elif state == "unavail" or s == "unavailable":
            state = "unavail"

        c = self._COLORS.get(state, self._COLORS["idle"])
        tr = 10.0
        rw = 3
        ra = 255
        mx = False

        if state == "error":
            ra = int(128 + 127 * abs(math.sin(t * 4)))
        elif state == "unavail":
            ra = 60
        elif state == "muted":
            tr = 12; ra = 100; mx = True
        elif state == "think":
            tr = 12 + 4 * math.sin(t * 3)
            rw = 4
            ra = int(180 + 75 * abs(math.sin(t * 3)))
        elif state in ("speak", "listening"):
            tr = (12 + 20 * self._rms) if state == "speak" else (10 + 22 * self._rms)
            rw = 4
            ra = int(130 + 125 * self._rms)
        else:
            ra = int(60 + 30 * abs(math.sin(t * 1.5)))

        self._radius += (tr - self._radius) * 0.2
        r = int(self._radius)

        ca = QColor(c)
        ca.setAlpha(ra)
        p.setPen(QPen(ca, rw))
        p.setBrush(QBrush())
        p.drawEllipse(int(cx - r - 4), int(cy - r - 4), int((r + 4) * 2), int((r + 4) * 2))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c))
        p.drawEllipse(int(cx - r), int(cy - r), r * 2, r * 2)

        if mx:
            s2 = int(r * 0.4)
            p.setPen(QPen(QColor("#ef4444"), 3))
            p.drawLine(cx - s2, cy - s2, cx + s2, cy + s2)

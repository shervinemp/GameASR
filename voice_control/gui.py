import math
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline


class MicButton:
    """Always-on-top overlay — state text + audio level meter."""

    _WIN_SIZE = 120
    _TRANS = "#0d0d0d"
    _COLORS = {
        "idle":      "#6b7280",
        "listening": "#22c55e",
        "think":     "#eab308",
        "speak":     "#3b82f6",
        "muted":     "#ef4444",
        "error":     "#ef4444",
        "unavail":   "#374151",
    }
    _LABELS = {
        "idle":      "",
        "listening": "LISTENING",
        "think":     "THINKING",
        "speak":     "SPEAKING",
        "muted":     "MUTED",
        "error":     "ERROR",
        "unavail":   "OFF",
    }

    def __init__(self, pipeline: "Pipeline"):
        self._pipeline = pipeline
        self._queue: Queue = Queue()
        self._rms = 0.0
        self._target_rms = 0.0
        self._state = "idle"
        self._muted = False
        self._running = False
        self._drag_start = None
        self._radius = 14.0
        self._anim = 0.0

        pipeline.events.on("vad:level", self._on_level, async_=True)
        pipeline.events.on("pipeline:error", lambda e: self._queue.put(("state", "error")))
        pipeline.events.on("pipeline:state", lambda s: self._queue.put(("state", s)))

    def _on_level(self, rms: float, prob: float):
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

    def run(self):
        import tkinter as tk

        self._running = True
        self._root = tk.Tk()
        self._root.title("Mic")
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-transparentcolor", self._TRANS)
        self._root.configure(bg=self._TRANS)

        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = screen_w - self._WIN_SIZE - 20
        y = screen_h - self._WIN_SIZE - 120
        self._root.geometry(f"{self._WIN_SIZE}x{self._WIN_SIZE}+{x}+{y}")

        self._canvas = tk.Canvas(
            self._root, width=self._WIN_SIZE, height=self._WIN_SIZE,
            bg=self._TRANS, highlightthickness=0,
        )
        self._canvas.pack()

        cx = cy = self._WIN_SIZE // 2

        self._ring = self._canvas.create_oval(0, 0, 0, 0, fill="", outline="", width=4)
        self._dot = self._canvas.create_oval(0, 0, 0, 0, fill="", outline="", width=0)
        self._label = self._canvas.create_text(
            cx, cy + 20, text="", font=("Segoe UI", 8, "bold"),
            fill="#ffffff", anchor="center",
        )
        self._level_bar = self._canvas.create_rectangle(
            0, 0, 0, 0, fill="", outline="", width=0,
        )
        self._mute_bar = self._canvas.create_line(0, 0, 0, 0, fill="#ef4444", width=3, capstyle="round")

        for tag in (self._ring, self._dot, self._label, self._level_bar, self._mute_bar):
            self._canvas.tag_bind(tag, "<Button-1>", self._on_drag_start)
            self._canvas.tag_bind(tag, "<B1-Motion>", self._on_drag_move)
            self._canvas.tag_bind(tag, "<ButtonRelease-1>", self._on_drag_end)

        try:
            self._poll()
            self._root.mainloop()
        except Exception:
            pass

    def _on_drag_start(self, event):
        self._drag_start = (event.x_root, event.y_root, self._root.winfo_x(), self._root.winfo_y())

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return
        dx = event.x_root - self._drag_start[0]
        dy = event.y_root - self._drag_start[1]
        new_x = self._drag_start[2] + dx
        new_y = self._drag_start[3] + dy
        self._root.geometry(f"+{new_x}+{new_y}")

    def _on_drag_end(self, event):
        if self._drag_start is None:
            return
        dx = abs(event.x_root - self._drag_start[0])
        dy = abs(event.y_root - self._drag_start[1])
        if dx < 5 and dy < 5:
            self._toggle_mute()
        self._drag_start = None

    def _poll(self):
        try:
            while True:
                kind, value = self._queue.get_nowait()
                if kind == "rms":
                    self._target_rms = min(1.0, value * 10)
                elif kind == "state":
                    self._state = value
        except Empty:
            pass

        self._rms += (self._target_rms - self._rms) * 0.3
        self._anim += 1
        self._redraw()
        if self._running:
            self._root.after(33, self._poll)

    def _redraw(self):
        t = self._anim * 0.033
        cx = cy = self._WIN_SIZE // 2

        state = self._state
        asr_status = self._pipeline.status["asr"]
        if self._muted or asr_status == "muted":
            state = "muted"
        elif state == "unavail" or asr_status == "unavailable":
            state = "unavail"

        color = self._COLORS.get(state, self._COLORS["idle"])
        label = self._LABELS.get(state, "")

        target_r = 14.0
        ring_w = 4
        show_mute = False

        if state == "error":
            target_r = 14
            ring_w = 4
        elif state == "unavail":
            target_r = 14
            ring_w = 2
        elif state == "muted":
            target_r = 16
            ring_w = 3
            show_mute = True
        elif state == "think":
            target_r = 16 + 3 * math.sin(t * 3)
            ring_w = 5
        elif state == "speak":
            target_r = 16 + 16 * self._rms
            ring_w = 5
        elif state == "listening":
            target_r = 14 + 20 * self._rms
            ring_w = 5
        else:
            target_r = 14
            ring_w = 3

        self._radius += (target_r - self._radius) * 0.2
        r = int(self._radius)

        # Ring
        rr = r + 4
        self._canvas.coords(self._ring, cx - rr, cy - rr, cx + rr, cy + rr)
        self._canvas.itemconfig(self._ring, outline=color, width=ring_w)

        # Dot
        self._canvas.coords(self._dot, cx - r, cy - r, cx + r, cy + r)
        self._canvas.itemconfig(self._dot, fill=color)

        # State label
        self._canvas.itemconfig(self._label, text=label)

        # Audio level bar
        bar_w = 60
        bar_h = 3
        bar_x = cx - bar_w // 2
        bar_y = int(self._WIN_SIZE * 0.75)
        filled_w = int(bar_w * self._rms)
        if filled_w > 0:
            self._canvas.coords(self._level_bar, bar_x, bar_y, bar_x + filled_w, bar_y + bar_h)
            self._canvas.itemconfig(self._level_bar, fill=color, outline="")
        else:
            self._canvas.coords(self._level_bar, 0, 0, 0, 0)

        # Mute X
        if show_mute:
            s = int(r * 0.4)
            self._canvas.coords(self._mute_bar, cx - s, cy - s, cx + s, cy + s)
            self._canvas.itemconfig(self._mute_bar, state="normal")
        else:
            self._canvas.itemconfig(self._mute_bar, state="hidden")

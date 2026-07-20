import math
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline


class MicButton:
    """Always-on-top transparent overlay with a ballooning mic indicator."""

    _WIN_SIZE = 100
    _TRANS = "#1e1e1e"  # transparent color (invisible pixels)
    _IDLE = "#555555"
    _LISTEN = "#22c55e"
    _THINK = "#eab308"
    _SPEAK = "#3b82f6"
    _MUTED = "#ef4444"
    _UNAVAIL = "#333333"

    def __init__(self, pipeline: "Pipeline"):
        self._pipeline = pipeline
        self._queue: Queue = Queue()
        self._rms = 0.0
        self._state = "idle"
        self._muted = False
        self._running = False
        self._drag_start = None
        self._radius = 14  # current visual radius

        pipeline.events.on("vad:level", self._on_level, async_=True)
        pipeline.events.on("pipeline:error", lambda e: self._queue.put(("state", "error")))

    def _on_level(self, rms: float, prob: float):
        self._queue.put(("rms", rms))
        self._queue.put(("state", "listening" if prob > 0.3 else "idle"))

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
        y = screen_h - self._WIN_SIZE - 80
        self._root.geometry(f"{self._WIN_SIZE}x{self._WIN_SIZE}+{x}+{y}")

        self._canvas = tk.Canvas(
            self._root, width=self._WIN_SIZE, height=self._WIN_SIZE,
            bg=self._TRANS, highlightthickness=0,
        )
        self._canvas.pack()

        cx = cy = self._WIN_SIZE // 2
        r = self._radius
        self._ring = self._canvas.create_oval(
            cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4,
            outline="", width=0,
        )
        self._dot = self._canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=self._IDLE, outline="",
        )

        # Drag support
        self._canvas.tag_bind(self._dot, "<Button-1>", self._on_drag_start)
        self._canvas.tag_bind(self._dot, "<B1-Motion>", self._on_drag_move)
        self._canvas.tag_bind(self._dot, "<ButtonRelease-1>", self._on_drag_end)
        # Also allow drag on ring (outer glow area)
        self._canvas.tag_bind(self._ring, "<Button-1>", self._on_drag_start)
        self._canvas.tag_bind(self._ring, "<B1-Motion>", self._on_drag_move)
        self._canvas.tag_bind(self._ring, "<ButtonRelease-1>", self._on_drag_end)

        self._poll()
        self._root.mainloop()

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
        # Only toggle mute if mouse barely moved (click, not drag)
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
                    self._rms = min(1.0, value * 10)
                elif kind == "state":
                    self._state = value
        except Empty:
            pass

        self._redraw()
        if self._running:
            self._root.after(50, self._poll)

    def _redraw(self):
        color = self._IDLE
        intensity = 0.3
        target_r = 14  # base radius

        if self._state == "error":
            color = self._MUTED
            target_r = 14
            intensity = 0.3 + 0.3 * math.sin(time.monotonic() * 4)
        elif self._state == "unavail" or self._pipeline.status["asr"] == "unavailable":
            color = self._UNAVAIL
            target_r = 14
        elif self._muted or self._pipeline.status["asr"] == "muted":
            color = self._MUTED
            intensity = 0.5
            target_r = 16
        elif self._state == "think":
            color = self._THINK
            intensity = 0.6 + 0.3 * math.sin(time.monotonic() * 3)
            target_r = 18 + 4 * math.sin(time.monotonic() * 2)
        elif self._state == "speak":
            color = self._SPEAK
            intensity = 0.5 + 0.4 * self._rms
            target_r = 18 + 22 * self._rms
        elif self._state == "listening":
            color = self._LISTEN
            intensity = 0.4 + 0.5 * self._rms
            target_r = 14 + 25 * self._rms

        # Smooth radius interpolation
        self._radius = self._radius * 0.8 + target_r * 0.2
        r = int(self._radius)
        cx = cy = self._WIN_SIZE // 2

        r_, g_, b_ = self._hex_to_rgb(color)
        r_ = int(r_ * intensity)
        g_ = int(g_ * intensity)
        b_ = int(b_ * intensity)
        fill = f"#{r_:02x}{g_:02x}{b_:02x}"

        self._canvas.coords(self._dot, cx - r, cy - r, cx + r, cy + r)
        self._canvas.itemconfig(self._dot, fill=fill)

        # Glow ring (balloons with audio)
        glow_r = r + 4 + int(8 * self._rms)
        self._canvas.coords(self._ring, cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r)
        glow_alpha = 0.15 + 0.3 * self._rms
        gr, gg, gb = self._hex_to_rgb(color)
        gr = int(gr * glow_alpha)
        gg = int(gg * glow_alpha)
        gb = int(gb * glow_alpha)
        self._canvas.itemconfig(self._ring, fill=f"#{gr:02x}{gg:02x}{gb:02x}")

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

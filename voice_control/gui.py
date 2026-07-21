import math
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline


class MicButton:
    """Always-on-top transparent overlay with an animated mic indicator."""

    _WIN_SIZE = 120
    _TRANS = "#1e1e1e"
    _IDLE = "#555555"
    _LISTEN = "#22c55e"
    _THINK = "#f59e0b"
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
        self._radius = 14.0
        self._pulse = 0.0
        self._target_rms = 0.0
        self._hover = False

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

        # Glow layers (outer → inner)
        self._glow_layers = []
        for i in range(4):
            layer = self._canvas.create_oval(0, 0, 0, 0, fill="", outline="", width=0)
            self._glow_layers.append(layer)

        # Main dot
        self._dot = self._canvas.create_oval(0, 0, 0, 0, fill=self._IDLE, outline="")

        # Mic icon (drawn with canvas primitives)
        self._mic = self._canvas.create_text(
            cx, cy, text="🎤", font=("Segoe UI", 14),
            fill="#ffffff", anchor="center",
        )

        # Drag + click
        for tag in (self._dot, *self._glow_layers, self._mic):
            self._canvas.tag_bind(tag, "<Button-1>", self._on_drag_start)
            self._canvas.tag_bind(tag, "<B1-Motion>", self._on_drag_move)
            self._canvas.tag_bind(tag, "<ButtonRelease-1>", self._on_drag_end)
            self._canvas.tag_bind(tag, "<Enter>", lambda e: setattr(self, "_hover", True))
            self._canvas.tag_bind(tag, "<Leave>", lambda e: setattr(self, "_hover", False))

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

        # Smooth RMS
        self._rms += (self._target_rms - self._rms) * 0.3

        self._redraw()
        if self._running:
            self._root.after(33, self._poll)  # ~30 fps

    def _redraw(self):
        t = time.monotonic()
        cx = cy = self._WIN_SIZE // 2

        color = self._IDLE
        target_r = 14.0
        glow_intensity = 0.0
        mic_color = "#ffffffcc"

        if self._state == "error":
            color = self._MUTED
            target_r = 14
            glow_intensity = 0.3 + 0.2 * math.sin(t * 4)
        elif self._state == "unavail" or self._pipeline.status["asr"] == "unavailable":
            color = self._UNAVAIL
            target_r = 14
        elif self._muted or self._pipeline.status["asr"] == "muted":
            color = self._MUTED
            target_r = 16
            glow_intensity = 0.3
            mic_color = "#ffffff88"
        elif self._state == "think":
            color = self._THINK
            target_r = 18 + 3 * math.sin(t * 2)
            glow_intensity = 0.5 + 0.3 * math.sin(t * 3)
        elif self._state == "speak":
            color = self._SPEAK
            target_r = 18 + 22 * self._rms
            glow_intensity = 0.4 + 0.5 * self._rms
        elif self._state == "listening":
            color = self._LISTEN
            target_r = 14 + 25 * self._rms
            glow_intensity = 0.2 + 0.6 * self._rms
        else:
            target_r = 14
            glow_intensity = 0.15 + 0.08 * math.sin(t * 1.5)

        # Smooth radius
        self._radius += (target_r - self._radius) * 0.2
        r = int(self._radius)

        # Main dot
        self._canvas.coords(self._dot, cx - r, cy - r, cx + r, cy + r)
        self._canvas.itemconfig(self._dot, fill=color)

        # Glow layers (decresing opacity outward)
        for i, layer in enumerate(self._glow_layers):
            spread = 3 + i * 5
            alpha = glow_intensity * (0.5 / (i + 1))
            gr, gg, gb = self._hex_to_rgb(color)
            gr = min(255, int(gr * alpha + 255 * (1 - alpha) * 0.05))
            gg = min(255, int(gg * alpha + 255 * (1 - alpha) * 0.05))
            gb = min(255, int(gb * alpha + 255 * (1 - alpha) * 0.05))
            fill = f"#{gr:02x}{gg:02x}{gb:02x}"
            gr2 = int(r + spread)
            self._canvas.coords(layer, cx - gr2, cy - gr2, cx + gr2, cy + gr2)
            self._canvas.itemconfig(layer, fill=fill)

        # Mic icon
        self._canvas.itemconfig(self._mic, fill=mic_color)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

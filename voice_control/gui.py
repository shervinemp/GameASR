import math
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline


class MicButton:

    SIZE = 60
    _BG = "#1e1e1e"
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

        # Subscribe to pipeline events
        pipeline.events.on("vad:level", self._on_level, async_=True)
        pipeline.events.on("pipeline:error", lambda e: self._queue.put(("state", "error")))

    def _on_level(self, rms: float, prob: float):
        self._queue.put(("rms", rms))
        self._queue.put(("state", "listening" if prob > 0.3 else "idle"))

    def _update_status(self):
        status = self._pipeline.status
        if status["asr"] == "unavailable":
            self._queue.put(("state", "unavail"))
        elif status["llm"] == "generating":
            self._queue.put(("state", "think"))
        elif status["tts"] == "speaking":
            self._queue.put(("state", "speak"))
        elif status["asr"] == "muted":
            self._queue.put(("state", "muted"))
        self._pipeline.events.emit("gui:update", status)

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
        self._root.configure(bg=self._BG)
        self._root.geometry(f"{self.SIZE}x{self.SIZE}+{self._root.winfo_screenwidth() - self.SIZE - 20}+{self._root.winfo_screenheight() - self.SIZE - 60}")

        self._canvas = tk.Canvas(
            self._root, width=self.SIZE, height=self.SIZE,
            bg=self._BG, highlightthickness=0,
        )
        self._canvas.pack()
        self._canvas.bind("<Button-1>", lambda e: self._toggle_mute())

        self._ring = self._canvas.create_oval(4, 4, self.SIZE - 4, self.SIZE - 4, outline="", width=0)
        self._dot = self._canvas.create_oval(
            self.SIZE * 0.3, self.SIZE * 0.3,
            self.SIZE * 0.7, self.SIZE * 0.7,
            fill=self._IDLE, outline="",
        )

        self._poll()
        self._root.mainloop()

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

        if self._state == "unavail" or self._pipeline.status["asr"] == "unavailable":
            color = self._UNAVAIL
        elif self._muted or self._pipeline.status["asr"] == "muted":
            color = self._MUTED
            intensity = 0.5
        elif self._state == "think":
            color = self._THINK
            intensity = 0.6 + 0.3 * math.sin(time.monotonic() * 3)
        elif self._state == "speak":
            color = self._SPEAK
            intensity = 0.5 + 0.4 * self._rms
        elif self._state == "listening":
            color = self._LISTEN
            intensity = 0.4 + 0.5 * self._rms

        r, g, b = self._hex_to_rgb(color)
        r = int(r * intensity)
        g = int(g * intensity)
        b = int(b * intensity)
        fill = f"#{r:02x}{g:02x}{b:02x}"

        self._canvas.itemconfig(self._dot, fill=fill)

        # Outer glow ring
        glow = max(1, int(12 * self._rms)) if self._rms > 0.05 else 0
        ring_color = fill if glow > 0 else self._BG
        self._canvas.itemconfig(self._ring, fill=ring_color, width=glow)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

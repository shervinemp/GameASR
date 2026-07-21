import math
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline


class MicButton:
    """Transparent overlay — glass orb + ring audio indicator."""

    _WIN_SIZE = 96
    _TRANS = "#1e1e1e"
    _COLORS = {
        "idle":      "#6b7280",
        "listening": "#22c55e",
        "think":     "#eab308",
        "speak":     "#3b82f6",
        "muted":     "#ef4444",
        "error":     "#ef4444",
        "unavail":   "#374151",
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
        self._radius = 10.0
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
        y = screen_h - self._WIN_SIZE - 80
        self._root.geometry(f"{self._WIN_SIZE}x{self._WIN_SIZE}+{x}+{y}")

        self._canvas = tk.Canvas(
            self._root, width=self._WIN_SIZE, height=self._WIN_SIZE,
            bg=self._TRANS, highlightthickness=0,
        )
        self._canvas.pack()

        cx = cy = self._WIN_SIZE // 2

        # Glass orb layers
        self._glow = self._canvas.create_oval(0, 0, 0, 0, fill="", outline="", width=0)
        self._ring = self._canvas.create_oval(0, 0, 0, 0, fill="", outline="", width=0)
        self._orb = self._canvas.create_oval(0, 0, 0, 0, fill="", outline="", width=0)
        self._highlight = self._canvas.create_oval(0, 0, 0, 0, fill="", outline="", width=0)
        self._status = self._canvas.create_text(
            cx, cy + 22, text="", font=("Segoe UI", 6, "bold"),
            fill="#ffffffaa", anchor="center",
        )

        for tag in (self._glow, self._ring, self._orb, self._highlight, self._status):
            self._canvas.tag_bind(tag, "<Button-1>", self._on_drag_start)
            self._canvas.tag_bind(tag, "<B1-Motion>", self._on_drag_move)
            self._canvas.tag_bind(tag, "<ButtonRelease-1>", self._on_drag_end)

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

        self._rms += (self._target_rms - self._rms) * 0.3
        self._anim += 1
        self._redraw()
        if self._running:
            self._root.after(33, self._poll)

    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    def _draw_mic(self, cx, cy, r):
        pass

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
        r_, g_, b_ = self._hex_to_rgb(color)

        target_r = 10.0
        ring_w = 0
        ring_alpha = 0.0
        glow_alpha = 0.0
        label_text = ""
        hl_alpha = 0.0

        if state == "error":
            target_r = 10
            ring_w = 2
            ring_alpha = 0.5 + 0.3 * math.sin(t * 4)
            glow_alpha = ring_alpha * 0.5
            label_text = "ERR"
        elif state == "unavail":
            target_r = 10
            label_text = "OFF"
        elif state == "muted":
            target_r = 12
            ring_w = 2
            ring_alpha = 0.3
            label_text = "MUTED"
        elif state == "think":
            target_r = 14 + 3 * math.sin(t * 2.5)
            ring_w = 3
            ring_alpha = 0.4 + 0.3 * math.sin(t * 3)
            glow_alpha = ring_alpha * 0.6
            hl_alpha = 0.3
            label_text = "THINK"
        elif state == "speak":
            target_r = 14 + 20 * self._rms
            ring_w = 3
            ring_alpha = 0.3 + 0.5 * self._rms
            glow_alpha = ring_alpha * 0.7
            hl_alpha = 0.4
            label_text = "SPEAK"
        elif state == "listening":
            target_r = 10 + 24 * self._rms
            ring_w = 3
            ring_alpha = 0.2 + 0.6 * self._rms
            glow_alpha = ring_alpha * 0.6
            hl_alpha = 0.2 + 0.3 * self._rms
            label_text = "LISTEN"
        else:
            target_r = 10
            ring_w = 2
            ring_alpha = 0.12 + 0.05 * math.sin(t * 1.2)
            glow_alpha = 0.04 + 0.02 * math.sin(t * 1.2)

        self._radius = self._lerp(self._radius, target_r, 0.2)
        r = int(self._radius)

        def _alpha_color(alpha):
            return f"#{int(r_*alpha):02x}{int(g_*alpha):02x}{int(b_*alpha):02x}"

        # Glow
        gr = r + 10 + int(10 * ring_alpha)
        self._canvas.coords(self._glow, cx - gr, cy - gr, cx + gr, cy + gr)
        self._canvas.itemconfig(self._glow, fill=_alpha_color(glow_alpha))

        # Ring
        rr = r + 2 + ring_w
        self._canvas.coords(self._ring, cx - rr, cy - rr, cx + rr, cy + rr)
        self._canvas.itemconfig(self._ring, fill=_alpha_color(ring_alpha * 0.3),
                                outline=_alpha_color(ring_alpha), width=ring_w)

        # Orb
        self._canvas.coords(self._orb, cx - r, cy - r, cx + r, cy + r)
        self._canvas.itemconfig(self._orb, fill=color)

        # Highlight (glass reflection)
        hlr = int(r * 0.6)
        hx = cx - int(r * 0.25)
        hy = cy - int(r * 0.25)
        self._canvas.coords(self._highlight, hx - hlr, hy - hlr, hx + hlr, hy + hlr)
        self._canvas.itemconfig(self._highlight, fill=_alpha_color(hl_alpha * 0.4))

        # Status text
        self._canvas.itemconfig(self._status, text=label_text)

    @staticmethod
    def _hex_to_rgb(hex_color):
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

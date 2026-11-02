"""Show an ``Expression`` on the duck: antenna servos, eye LEDs, sounds, screen eyes.

Each output is optional and turned on by ``expression_features`` in
duck_config.json (``antennas``, ``eyes``, ``speaker``, ``screen_eyes``). An
output whose hardware fails to open prints one warning and is skipped, so a
missing LED never stops the walk.

Writes are cheap and rate-limited (antennas and LEDs only when the value
changes, at most ``rate_hz``), and sounds play in a background thread, so
``apply()`` can be called from the 50 Hz loop.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import List, Optional, Tuple

DEFAULT_EXPRESSION_CFG = {
    # pins from docs/WIRING.md
    "left_antenna": ["A31", 7],  # [pin, PWM id]
    "right_antenna": ["B2", 4],
    "left_eye": ["B3", 5],
    "right_eye": ["A30", 6],
    # antenna servo pulse in % duty at 50 Hz for antenna value -1 and +1
    # (5..10 % = 1..2 ms). Swap the two numbers to reverse a servo.
    "left_antenna_duty": [5.0, 10.0],
    "right_antenna_duty": [10.0, 5.0],
    "led_freq": 1000,
    "sounds_dir": "~/sounds",  # <name>.wav: happy, curious, startled, sleepy, hello, ok
    "volume": 70,
    "rate_hz": 25,
}


def antenna_duty(value: float, duty_range) -> float:
    """-1..1 -> servo duty %."""
    lo, hi = duty_range
    v = max(-1.0, min(1.0, value))
    return lo + (hi - lo) * (v + 1) / 2


class Expressions:
    def __init__(self, config, mood_slot=None, pwm_module=None, audio_module=None, pinmap_module=None):
        """config: DuckConfig. mood_slot: shared value read by the vision process to draw screen eyes."""
        self.cfg = {**DEFAULT_EXPRESSION_CFG, **config.maixcam2.get("expression", {})}
        self.mood_slot = mood_slot if config.feature("screen_eyes") else None
        self.min_dt = 1.0 / self.cfg["rate_hz"]
        self._last_t = 0.0
        self._last = {}
        self.antennas = []
        self.eyes = []
        want_antennas = config.feature("antennas")
        want_eyes = config.feature("eyes")
        if want_antennas or want_eyes:
            try:
                if pwm_module is None:
                    from maix import pinmap as pinmap_module
                    from maix import pwm as pwm_module
                if want_antennas:
                    for side in ("left", "right"):
                        pin, pid = self.cfg[f"{side}_antenna"]
                        pinmap_module.set_pin_function(pin, f"PWM{pid}")
                        mid = antenna_duty(0.0, self.cfg[f"{side}_antenna_duty"])
                        self.antennas.append(pwm_module.PWM(pid, freq=50, duty=mid, enable=True))
                if want_eyes:
                    for side in ("left", "right"):
                        pin, pid = self.cfg[f"{side}_eye"]
                        pinmap_module.set_pin_function(pin, f"PWM{pid}")
                        self.eyes.append(pwm_module.PWM(pid, freq=self.cfg["led_freq"], duty=0, enable=True))
            except Exception as e:
                print(f"[expression] antennas/eye LEDs off: {e}")
                self.antennas, self.eyes = [], []

        self._sounds: Optional[queue.Queue] = None
        if config.feature("speaker"):
            self._audio = audio_module
            self._sounds_dir = os.path.expanduser(self.cfg["sounds_dir"])
            self._sounds = queue.Queue(maxsize=2)
            threading.Thread(target=self._sound_worker, daemon=True).start()

    # -- sound --------------------------------------------------------------
    def sound_path(self, name: str) -> str:
        return os.path.join(self._sounds_dir, f"{name}.wav")

    def _sound_worker(self):
        try:
            # keep audio off core 0, where the walk loop runs
            os.sched_setaffinity(0, {1})
        except (AttributeError, OSError):
            pass
        audio = self._audio
        if audio is None:
            try:
                from maix import audio
            except Exception as e:
                print(f"[expression] speaker off: {e}")
                return
        while True:
            name = self._sounds.get()
            if name is None:
                return
            path = self.sound_path(name)
            if not os.path.exists(path):
                continue
            try:
                p = audio.Player(path)
                p.volume(self.cfg["volume"])
                p.play()
            except Exception as e:
                print(f"[expression] could not play {path}: {e}")

    def play(self, name: str) -> None:
        if self._sounds is None:
            return
        try:
            self._sounds.put_nowait(name)
        except queue.Full:
            pass  # never queue up a backlog of quacks

    # -- per-step output -----------------------------------------------------
    def apply(self, expr, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        if expr.sound:
            self.play(expr.sound)
        if self.mood_slot is not None:
            from .behavior import MOOD_ID

            self.mood_slot.value = MOOD_ID.get(expr.mood, 0)
        if now - self._last_t < self.min_dt:
            return
        self._last_t = now
        for out, value, side in zip(self.antennas, expr.antennas, ("left", "right")):
            duty = round(antenna_duty(value, self.cfg[f"{side}_antenna_duty"]), 2)
            self._write(out, f"antenna_{side}", duty)
        for out, side in zip(self.eyes, ("left", "right")):
            self._write(out, f"eye_{side}", round(100.0 * max(0.0, min(1.0, expr.eye_leds)), 1))

    def _write(self, out, key, duty):
        if self._last.get(key) == duty:
            return
        self._last[key] = duty
        try:
            out.duty(duty)
        except Exception as e:
            print(f"[expression] {key}: {e}")

    def stop(self):
        if self._sounds is not None:
            try:
                self._sounds.put_nowait(None)
            except queue.Full:
                pass
        for out in self.antennas + self.eyes:
            try:
                out.disable()
            except Exception:
                pass


# -- screen eyes ---------------------------------------------------------------
Shape = Tuple[str, Tuple[int, ...]]  # ("ellipse", (cx, cy, rx, ry)) or ("rect", (x, y, w, h)) in black


def eye_shapes(mood: str, t: float, width: int, height: int) -> List[Shape]:
    """Two cartoon eyes for the screen, as filled shapes. White ellipses are the
    eyes; black rects drawn after them act as eyelids. Pure geometry, so it is
    unit-tested; the vision process draws it with maix.image."""
    r = int(min(width, height) * 0.22)
    cy = height // 2
    xs = (width // 2 - int(r * 1.5), width // 2 + int(r * 1.5))
    # blink for 0.15 s every 4 s
    blink = (t % 4.0) < 0.15
    shapes: List[Shape] = []
    for i, cx in enumerate(xs):
        if mood == "curious" and i == 0:
            shapes.append(("ellipse", (cx, cy, int(r * 0.8), int(r * 0.8))))
        elif mood == "startled":
            shapes.append(("ellipse", (cx, cy, int(r * 1.15), int(r * 1.15))))
        else:
            shapes.append(("ellipse", (cx, cy, r, r)))
    for cx in xs:
        lid = 0.0
        if blink or mood == "sleepy":
            lid = 0.75 if not blink else 1.0
        elif mood == "happy":
            # happy: lower lid pushes up into a smile-shaped eye
            shapes.append(("rect", (cx - r - 2, cy + int(r * 0.2), 2 * r + 4, r + 4)))
        elif mood == "neutral":
            lid = 0.2
        if lid > 0:
            shapes.append(("rect", (cx - r - 2, cy - r - 2, 2 * r + 4, int((2 * r + 4) * lid))))
    # pupils look up a bit when attentive
    if mood in ("attentive", "curious"):
        for cx in xs:
            pr = max(2, r // 3)
            shapes.append(("pupil", (cx, cy - r // 4, pr, pr)))
    return shapes


def draw_eyes(img, mood: str, t: float, image_module) -> None:
    """Draw eye_shapes on a maix.image.Image."""
    w, h = img.width(), img.height()
    img.draw_rect(0, 0, w, h, image_module.COLOR_BLACK, -1)
    for kind, a in eye_shapes(mood, t, w, h):
        if kind == "ellipse":
            img.draw_ellipse(a[0], a[1], a[2], a[3], 0, 0, 360, image_module.COLOR_WHITE, -1)
        elif kind == "pupil":
            img.draw_circle(a[0], a[1], a[2], image_module.COLOR_BLACK, -1)
        else:
            img.draw_rect(a[0], a[1], a[2], a[3], image_module.COLOR_BLACK, -1)


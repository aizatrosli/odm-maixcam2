"""Sources of walking commands: [vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll].

Step 2 adds a camera-driven source here with the same ``get()`` interface.
"""

from __future__ import annotations

import threading
import time

import numpy as np

X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]


class ZeroCommands:
    """Walk in place. Useful for first tests."""

    def get(self):
        return np.zeros(7), {}

    def stop(self):
        pass


class GamepadCommands:
    """Xbox-style gamepad over Bluetooth, via pygame (same mapping as the Pi runtime).

    Buttons: A pauses/resumes, LB held = faster gait.
    """

    def __init__(self, freq: int = 20):
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise IOError("no gamepad found")
        self.pad = pygame.joystick.Joystick(0)
        self.pad.init()
        self.freq = freq
        self._lock = threading.Lock()
        self._cmd = np.zeros(7)
        self._events = {}
        self._running = True
        threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _scale(v, rng):
        return v * abs(rng[1] if v >= 0 else rng[0])

    def _worker(self):
        pg = self.pygame
        while self._running:
            events = {}
            for e in pg.event.get():
                if e.type == pg.JOYBUTTONDOWN:
                    if e.button == 0:
                        events["A"] = True
            cmd = np.zeros(7)
            cmd[0] = self._scale(-self.pad.get_axis(1), X_RANGE)
            cmd[1] = self._scale(-self.pad.get_axis(0), Y_RANGE)
            cmd[2] = self._scale(-self.pad.get_axis(2), YAW_RANGE)
            lb = self.pad.get_numbuttons() > 6 and self.pad.get_button(6)
            with self._lock:
                self._cmd = np.around(cmd, 3)
                for k in events:
                    self._events[k] = True
                self._events["LB_held"] = bool(lb)
            time.sleep(1 / self.freq)

    def get(self):
        with self._lock:
            ev = dict(self._events)
            self._events = {"LB_held": ev.get("LB_held", False)}
            return self._cmd.copy(), ev

    def stop(self):
        self._running = False

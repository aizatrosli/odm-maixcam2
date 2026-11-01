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


class FollowCommands:
    """Follow a person seen by the camera (step 2).

    The detector runs on the NPU in a separate process; this object only reads
    its latest boxes and runs the follow controller, so ``get()`` is cheap.
    A gamepad, if given, still pauses with A and overrides the walk while a
    stick is pushed.
    """

    def __init__(self, vision_cfg: dict, gamepad=None, detector=None, clock=time.monotonic):
        from .vision.follow import FollowController, FollowParams

        params = FollowParams(**vision_cfg.get("follow", {}))
        self.controller = FollowController(params)
        if detector is None:
            from .vision.detector import DetectorProcess

            detector = DetectorProcess(
                model_path=vision_cfg.get("model_path", "/root/models/yolo11n.mud"),
                class_name=vision_cfg.get("class_name", "person"),
                conf=vision_cfg.get("conf_threshold", 0.5),
                show=vision_cfg.get("show", False),
            )
        self.detector = detector
        self.gamepad = gamepad
        self.clock = clock
        self._last_t = None

    def get(self):
        now = self.clock()
        dt = 0.02 if self._last_t is None else min(0.1, now - self._last_t)
        self._last_t = now
        new, _, dets = self.detector.latest()
        cmd = self.controller.update(dets if new else [], now, dt)
        events = {}
        if self.gamepad is not None:
            pad_cmd, events = self.gamepad.get()
            if np.any(np.abs(pad_cmd[:3]) > 1e-3):
                cmd[:3] = pad_cmd[:3]
        return cmd, events

    @property
    def state(self):
        return self.controller.state

    def extra_obs(self, names):
        """Camera features for policies whose profile lists extra inputs.

        Known names: target_visible, target_bearing_sin, target_bearing_cos,
        target_size (box height, 0..1). Unknown names raise, so a profile
        that needs a feature this runtime cannot provide fails at start-up.
        """
        import math

        from .vision.follow import pixel_to_angle

        t = self.controller.target
        visible = t is not None
        bearing = pixel_to_angle(t.cx, self.controller.params.hfov_deg) if visible else 0.0
        feats = {
            "target_visible": float(visible),
            "target_bearing_sin": math.sin(bearing),
            "target_bearing_cos": math.cos(bearing) if visible else 0.0,
            "target_size": t.h if visible else 0.0,
        }
        try:
            return [feats[n] for n in names]
        except KeyError as e:
            raise KeyError(f"camera feature {e} is not provided by FollowCommands") from None

    def stop(self):
        self.detector.stop()
        if self.gamepad is not None:
            self.gamepad.stop()

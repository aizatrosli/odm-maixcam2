"""Sources of walking commands: [vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll].

FollowCommands is the camera-driven source (step 2); with ``interact=True`` it
also reacts to hand gestures and faces, and drives the duck's expressions.
"""

from __future__ import annotations

import threading
import time

import numpy as np

X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]

# Interaction mode (walk.py --interact), see maixcam2.vision.interact in config.py
DEFAULT_INTERACT = {
    "hand_model": "/root/models/hand_landmarks.mud",
    "faces": True,
    "face_detect_model": "/root/models/yolov8n_face.mud",
    "face_feature_model": "/root/models/insghtface_webface_r50.mud",
    "faces_path": "/root/faces.bin",
    "face_threshold": 0.85,
    "hands_every": 2,  # run hand landmarks every N camera frames
    "faces_every": 5,  # run face recognition every N camera frames
    "camera_size": None,  # [w, h]; None = the person detector's input size
    "screen_eyes": False,  # set from expression_features.screen_eyes by walk.py
    "behavior": {},  # BehaviorParams overrides, see odm_maix/behavior.py
    "gestures": {},  # GestureParams overrides, see odm_maix/vision/gestures.py
}


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

    def __init__(
        self,
        vision_cfg: dict,
        gamepad=None,
        detector=None,
        clock=time.monotonic,
        interact: bool = False,
        expressions=None,
    ):
        from .vision.follow import FollowController, FollowParams

        params = FollowParams(**vision_cfg.get("follow", {}))
        self.controller = FollowController(params)
        interact_cfg = {**DEFAULT_INTERACT, **vision_cfg.get("interact", {})} if interact else None
        if detector is None:
            from .vision.detector import DetectorProcess

            detector = DetectorProcess(
                model_path=vision_cfg.get("model_path", "/root/models/yolo11n.mud"),
                class_name=vision_cfg.get("class_name", "person"),
                conf=vision_cfg.get("conf_threshold", 0.5),
                show=vision_cfg.get("show", False),
                interact=interact_cfg,
            )
        self.detector = detector
        self.gamepad = gamepad
        self.clock = clock
        self._last_t = None

        # interaction mode: gestures, faces, moods
        self.behavior = self.gestures = None
        self.expressions = expressions
        self.expression = None
        self.last_gesture = None
        self.last_face = None
        if interact_cfg is not None:
            from .behavior import Behavior, BehaviorParams
            from .vision.gestures import GestureParams, GestureTracker

            self.behavior = Behavior(BehaviorParams(**interact_cfg.get("behavior", {})))
            self.gestures = GestureTracker(GestureParams(**interact_cfg.get("gestures", {})))

    def get(self):
        now = self.clock()
        dt = 0.02 if self._last_t is None else min(0.1, now - self._last_t)
        self._last_t = now
        new, _, dets = self.detector.latest()
        cmd = self.controller.update(dets if new else [], now, dt)
        if self.behavior is not None:
            cmd = self._interact(cmd, now, dt)
        events = {}
        if self.gamepad is not None:
            pad_cmd, events = self.gamepad.get()
            if np.any(np.abs(pad_cmd[:3]) > 1e-3):
                cmd[:3] = pad_cmd[:3]
        return cmd, events

    def _interact(self, cmd, now, dt):
        gesture = None
        new_h, hand = self.detector.latest_hand()
        if new_h:
            gesture = self.gestures.update(now, hand.pose if hand else None, hand.cx if hand else 0.5)
            if gesture:
                self.last_gesture = (now, gesture)
        face_name, known = None, False
        new_f, face = self.detector.latest_face()
        if new_f and face is not None:
            face_name, known = face.name, face.known
            self.last_face = (now, face.name)
        c = self.controller
        target_h = c.target.h if (c.target is not None and c.state != "searching") else None
        cmd, expr = self.behavior.update(now, dt, cmd, target_h, gesture, face_name, known)
        self.expression = expr
        if self.expressions is not None:
            self.expressions.apply(expr, now)
        return cmd

    @property
    def state(self):
        if self.behavior is not None:
            return f"{self.controller.state}/{self.behavior.mode}/{self.behavior.mood}"
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
        if self.expressions is not None:
            self.expressions.stop()
        if self.gamepad is not None:
            self.gamepad.stop()

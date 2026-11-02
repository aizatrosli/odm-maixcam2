"""Moods and gesture commands on top of follow mode.

Pure Python and numpy, no maix imports, so it is unit-tested on a PC.

Each control step, ``Behavior.update`` takes the follow controller's command,
what the camera saw (person, hand gesture, face) and returns:

- the walking command to send, after gesture commands are applied
  ("stay" scales walking to zero, "come" lets follow mode walk again);
- an ``Expression``: the mood, antenna positions, eye LED brightness, a
  one-shot sound name, and small head offsets that animate the mood.

Head animations only play while the duck stands still. The stock walking
policy never trained with the head moving, so they ramp to zero as soon as
it walks (see training/README.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

MOODS = ["neutral", "attentive", "happy", "curious", "startled", "sleepy"]
MOOD_ID = {m: i for i, m in enumerate(MOODS)}

# head offset limits: neck_pitch, head_pitch, head_yaw, head_roll (rad)
HEAD_LIMITS = np.array([0.3, 0.5, 0.5, 0.3])

DEFAULT_GESTURE_ACTIONS = {
    "wave": "come",  # start (or resume) following
    "stop": "stay",  # open palm held still: stand where you are
    "thumbs_up": "celebrate",
    "peace": "celebrate",
}


@dataclass
class Expression:
    mood: str = "neutral"
    antennas: tuple = (0.0, 0.0)  # left, right in -1 (down) .. 1 (up)
    eye_leds: float = 0.6  # 0..1
    sound: Optional[str] = None  # play once, e.g. "happy" -> <sounds_dir>/happy.wav
    head: np.ndarray = field(default_factory=lambda: np.zeros(4))  # added to cmd[3:7]


@dataclass
class BehaviorParams:
    start_mode: str = "stay"  # "stay": wait for a wave before walking; "follow": walk right away
    gesture_actions: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_GESTURE_ACTIONS))
    happy_time: float = 2.0
    curious_time: float = 1.5
    startled_time: float = 1.0
    sleepy_after: float = 30.0  # s with nobody around (while staying) before dozing off
    greet_interval: float = 60.0  # s before the same known face is greeted again
    unknown_face_interval: float = 10.0  # s between "who's that?" reactions to strangers
    startle_jump: float = 0.25  # person box height growing this much ...
    startle_window: float = 0.3  # ... within this many seconds is startling
    walk_ramp: float = 2.0  # 1/s, how fast "stay" / "come" fade the walk in and out
    still_vx: float = 0.02  # below these the duck counts as standing: head animations allowed
    still_wz: float = 0.1
    max_head_rate: float = 1.5  # rad/s for the animation offsets
    animate_head: bool = True


def _slew_vec(cur: np.ndarray, target: np.ndarray, rate: float, dt: float) -> np.ndarray:
    step = rate * dt
    return cur + np.clip(target - cur, -step, step)


class Behavior:
    def __init__(self, params: Optional[BehaviorParams] = None):
        self.p = params or BehaviorParams()
        self.mode = "follow" if self.p.start_mode == "follow" else "stay"
        self.walk_gain = 1.0 if self.mode == "follow" else 0.0
        self.mood = "neutral"
        self.mood_until = 0.0
        self.mood_since = 0.0
        self.head = np.zeros(4)
        self.last_person: Optional[float] = None
        self.person_visible = False
        self.size_hist: List[tuple] = []
        self.greeted: Dict[str, float] = {}
        self.last_stranger = -1e9
        self.events: List[str] = []
        self.t0: Optional[float] = None

    # -- mood helpers -------------------------------------------------------
    def _set_mood(self, mood: str, now: float, duration: float, sound: Optional[str] = None) -> Optional[str]:
        # a startle is not interrupted by milder moods
        if self.mood == "startled" and now < self.mood_until and mood != "startled":
            return None
        changed = mood != self.mood
        self.mood = mood
        self.mood_since = now if changed else self.mood_since
        self.mood_until = now + duration
        self.events.append(f"mood:{mood}")
        return sound if sound is not None else (mood if changed else None)

    def _base_mood(self, now: float) -> str:
        if self.person_visible:
            return "attentive"
        idle = now - (self.last_person if self.last_person is not None else self.t0)
        if self.mode == "stay" and idle > self.p.sleepy_after:
            return "sleepy"
        return "neutral"

    def _action(self, action: str, now: float) -> Optional[str]:
        self.events.append(f"action:{action}")
        if action == "come":
            self.mode = "follow"
            return self._set_mood("happy", now, self.p.happy_time)
        if action == "stay":
            self.mode = "stay"
            return self._set_mood("attentive", now, 1.0, sound="ok")
        if action == "celebrate":
            return self._set_mood("happy", now, self.p.happy_time)
        return None

    # -- main entry ---------------------------------------------------------
    def update(
        self,
        now: float,
        dt: float,
        cmd: np.ndarray,
        target_h: Optional[float] = None,
        gesture: Optional[str] = None,
        face: Optional[str] = None,
        face_known: bool = False,
    ):
        """
        cmd: follow controller output (7 values), not modified.
        target_h: box height of the followed person (None if nobody is tracked).
        gesture: a GestureTracker event, or None.
        face: a face seen this frame: its name if face_known, else any non-empty string.
        Returns (cmd_out, Expression).
        """
        p = self.p
        if self.t0 is None:
            self.t0 = now
        self.events = []
        sound = None

        # person presence and startle
        was_visible = self.person_visible
        self.person_visible = target_h is not None
        if self.person_visible:
            self.last_person = now
            self.size_hist.append((now, target_h))
            self.size_hist = [(t, h) for t, h in self.size_hist if now - t <= p.startle_window]
            if target_h - min(h for _, h in self.size_hist) >= p.startle_jump:
                sound = self._set_mood("startled", now, p.startled_time) or sound
                self.size_hist = [(now, target_h)]
            elif not was_visible and self.mood in ("neutral", "sleepy"):
                sound = self._set_mood("curious", now, p.curious_time) or sound
        else:
            self.size_hist = []

        # gesture commands
        if gesture:
            self.events.append(f"gesture:{gesture}")
            action = p.gesture_actions.get(gesture)
            if action:
                sound = self._action(action, now) or sound

        # faces
        if face:
            if face_known:
                last = self.greeted.get(face, -1e9)
                if now - last > p.greet_interval:
                    self.greeted[face] = now
                    self.events.append(f"greet:{face}")
                    self._set_mood("happy", now, p.happy_time)
                    sound = "hello"
            elif now - self.last_stranger > p.unknown_face_interval:
                self.last_stranger = now
                sound = self._set_mood("curious", now, p.curious_time) or sound

        # timed moods fall back to the base mood
        if now >= self.mood_until:
            base = self._base_mood(now)
            if base != self.mood:
                if base == "sleepy":
                    sound = "sleepy"
                self.mood = base
                self.mood_since = now
                self.events.append(f"mood:{base}")

        # walking: "stay" fades the walk out, "come" fades it back in
        goal = 1.0 if self.mode == "follow" else 0.0
        self.walk_gain += float(np.clip(goal - self.walk_gain, -p.walk_ramp * dt, p.walk_ramp * dt))
        out = np.array(cmd, dtype=float)
        out[:3] *= self.walk_gain

        # head animation, only while standing
        standing = abs(out[0]) < p.still_vx and abs(out[2]) < p.still_wz
        target = self._head_animation(now) if (p.animate_head and standing) else np.zeros(4)
        self.head = _slew_vec(self.head, target, p.max_head_rate, dt)
        out[3:7] = np.clip(out[3:7] + self.head, -HEAD_LIMITS, HEAD_LIMITS)

        return out, Expression(
            mood=self.mood,
            antennas=self._antennas(now),
            eye_leds=self._eyes(now),
            sound=sound,
            head=self.head.copy(),
        )

    # -- what each mood looks like ------------------------------------------
    def _head_animation(self, now: float) -> np.ndarray:
        """Offsets for [neck_pitch, head_pitch, head_yaw, head_roll].
        head_pitch positive tilts the head down (FollowParams.head_pitch_sign = 1)."""
        t = now - self.mood_since
        m = self.mood
        if m == "happy":  # quick nods with a little wiggle
            return np.array([0.0, 0.15 * math.sin(2 * math.pi * 2.5 * t), 0.0, 0.12 * math.sin(2 * math.pi * 1.25 * t)])
        if m == "curious":  # head tilt
            return np.array([0.05, 0.0, 0.0, 0.25])
        if m == "startled":  # pull the head back
            return np.array([-0.15, -0.25, 0.0, 0.0])
        if m == "sleepy":  # droop, slow breathing bob
            return np.array([0.0, 0.3 + 0.05 * math.sin(2 * math.pi * 0.25 * t), 0.0, 0.0])
        return np.zeros(4)

    def _antennas(self, now: float) -> tuple:
        t = now - self.mood_since
        m = self.mood
        if m == "happy":
            w = 0.6 * math.sin(2 * math.pi * 3 * t)
            return (0.3 + w, 0.3 - w)
        return {
            "neutral": (0.0, 0.0),
            "attentive": (0.4, 0.4),
            "curious": (0.7, -0.2),
            "startled": (1.0, 1.0),
            "sleepy": (-0.8, -0.8),
        }.get(m, (0.0, 0.0))

    def _eyes(self, now: float) -> float:
        t = now - self.mood_since
        m = self.mood
        if m == "happy":
            return 0.7 + 0.3 * math.sin(2 * math.pi * 2 * t)
        if m == "sleepy":
            return 0.15
        return {"neutral": 0.6, "attentive": 1.0, "curious": 1.0, "startled": 1.0}.get(m, 0.6)

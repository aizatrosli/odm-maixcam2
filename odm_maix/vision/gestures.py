"""Hand poses from 21 hand keypoints, and gesture events over time.

Pure Python, no maix imports, so it is unit-tested on a PC.

Keypoints follow the MediaPipe order used by ``maix.nn.HandLandmarks``:
0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
(for each finger: MCP, PIP, DIP, tip). Only x and y are used, in any unit,
with y growing downward as in an image.

``classify_pose`` names the hand shape in one frame. ``GestureTracker`` turns
the per-frame poses into events: a pose has to be held for a moment before it
counts (so a hand passing through a shape does nothing), and a wave is an open
palm moving side to side.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

POSES = ["none", "open_palm", "fist", "point", "peace", "thumbs_up"]
POSE_ID = {name: i for i, name in enumerate(POSES)}

# finger -> (mcp, pip, tip)
_FINGERS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_extended(kp: Sequence[Tuple[float, float]], ratio: float = 1.15) -> dict:
    """Which fingers are straight. A finger is extended when its tip is clearly
    further from the wrist than its middle joint, which works at any hand
    rotation and scale."""
    w = kp[0]
    out = {}
    for name, (_, pip, tip) in _FINGERS.items():
        out[name] = _d(kp[tip], w) > ratio * _d(kp[pip], w)
    # thumb: tip away from the index knuckle compared with its own IP joint
    out["thumb"] = _d(kp[4], kp[5]) > 1.3 * _d(kp[3], kp[5])
    return out


def hand_size(kp: Sequence[Tuple[float, float]]) -> float:
    """Wrist to middle-finger knuckle: a scale that doesn't change with the pose."""
    return _d(kp[0], kp[9])


def classify_pose(kp: Sequence[Tuple[float, float]]) -> str:
    """One of POSES for 21 (x, y) keypoints."""
    if kp is None or len(kp) < 21:
        return "none"
    size = hand_size(kp)
    if size <= 1e-6:
        return "none"
    ext = fingers_extended(kp)
    four = [ext[f] for f in ("index", "middle", "ring", "pinky")]
    n = sum(four)
    if n == 4:
        return "open_palm"
    if n == 0:
        # thumbs up: thumb out and its tip well above the knuckles (image y is down)
        top_knuckle = min(kp[i][1] for i in (5, 9, 13, 17))
        if ext["thumb"] and kp[4][1] < top_knuckle - 0.5 * size:
            return "thumbs_up"
        return "fist"
    if four == [True, False, False, False]:
        return "point"
    if four == [True, True, False, False]:
        return "peace"
    return "none"


def keypoints_from_maix(points: Sequence[float]) -> list:
    """``obj.points`` from maix.nn.HandLandmarks (4 box corners, then 21 x,y,z) -> 21 (x, y)."""
    pts = list(points)
    base = 8
    if len(pts) < base + 21 * 3:
        return []
    return [(pts[base + 3 * i], pts[base + 3 * i + 1]) for i in range(21)]


@dataclass
class GestureParams:
    hold_time: float = 0.4  # s a pose must be held before it counts
    stop_hold_time: float = 0.8  # open palm held still this long -> "stop"
    still_range: float = 0.05  # palm counts as still if it moves less than this (image fraction)
    wave_window: float = 1.5  # s of history used to spot a wave
    wave_amplitude: float = 0.04  # each side-to-side swing must cover this much of the image
    wave_reversals: int = 2  # direction changes needed for a wave
    hand_timeout: float = 0.3  # s without a hand before the current pose is dropped
    cooldown: float = 1.5  # s before the same gesture can fire again


class GestureTracker:
    """Feed one hand observation per vision frame, get gesture events.

    Events: "wave", "stop" (open palm held still), "thumbs_up", "peace",
    "point", "fist". Each fires once, then re-arms when the pose changes or
    after ``cooldown``.
    """

    def __init__(self, params: Optional[GestureParams] = None):
        self.p = params or GestureParams()
        self.pose = "none"
        self.pose_since = 0.0
        self.last_hand = None
        self.fired = set()
        self.last_fire = {}
        self.track: Deque[Tuple[float, float]] = deque()

    def _fire(self, name: str, now: float) -> Optional[str]:
        if name in self.fired:
            return None
        if now - self.last_fire.get(name, -1e9) < self.p.cooldown:
            return None
        self.fired.add(name)
        self.last_fire[name] = now
        return name

    def _wave(self) -> bool:
        xs = [x for _, x in self.track]
        if len(xs) < 4:
            return False
        reversals, direction, extreme = 0, 0, xs[0]
        for x in xs[1:]:
            if direction >= 0 and x < extreme - self.p.wave_amplitude:
                reversals += direction > 0
                direction, extreme = -1, x
            elif direction <= 0 and x > extreme + self.p.wave_amplitude:
                reversals += direction < 0
                direction, extreme = 1, x
            elif (direction >= 0 and x > extreme) or (direction <= 0 and x < extreme):
                extreme = x
        return reversals >= self.p.wave_reversals

    def update(self, now: float, pose: Optional[str], cx: float = 0.5) -> Optional[str]:
        """pose: this frame's pose (None or "none" = no hand), cx: hand centre x in 0..1."""
        p = self.p
        if not pose or pose == "none":
            if self.last_hand is None or now - self.last_hand > p.hand_timeout:
                self.pose = "none"
                self.fired.clear()
                self.track.clear()
            return None
        self.last_hand = now

        if pose != self.pose:
            self.pose = pose
            self.pose_since = now
            self.fired.clear()
            if pose != "open_palm":
                self.track.clear()

        if pose == "open_palm":
            self.track.append((now, cx))
            while self.track and now - self.track[0][0] > p.wave_window:
                self.track.popleft()
            if self._wave():
                self.track.clear()
                self.pose_since = now  # a wave is not a held palm
                self.fired.discard("wave")
                return self._fire("wave", now)
            xs = [x for t, x in self.track if now - t <= p.stop_hold_time]
            still = xs and max(xs) - min(xs) < p.still_range
            if still and now - self.pose_since >= p.stop_hold_time:
                return self._fire("stop", now)
            return None

        if now - self.pose_since >= p.hold_time:
            return self._fire(pose, now)
        return None

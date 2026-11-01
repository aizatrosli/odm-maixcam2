"""Turn a tracked person into walking commands.

Pure Python and numpy, no maix imports, so it is unit-tested on a PC.

Commands are the policy's 7 inputs: [vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll],
kept inside the ranges the stock policy was trained on (vx +-0.15 m/s, wz +-1 rad/s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Policy command ranges (same as the gamepad mapping)
VX_MAX = 0.15
WZ_MAX = 1.0
HEAD_YAW_RANGE = (-0.5, 0.5)
HEAD_PITCH_RANGE = (-0.5, 0.5)  # conservative subset of the gamepad range


@dataclass
class Detection:
    """One box, normalised to the image: cx, cy, w, h in 0..1."""

    cx: float
    cy: float
    w: float
    h: float
    score: float = 1.0


@dataclass
class FollowParams:
    hfov_deg: float = 70.0  # camera horizontal field of view, measure yours
    vfov_deg: float = 55.0
    camera_on_head: bool = False  # True when the camera moves with head_yaw/head_pitch
    target_height: float = 0.55  # box height (fraction of image) at the distance to keep
    stop_height: float = 0.8  # closer than this -> stand still
    k_turn: float = 1.5  # wz per rad of bearing
    k_forward: float = 0.5  # vx per unit of box-height error
    walk_bearing_deg: float = 20.0  # only walk forward when roughly facing the person
    head_track: bool = True
    k_head: float = 0.6  # share of the bearing the head points at
    # Joint sign conventions: flip to -1 if the head turns away from the person
    head_yaw_sign: float = 1.0  # +1: positive head_yaw turns the head left
    head_pitch_sign: float = 1.0  # +1: positive head_pitch tilts the head down
    lost_timeout: float = 0.6  # s without a detection before giving up
    max_vx_rate: float = 0.3  # m/s^2, command slew limits
    max_wz_rate: float = 3.0  # rad/s^2
    max_head_rate: float = 2.0  # rad/s


def pixel_to_angle(u: float, fov_deg: float) -> float:
    """u in 0..1 (0.5 = centre) -> angle in rad, right/down positive."""
    half = math.radians(fov_deg) / 2
    return math.atan((u - 0.5) * 2 * math.tan(half))


def pick_target(dets: List[Detection], previous: Optional[Detection]) -> Optional[Detection]:
    """Keep following the same person: closest to the last target, else the biggest box."""
    if not dets:
        return None
    if previous is not None:
        return min(dets, key=lambda d: (d.cx - previous.cx) ** 2 + (d.cy - previous.cy) ** 2 - 0.5 * d.h * d.w)
    return max(dets, key=lambda d: d.w * d.h)


def _slew(current: float, target: float, max_rate: float, dt: float) -> float:
    step = max_rate * dt
    return current + float(np.clip(target - current, -step, step))


@dataclass
class FollowController:
    params: FollowParams = field(default_factory=FollowParams)

    def __post_init__(self):
        self.target: Optional[Detection] = None
        self.last_seen: Optional[float] = None
        self.cmd = np.zeros(7)
        self.state = "searching"

    def update(self, dets: List[Detection], now: float, dt: float) -> np.ndarray:
        p = self.params
        target = pick_target(dets, self.target)
        if target is not None:
            self.target = target
            self.last_seen = now

        lost = self.last_seen is None or (now - self.last_seen) > p.lost_timeout
        vx_t = wz_t = 0.0
        head_yaw_t = head_pitch_t = 0.0

        if lost:
            self.target = None
            self.state = "searching"
        else:
            t = self.target
            head_yaw, head_pitch = self.cmd[5], self.cmd[4]
            # bearing in the robot frame (right positive in the image)
            bearing = pixel_to_angle(t.cx, p.hfov_deg)
            elevation = pixel_to_angle(t.cy, p.vfov_deg)
            if p.camera_on_head:
                # the image moves with the head: head left shifts bearings left, head down shifts them down
                bearing -= p.head_yaw_sign * head_yaw
                elevation += p.head_pitch_sign * head_pitch
            # wz is left-positive (counter-clockwise), image x is right-positive
            wz_t = float(np.clip(-p.k_turn * bearing, -WZ_MAX, WZ_MAX))

            if t.h >= p.stop_height:
                self.state = "too_close"
            elif abs(bearing) < math.radians(p.walk_bearing_deg):
                vx_t = float(np.clip(p.k_forward * (p.target_height - t.h), 0.0, VX_MAX))
                self.state = "following" if vx_t > 0 else "arrived"
            else:
                self.state = "turning"

            if p.head_track:
                # bearing is right-positive, so head left (+) needs the opposite sign
                head_yaw_t = float(np.clip(-p.head_yaw_sign * p.k_head * bearing, *HEAD_YAW_RANGE))
                head_pitch_t = float(np.clip(p.head_pitch_sign * p.k_head * elevation, *HEAD_PITCH_RANGE))

        c = self.cmd
        c[0] = _slew(c[0], vx_t, p.max_vx_rate, dt)
        c[1] = 0.0
        c[2] = _slew(c[2], wz_t, p.max_wz_rate, dt)
        c[4] = _slew(c[4], head_pitch_t, p.max_head_rate, dt)
        c[5] = _slew(c[5], head_yaw_t, p.max_head_rate, dt)
        return c.copy()

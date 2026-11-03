"""Where the virtual people walk."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class WalkerParams:
    speed: Tuple[float, float] = (0.15, 0.45)  # m/s while walking
    turn_rate: float = 2.0  # rad/s
    pause_prob: float = 0.35  # chance of standing still at each waypoint
    pause_s: Tuple[float, float] = (1.0, 4.0)
    near: Tuple[float, float] = (0.9, 3.0)  # target: waypoints this far from the duck
    start: Tuple[float, float] = (1.2, 2.2)  # start this far in front of the duck
    start_bearing_deg: float = 20.0  # and within this angle of its heading


class PersonWalker:
    """Moves one mocap person between waypoints.

    ``follow_duck=True`` keeps waypoints around the duck (the target to
    follow); otherwise waypoints are anywhere in the room (a distractor).
    ``manual=True`` stops the walker so a viewer can place the person.
    """

    def __init__(self, params: WalkerParams, room_half: float, follow_duck: bool = True):
        self.p, self.half, self.follow_duck = params, room_half - 0.4, follow_duck
        self.manual = False
        self.x = self.y = self.yaw = 0.0
        self.goal = (0.0, 0.0)
        self.speed = 0.0
        self.pause = 0.0

    def reset(self, rng, duck_xy, duck_yaw):
        p = self.p
        if self.follow_duck:
            d = rng.uniform(*p.start)
            a = duck_yaw + math.radians(rng.uniform(-p.start_bearing_deg, p.start_bearing_deg))
        else:
            d = rng.uniform(2.0, self.half)
            a = rng.uniform(-math.pi, math.pi)
        self.x = float(np.clip(duck_xy[0] + d * math.cos(a), -self.half, self.half))
        self.y = float(np.clip(duck_xy[1] + d * math.sin(a), -self.half, self.half))
        self.yaw = math.atan2(duck_xy[1] - self.y, duck_xy[0] - self.x)  # start facing the duck
        self.pause = rng.uniform(0.5, 1.5)
        self._new_goal(rng, duck_xy)

    def _new_goal(self, rng, duck_xy):
        if self.follow_duck:
            d = rng.uniform(*self.p.near)
            a = rng.uniform(-math.pi, math.pi)
            g = (duck_xy[0] + d * math.cos(a), duck_xy[1] + d * math.sin(a))
        else:
            g = tuple(rng.uniform(-self.half, self.half, 2))
        self.goal = (float(np.clip(g[0], -self.half, self.half)), float(np.clip(g[1], -self.half, self.half)))
        self.speed = rng.uniform(*self.p.speed)

    def step(self, dt, rng, duck_xy):
        if self.manual:
            return self.pose
        if self.pause > 0:
            self.pause -= dt
            return self.pose
        dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            if rng.uniform() < self.p.pause_prob:
                self.pause = rng.uniform(*self.p.pause_s)
            self._new_goal(rng, duck_xy)
            return self.pose
        want = math.atan2(dy, dx)
        err = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += float(np.clip(err, -self.p.turn_rate * dt, self.p.turn_rate * dt))
        v = self.speed * max(0.0, math.cos(err))  # slow down while turning
        self.x += v * dt * math.cos(self.yaw)
        self.y += v * dt * math.sin(self.yaw)
        return self.pose

    @property
    def pose(self):
        return self.x, self.y, self.yaw

    def set_pose(self, x, y, yaw):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

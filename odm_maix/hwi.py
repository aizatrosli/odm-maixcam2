"""Hardware interface for the 14 STS3215 servos (port of rustypot_position_hwi)."""

from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np

from .config import JOINT_NAMES
from .feetech import FeetechBus, FeetechError

# Servo IDs, in policy order. Same as the Pi runtime.
JOINT_IDS = {
    "left_hip_yaw": 20,
    "left_hip_roll": 21,
    "left_hip_pitch": 22,
    "left_knee": 23,
    "left_ankle": 24,
    "neck_pitch": 30,
    "head_pitch": 31,
    "head_yaw": 32,
    "head_roll": 33,
    "right_hip_yaw": 10,
    "right_hip_roll": 11,
    "right_hip_pitch": 12,
    "right_knee": 13,
    "right_ankle": 14,
}

INIT_POS = {
    "left_hip_yaw": 0.002,
    "left_hip_roll": 0.053,
    "left_hip_pitch": -0.63,
    "left_knee": 1.368,
    "left_ankle": -0.784,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": -0.003,
    "right_hip_roll": -0.065,
    "right_hip_pitch": 0.635,
    "right_knee": 1.379,
    "right_ankle": -0.796,
}


class HWI:
    def __init__(self, config, bus: FeetechBus = None):
        m = config.maixcam2
        self.joints = dict(JOINT_IDS)
        self.ids = [self.joints[j] for j in JOINT_NAMES]
        self.init_pos = dict(INIT_POS)
        self.offsets = np.array([config.joints_offsets.get(j, 0.0) for j in JOINT_NAMES])
        self.kps = [32] * len(self.ids)
        self.kds = [0] * len(self.ids)
        self.low_torque_kps = [2] * len(self.ids)
        # Optional soft limits from scripts/calibrate.py ranges: {joint: [min, max]} in
        # radians, policy frame (before offsets). Targets are clamped to them.
        limits = m.get("joint_limits") or {}
        self.lower = np.array([limits.get(j, [-np.inf, np.inf])[0] for j in JOINT_NAMES], dtype=float)
        self.upper = np.array([limits.get(j, [-np.inf, np.inf])[1] for j in JOINT_NAMES], dtype=float)
        self.io = bus or FeetechBus(m["servo_port"], m["servo_baudrate"])

    def set_kps(self, kps: Sequence[float]) -> None:
        self.kps = list(kps)
        self.io.set_kps(self.ids, self.kps)

    def set_kds(self, kds: Sequence[float]) -> None:
        self.kds = list(kds)
        self.io.set_kds(self.ids, self.kds)

    def turn_on(self) -> None:
        self.io.set_kps(self.ids, self.low_torque_kps)
        self.io.enable_torque(self.ids)
        print("turn on : low KPS set")
        time.sleep(1)
        self.set_position_all(self.init_pos)
        print("turn on : init pos set")
        time.sleep(1)
        self.io.set_kps(self.ids, self.kps)
        print("turn on : high kps")

    def turn_off(self) -> None:
        self.io.disable_torque(self.ids)

    def set_position_all(self, joints_positions: Dict[str, float]) -> None:
        """Radians, keyed by joint name. Joint offsets are added here."""
        ids, pos = [], []
        for i, j in enumerate(JOINT_NAMES):
            if j in joints_positions:
                ids.append(self.ids[i])
                p = min(max(joints_positions[j], self.lower[i]), self.upper[i])
                pos.append(p + self.offsets[i])
        self.io.write_goal_position(ids, pos)

    def set_position_array(self, positions: np.ndarray) -> None:
        """Radians, one value per joint in JOINT_NAMES order."""
        p = np.clip(np.asarray(positions, dtype=float), self.lower, self.upper)
        self.io.write_goal_position(self.ids, list(p + self.offsets))

    def get_present_positions_velocities(self):
        """(positions rad, velocities rad/s) in JOINT_NAMES order, or (None, None)."""
        try:
            pos, vel = self.io.read_present_position_velocity(self.ids)
        except FeetechError as e:
            print("[HWI]", e)
            return None, None
        pos = np.around(np.array(pos) - self.offsets, 3)
        vel = np.around(np.array(vel), 3)
        return pos, vel

    def get_present_positions(self):
        return self.get_present_positions_velocities()[0]

    def get_present_velocities(self):
        return self.get_present_positions_velocities()[1]

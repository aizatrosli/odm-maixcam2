"""Robot configuration.

Reads the same ``duck_config.json`` as the Raspberry Pi runtime (joint
offsets, IMU orientation, expression features) plus a ``maixcam2`` section
for the board-specific wiring. Unknown keys are ignored, so an existing Pi
config file works unchanged; the defaults below fill in the rest.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

HOME_DIR = os.path.expanduser("~")
DEFAULT_CONFIG_PATH = os.path.join(HOME_DIR, "duck_config.json")

JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

DEFAULT_MAIXCAM2 = {
    # Servo bus: the Waveshare bus-servo adapter over USB (needs the
    # MaixCAM2 Type-C port in host mode) or a MaixCAM2 UART such as
    # /dev/ttyS1 wired to the adapter's UART header.
    "servo_port": "/dev/ttyACM0",
    "servo_baudrate": 1000000,
    # BNO055 on I2C. null bus = scan /dev/i2c-0..7 at start-up.
    "imu_i2c_bus": None,
    "imu_address": 0x28,
    "imu_offsets_path": os.path.join(HOME_DIR, "imu_offsets.json"),
    # Foot switches: MaixPy GPIO names (see the MaixCAM2 pinout). The
    # placeholders below MUST be checked against your wiring.
    "left_foot_gpio": "GPIOA4",
    "right_foot_gpio": "GPIOA5",
    # Extra pin functions applied with maix.pinmap at start-up,
    # e.g. {"A21": "UART4_TX", "A22": "UART4_RX", "A1": "I2C5_SCL"}.
    "pinmap": {},
    # Walking policy
    "policy_backend": "onnx",  # "onnx" (CPU) or "npu" (axmodel through maix.nn)
    "policy_path": os.path.join(HOME_DIR, "BEST_WALK_ONNX_2.onnx"),
    # Gait phase length. Either the polynomial_coefficients.pkl from the
    # Pi runtime, or set nb_steps_in_period directly.
    "polynomial_coefficients_path": os.path.join(HOME_DIR, "polynomial_coefficients.pkl"),
    "nb_steps_in_period": None,
}


@dataclass
class DuckConfig:
    start_paused: bool = False
    imu_upside_down: bool = False
    phase_frequency_factor_offset: float = 0.0
    expression_features: Dict[str, bool] = field(default_factory=dict)
    joints_offsets: Dict[str, float] = field(default_factory=lambda: {j: 0.0 for j in JOINT_NAMES})
    maixcam2: Dict = field(default_factory=lambda: dict(DEFAULT_MAIXCAM2))
    path: Optional[str] = None
    is_default: bool = True

    @classmethod
    def load(cls, path: Optional[str] = DEFAULT_CONFIG_PATH) -> "DuckConfig":
        data = {}
        is_default = True
        if path and os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            is_default = False

        offsets = {j: 0.0 for j in JOINT_NAMES}
        offsets.update(data.get("joints_offsets", {}))
        maix = dict(DEFAULT_MAIXCAM2)
        maix.update(data.get("maixcam2", {}))
        for k in ("imu_offsets_path", "policy_path", "polynomial_coefficients_path"):
            if maix.get(k):
                maix[k] = os.path.expanduser(maix[k])

        return cls(
            start_paused=data.get("start_paused", False),
            imu_upside_down=data.get("imu_upside_down", False),
            phase_frequency_factor_offset=data.get("phase_frequency_factor_offset", 0.0),
            expression_features=data.get("expression_features", {}),
            joints_offsets=offsets,
            maixcam2=maix,
            path=path,
            is_default=is_default,
        )

    def feature(self, name: str) -> bool:
        return bool(self.expression_features.get(name, False))

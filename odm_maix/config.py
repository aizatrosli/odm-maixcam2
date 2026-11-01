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
    # Pin choices are explained in docs/WIRING.md.
    # Servo bus: the Waveshare bus-servo adapter over USB (needs the
    # MaixCAM2 Type-C port in host mode), or UART4 on the 6-pin connector
    # (/dev/ttyS4, add "A21": "UART4_TX", "A22": "UART4_RX" to pinmap).
    "servo_port": "/dev/ttyACM0",
    "servo_baudrate": 1000000,
    # BNO055 on I2C7 (A8 = SCL, A9 = SDA, see pinmap below).
    # null bus = scan /dev/i2c-0..7 at start-up.
    "imu_i2c_bus": None,
    "imu_address": 0x28,
    "imu_offsets_path": os.path.join(HOME_DIR, "imu_offsets.json"),
    # Foot switches to GND, on the left 2x6 header, each with an external
    # 10k pull-up to 3V3 (maix.gpio does not turn on a pull-up for inputs).
    "left_foot_gpio": "GPIOB21",
    "right_foot_gpio": "GPIOB20",
    # Pin functions applied with maix.pinmap at start-up. A pinmap in
    # duck_config.json replaces this one, so keep the I2C7 entries in it.
    "pinmap": {"A8": "I2C7_SCL", "A9": "I2C7_SDA"},
    # Walking policy
    "policy_backend": "onnx",  # "onnx" (CPU) or "npu" (axmodel through maix.nn)
    "policy_path": os.path.join(HOME_DIR, "BEST_WALK_ONNX_2.onnx"),
    # Gait phase length. Either the polynomial_coefficients.pkl from the
    # Pi runtime, or set nb_steps_in_period directly.
    "polynomial_coefficients_path": os.path.join(HOME_DIR, "polynomial_coefficients.pkl"),
    "nb_steps_in_period": None,
    # Step 2: camera follow mode (scripts/walk.py --follow)
    "vision": {
        "model_path": "/root/models/yolo11n.mud",
        "class_name": "person",
        "conf_threshold": 0.5,
        "show": False,  # draw boxes on the MaixCAM2 screen
        # FollowParams overrides, see odm_maix/vision/follow.py
        "follow": {},
    },
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
        user_maix = data.get("maixcam2", {})
        maix.update(user_maix)
        maix["vision"] = {**DEFAULT_MAIXCAM2["vision"], **user_maix.get("vision", {})}
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

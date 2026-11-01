"""Policy profile: a JSON file saved next to each exported ONNX.

The MaixCAM2 runtime reads it (odm_maix/policy_profile.py) so that the robot
runs the policy exactly the way it was trained. No JAX imports here.
"""

import json

STOCK_PROFILE = {
    # How the stock BEST_WALK_ONNX_2.onnx was trained / is run on the Pi
    "name": "stock",
    "obs_layout": "v2_101",
    "extra_obs": [],  # names of extra inputs appended after the 101 base values
    "action_scale": 0.25,
    "apply_head_commands": True,  # the runtime adds commands[3:7] to the head targets
    "motor_speed_limit": None,  # rad/s clip on target changes; None = off (Pi runtime)
    "head_in_motor_targets_obs": True,  # obs motor_targets include the head offsets
}

FOLLOW_PROFILE = {
    "name": "follow_v1",
    "obs_layout": "v2_101",
    "extra_obs": [],
    "action_scale": 0.25,
    "apply_head_commands": True,
    "motor_speed_limit": 5.24,
    "head_in_motor_targets_obs": True,
}


def save_profile(onnx_path: str, profile: dict) -> str:
    path = onnx_path[: -len(".onnx")] + ".json" if onnx_path.endswith(".onnx") else onnx_path + ".json"
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    return path

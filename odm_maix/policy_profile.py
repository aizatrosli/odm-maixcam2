"""Policy profiles: how a walking policy must be run on the robot.

A profile is a JSON file next to the policy (``walk.onnx`` -> ``walk.json``,
``walk_policy.mud`` -> ``walk_policy.json``). Training (training/odm_train)
writes one for every exported checkpoint. A policy without one is run the
way the Pi runtime ran the stock BEST_WALK_ONNX_2.onnx.

Keys:
    name                 label printed at start-up
    action_scale         targets = init_pos + action * action_scale
    apply_head_commands  add commands[3:7] to the head joint targets
    motor_speed_limit    rad/s clip on how fast policy targets may change (null = off)
    extra_obs            names of inputs appended after the 101 base values, e.g.
                         ["target_bearing_sin", "target_bearing_cos", "target_size"]
                         or a camera embedding ["emb_0", ..., "emb_15"]. The runtime
                         asks the command source for them (``extra_obs()``).
"""

from __future__ import annotations

import json
import os

BASE_OBS_SIZE = 101

STOCK_PROFILE = {
    "name": "stock",
    "action_scale": 0.25,
    "apply_head_commands": True,
    "motor_speed_limit": None,
    "extra_obs": [],
}


def profile_path_for(policy_path: str) -> str:
    return os.path.splitext(policy_path)[0] + ".json"


def load_profile(policy_path: str) -> dict:
    profile = dict(STOCK_PROFILE)
    path = profile_path_for(policy_path)
    if os.path.exists(path):
        with open(path) as f:
            profile.update(json.load(f))
    return profile


def obs_size(profile: dict) -> int:
    return BASE_OBS_SIZE + len(profile.get("extra_obs", []))

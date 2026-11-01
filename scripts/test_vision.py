"""Run the camera follow logic without moving the robot.

Prints the target, the follow state and the walking commands it would send.
Use --show to draw the boxes on the MaixCAM2 screen.
"""

import argparse
import time

import numpy as np
from _common import DEFAULT_CONFIG_PATH, load_config

from odm_maix.commands import FollowCommands

p = argparse.ArgumentParser()
p.add_argument("--duck_config_path", default=DEFAULT_CONFIG_PATH)
p.add_argument("--show", action="store_true")
args = p.parse_args()

config = load_config(args.duck_config_path)
vision = dict(config.maixcam2["vision"])
vision["show"] = args.show or vision.get("show", False)
follow = FollowCommands(vision)
try:
    while True:
        cmd, _ = follow.get()
        t = follow.controller.target
        where = f"person cx {t.cx:.2f} h {t.h:.2f}" if t else "no person"
        print(
            f"{follow.state:10s} {where:24s} vx {cmd[0]:+.3f} wz {cmd[2]:+.3f} "
            f"head_yaw {cmd[5]:+.2f} head_pitch {cmd[4]:+.2f}  det {follow.detector.fps():.1f} fps"
        )
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    follow.stop()

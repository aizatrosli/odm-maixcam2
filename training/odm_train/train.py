"""Train the follow-robust walking policy.

Run from the Open_Duck_Playground checkout (it provides the robot model,
reference motion and PPO runner), with this repo's training/ on PYTHONPATH:

    cd Open_Duck_Playground
    PYTHONPATH=$PYTHONPATH:/path/to/odm-maixcam2/training \\
      uv run python -m odm_train.train --num_timesteps 300000000 --output_dir runs/follow_v1

Every checkpoint is exported as <step>.onnx with a <step>.json policy profile
next to it. Copy both to the MaixCAM2.
"""

import argparse
import glob
import os

from playground.common.runner import BaseRunner

from odm_train import follow_env
from odm_train.profile import FOLLOW_PROFILE, save_profile
from odm_train.randomize import domain_randomize_strong


class FollowRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        self.env_config = follow_env.default_config()
        self.env = follow_env.FollowJoystick(task=args.task, config=self.env_config)
        self.eval_env = follow_env.FollowJoystick(task=args.task, config=self.env_config)
        self.randomizer = domain_randomize_strong if args.randomization == "strong" else None
        if args.randomization == "stock":
            from playground.common import randomize

            self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size} (must be 101 for the MaixCAM2 runtime)")

    def policy_params_fn(self, current_step, make_policy, params):
        super().policy_params_fn(current_step, make_policy, params)
        # write the profile next to the ONNX the base runner just exported
        newest = max(glob.glob(f"{self.output_dir}/*_{current_step}.onnx"), key=os.path.getmtime)
        save_profile(newest, FOLLOW_PROFILE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="runs/follow_v1")
    p.add_argument("--num_timesteps", type=int, default=300_000_000)
    p.add_argument("--task", default="flat_terrain_backlash", help="or rough_terrain_backlash")
    p.add_argument("--randomization", choices=["strong", "stock"], default="strong")
    p.add_argument("--restore_checkpoint_path", default=None)
    args = p.parse_args()
    FollowRunner(args).train()


if __name__ == "__main__":
    main()

"""Train an Open Duck Mini v2 walking policy for the MaixCAM2.

Run from the Open_Duck_Playground checkout (it provides the robot model,
reference motion and ONNX export), with this repo's training/ on PYTHONPATH:

    cd Open_Duck_Playground
    PYTHONPATH=$PYTHONPATH:/path/to/odm-maixcam2/training \\
      uv run python -m odm_train.train --flavor follow_flat_backlash --output_dir runs/follow_v1

Flavors are listed in odm_train/flavors.py (``--list_flavors``). Every
checkpoint is exported as <date>_<step>.onnx with a <date>_<step>.json policy
profile next to it. Copy both to the MaixCAM2.

``--resume`` continues from the newest checkpoint in --output_dir, so a run
that was cut off (for example a Colab disconnect) picks up where it stopped.
Use a smaller ``--num_envs`` on GPUs with little memory (4 GB: try 2048).
"""

import argparse
import functools
import glob
import json
import os
import re

from odm_train.flavors import FLAVORS
from odm_train.profile import FOLLOW_PROFILE, STANDING_PROFILE, STOCK_RETRAIN_PROFILE, save_profile

PROGRESS_FILE = "progress.json"


def latest_checkpoint(output_dir):
    """Newest orbax checkpoint folder (<date>_<step>) in output_dir, or None."""
    dirs = [d for d in glob.glob(os.path.join(output_dir, "*_*")) if os.path.isdir(d) and re.search(r"_\d+$", d)]
    return max(dirs, key=os.path.getmtime) if dirs else None


def read_progress(output_dir):
    try:
        with open(os.path.join(output_dir, PROGRESS_FILE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"done_steps": 0}


def make_env(flavor, env_name, task):
    if env_name == "follow":
        from odm_train import follow_env

        cfg = follow_env.default_config()
        return cfg, follow_env.FollowJoystick(task=task, config=cfg), follow_env.FollowJoystick(task=task, config=cfg)
    if env_name == "joystick":
        from playground.open_duck_mini_v2 import joystick

        cfg = joystick.default_config()
        return cfg, joystick.Joystick(task=task, config=cfg), joystick.Joystick(task=task, config=cfg)
    if env_name == "standing":
        from playground.open_duck_mini_v2 import standing

        cfg = standing.default_config()
        return cfg, standing.Standing(task=task, config=cfg), standing.Standing(task=task, config=cfg)
    raise ValueError(f"unknown env {env_name!r} in flavor {flavor!r}")


PROFILES = {"follow": FOLLOW_PROFILE, "joystick": STOCK_RETRAIN_PROFILE, "standing": STANDING_PROFILE}


def build_runner(args):
    # JAX-heavy imports live here so --list_flavors works without them.
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper
    from mujoco_playground.config import locomotion_params
    from playground.common.runner import BaseRunner

    flavor = FLAVORS[args.flavor]
    env_name = args.env or flavor["env"]
    task = args.task or flavor["task"]
    randomization = args.randomization or flavor["randomization"]

    class Runner(BaseRunner):
        def __init__(self):
            super().__init__(args)
            if args.jax_cache:
                import jax

                os.makedirs(args.jax_cache, exist_ok=True)
                jax.config.update("jax_compilation_cache_dir", args.jax_cache)
            self.env_config, self.env, self.eval_env = make_env(args.flavor, env_name, task)
            if randomization == "strong":
                from odm_train.randomize import domain_randomize_strong

                self.randomizer = domain_randomize_strong
            elif randomization == "stock":
                from playground.common import randomize

                self.randomizer = randomize.domain_randomize
            else:
                self.randomizer = None
            self.action_size = self.env.action_size
            self.obs_size = int(self.env.observation_size["state"][0])
            self.profile = dict(PROFILES[env_name], flavor=args.flavor, task=task)
            self.done_before = 0
            self.restore_checkpoint_path = args.restore_checkpoint_path
            if args.resume and not self.restore_checkpoint_path:
                self.restore_checkpoint_path = latest_checkpoint(str(self.output_dir))
                if self.restore_checkpoint_path:
                    self.done_before = read_progress(str(self.output_dir))["done_steps"]
            print(f"Flavor {args.flavor}: env={env_name} task={task} randomization={randomization}")
            print(f"Observation size: {self.obs_size} (the walking runtime expects 101)")
            if self.restore_checkpoint_path:
                print(f"Resuming from {self.restore_checkpoint_path} ({self.done_before} steps already done)")

        def policy_params_fn(self, current_step, make_policy, params):
            step = self.done_before + current_step
            super().policy_params_fn(step, make_policy, params)
            # write the profile next to the ONNX the base runner just exported
            newest = max(glob.glob(f"{self.output_dir}/*_{step}.onnx"), key=os.path.getmtime)
            save_profile(newest, self.profile)
            with open(os.path.join(self.output_dir, PROGRESS_FILE), "w") as f:
                json.dump({"done_steps": step, "flavor": args.flavor, "latest_onnx": os.path.basename(newest)}, f)

        def train(self):
            # Same as BaseRunner.train, plus --num_envs and resume step accounting.
            self.ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
            p = dict(self.ppo_params)
            if args.num_envs:
                p["num_envs"] = args.num_envs
                # brax needs batch_size * num_minibatches to be a multiple of num_envs
                while (p["batch_size"] * p["num_minibatches"]) % p["num_envs"]:
                    p["num_minibatches"] *= 2
            if args.num_evals:
                p["num_evals"] = args.num_evals
            remaining = args.num_timesteps - self.done_before
            if remaining <= 0:
                print(f"Already trained {self.done_before} >= {args.num_timesteps} steps, nothing to do.")
                return
            p["num_timesteps"] = remaining
            factory = ppo_networks.make_ppo_networks
            if "network_factory" in p:
                factory = functools.partial(ppo_networks.make_ppo_networks, **p.pop("network_factory"))
            print(f"PPO params: {p}")
            ppo.train(
                **p,
                network_factory=factory,
                randomization_fn=self.randomizer,
                progress_fn=self.progress_callback,
                policy_params_fn=self.policy_params_fn,
                restore_checkpoint_path=self.restore_checkpoint_path,
                environment=self.env,
                eval_env=self.eval_env,
                wrap_env_fn=wrapper.wrap_for_brax_training,
            )

    return Runner()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--flavor", default="follow_flat_backlash", choices=sorted(FLAVORS))
    p.add_argument("--list_flavors", action="store_true")
    p.add_argument("--output_dir", default="runs/follow_v1")
    p.add_argument("--num_timesteps", type=int, default=300_000_000, help="total, including resumed steps")
    p.add_argument("--num_envs", type=int, default=None, help="parallel envs (default 8192; 4 GB GPUs: 2048)")
    p.add_argument("--num_evals", type=int, default=None, help="checkpoints/evals per run (default 15)")
    p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --output_dir")
    p.add_argument("--restore_checkpoint_path", default=None)
    p.add_argument("--jax_cache", default=None, help="JAX compile cache dir (put it on Drive to skip recompiles)")
    # overrides of the flavor's settings
    p.add_argument("--env", choices=["follow", "joystick", "standing"], default=None)
    p.add_argument("--task", default=None, help="flat_terrain, flat_terrain_backlash or rough_terrain_backlash")
    p.add_argument("--randomization", choices=["strong", "stock", "none"], default=None)
    args = p.parse_args(argv)
    if args.list_flavors:
        for name, f in FLAVORS.items():
            print(f"{name:24s} env={f['env']:9s} task={f['task']:24s} randomization={f['randomization']}")
        return
    build_runner(args).train()


if __name__ == "__main__":
    main()

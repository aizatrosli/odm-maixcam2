"""Run the RL walking policy on the MaixCAM2.

    python3 scripts/walk.py                       # CPU (onnxruntime), gamepad if found
    python3 scripts/walk.py --backend npu --policy ~/models/walk_policy.mud
    python3 scripts/walk.py --save_obs obs.npy    # record observations for NPU calibration
    python3 scripts/walk.py --follow              # follow a person with the camera
    python3 scripts/walk.py --interact            # follow + gestures, faces and moods
"""

import argparse

from _common import DEFAULT_CONFIG_PATH, load_config, make_interact_commands

from odm_maix.commands import FollowCommands, GamepadCommands, ZeroCommands
from odm_maix.feet_contacts import FeetContacts, NoFeetContacts
from odm_maix.hwi import HWI
from odm_maix.imu import Imu
from odm_maix.policy import load_policy
from odm_maix.policy_profile import load_profile, obs_size
from odm_maix.rl_utils import nb_steps_from_config
from odm_maix.telemetry import TelemetryWriter
from odm_maix.walk import RLWalk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duck_config_path", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--backend", choices=["onnx", "npu"], help="overrides maixcam2.policy_backend")
    p.add_argument("--policy", help="overrides maixcam2.policy_path (.onnx or .mud)")
    p.add_argument("-a", "--action_scale", type=float, default=0.25)
    p.add_argument("-p", type=int, default=30)
    p.add_argument("-i", type=int, default=0)
    p.add_argument("-d", type=int, default=0)
    p.add_argument("-c", "--control_freq", type=int, default=50)
    p.add_argument("--cutoff_frequency", type=float, default=None)
    p.add_argument("--no_gamepad", action="store_true", help="walk in place with zero commands")
    p.add_argument("--follow", action="store_true", help="follow a person seen by the camera (NPU detector)")
    p.add_argument(
        "--interact",
        action="store_true",
        help="follow mode plus hand gestures, faces and moods (wave = come, open palm = stay)",
    )
    p.add_argument("--no_feet", action="store_true", help="foot switches not wired: report both feet down")
    p.add_argument("--save_obs", default=None, help="save observations to this .npy file")
    p.add_argument("--no_telemetry", action="store_true", help="don't publish the feed for the screen dashboard")
    args = p.parse_args()

    config = load_config(args.duck_config_path)
    m = config.maixcam2
    backend = args.backend or m["policy_backend"]
    policy_path = args.policy or m["policy_path"]
    profile = load_profile(policy_path)
    print(f"policy: {backend} {policy_path}  profile: {profile['name']}")
    policy = load_policy(backend, policy_path, obs_size=obs_size(profile))
    if getattr(policy, "obs_size", obs_size(profile)) != obs_size(profile):
        raise SystemExit(f"policy expects {policy.obs_size} inputs but its profile describes {obs_size(profile)}")

    mode = "interact" if args.interact else "follow" if args.follow else "walk"
    vision_on_screen = (args.interact or args.follow) and (config.feature("screen_eyes") or m["vision"].get("show"))
    telemetry = None if args.no_telemetry else TelemetryWriter(mode=mode, screen=bool(vision_on_screen))
    if telemetry is not None:
        telemetry.announce()  # a running dashboard releases the camera and the hardware

    gamepad = None
    if not args.no_gamepad:
        try:
            gamepad = GamepadCommands()
        except Exception as e:
            print(f"gamepad unavailable ({e})")

    if args.interact:
        commands = make_interact_commands(config, gamepad)
    elif args.follow:
        commands = FollowCommands(m["vision"], gamepad=gamepad)
    elif gamepad is not None:
        commands = gamepad
    else:
        print("no command source, walking in place")
        commands = ZeroCommands()

    hwi = HWI(config)
    walker = RLWalk(
        config,
        hwi,
        imu=None,
        feet_contacts=None,
        policy=policy,
        commands=commands,
        nb_steps_in_period=nb_steps_from_config(config),
        control_freq=args.control_freq,
        pid=(args.p, args.i, args.d),
        action_scale=args.action_scale,
        cutoff_frequency=args.cutoff_frequency,
        save_obs_path=args.save_obs,
        profile=profile,
    )
    walker.telemetry = telemetry
    walker.backend = backend
    walker.start()
    walker.imu = Imu(config, sampling_freq=args.control_freq)
    walker.feet_contacts = NoFeetContacts() if args.no_feet else FeetContacts(config)
    walker.run()


if __name__ == "__main__":
    main()

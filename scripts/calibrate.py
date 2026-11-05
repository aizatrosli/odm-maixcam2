"""Calibrate the duck before its first walk. Run on a stand, one step at a time.

    python3 scripts/calibrate.py servo-id --id 22       # new servo alone on the bus -> ID 22, stock settings
    python3 scripts/calibrate.py servo-id --id 22 --center   # ... and drive it to 0 rad to fit the horn centred
    python3 scripts/calibrate.py offsets                # hold the duck in its init pose, offsets from that
    python3 scripts/calibrate.py offsets --nudge        # then fine-tune joint by joint with low torque
    python3 scripts/calibrate.py ranges                 # move each joint by hand, find its range and the encoder wrap
    python3 scripts/calibrate.py ranges --save-limits   # ... and save soft limits the runtime clamps targets to
    python3 scripts/calibrate.py imu                    # BNO055 calibration, then upright/axis checks

Every step works with --dry-run against the simulated duck. Results are
written to duck_config.json (a timestamped .bak copy is kept) only after you
confirm, or with --yes.
"""

import argparse
import json
import os
import sys

from _common import DEFAULT_CONFIG_PATH, load_config

from odm_maix.config import JOINT_NAMES
from odm_maix.diag import calib
from odm_maix.diag.checks import check_imu_axes, check_imu_still, check_imu_upright
from odm_maix.diag.core import FAIL, Context, Skip, print_result
from odm_maix.hwi import INIT_POS, JOINT_IDS


def ask_yes(ctx, args, question):
    if args.yes:
        return True
    return ctx.confirm(question, default=False)


def cmd_offsets(ctx, args):
    bus = ctx.bus()
    joints = args.joints or JOINT_NAMES
    old = ctx.config.joints_offsets
    if not args.nudge_only:
        ctx.prompt("pose.init", "Torque is off. Hold the duck in its init pose: the standing pose of the Open Duck Mini "
                   "assembly guide (legs bent, head level and centred), on a stand")
        new = calib.capture_offsets(bus, joints)
        for line in calib.offsets_table(new, old):
            print("   ", line)
        if ctx.dry_run:
            err = max(abs(new[j] - ctx.sim.true_offsets[j]) for j in joints)
            print(f"    [dry run] largest error vs the simulated horn offsets: {err:.4f} rad")
    else:
        new = {j: float(old.get(j, 0.0)) for j in joints}
    if args.nudge:
        print("  Fine-tune: each joint is held at init + offset with low torque; adjust until it looks right.")
        if ctx.dry_run:
            answers = iter(["++", "-", ""] * len(joints))

            def ask(prompt):
                a = next(answers)
                print(prompt + a)
                return a
        else:
            ask = input
        for j in joints:
            try:
                v = calib.nudge_joint(bus, j, new[j], ask)
            except KeyboardInterrupt:
                print("    stopped")
                break
            if v is not None:
                new[j] = v
    if ask_yes(ctx, args, f"Save these offsets to {ctx.config.path}?"):
        bak = calib.save_offsets(ctx.config.path, new)
        print(f"  saved{' (backup ' + bak + ')' if bak else ''}. Next: python3 scripts/calibrate.py ranges")
    return 0


def cmd_ranges(ctx, args):
    bus = ctx.bus()
    joints = args.joints or JOINT_NAMES
    results, bad = {}, 0
    for j in joints:
        ctx.prompt(f"range.{j}", f"Move {j} (id {JOINT_IDS[j]}) slowly through its FULL range, end stop to end stop, "
                   f"for {args.seconds:.0f} s after Enter")
        lo, hi, crossed, n = calib.record_range(bus, j, args.seconds)
        r = calib.analyse_range(j, lo, hi, crossed, ctx.config.joints_offsets.get(j, 0.0))
        results[j] = r
        flag = "ok" if not r["problems"] else "; ".join(r["problems"])
        bad += bool(r["problems"])
        print(f"    {j:16s} {r['min']:+.2f} .. {r['max']:+.2f} rad ({r['span_deg']:.0f} deg), init {INIT_POS[j]:+.2f}, "
              f"wrap margin {r['wrap_margin_rad']:.2f}  {flag}")
    path = os.path.expanduser(args.out)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ranges written to {path}; {bad} joint(s) with problems")
    if args.save_limits:
        limits = calib.soft_limits(results)
        if limits and ask_yes(ctx, args, f"Save soft limits for {len(limits)} joints to {ctx.config.path}?"):
            bak = calib.save_joint_limits(ctx.config.path, limits)
            print(f"  saved maixcam2.joint_limits{' (backup ' + bak + ')' if bak else ''}")
    return 1 if bad else 0


def cmd_imu(ctx, args):
    dev = ctx.imu_device()
    path = ctx.config.maixcam2["imu_offsets_path"]
    if not args.check_only:
        print("  Move the duck slowly through several orientations (each face down for a few seconds, then a figure 8)")
        print("  until every value reaches 3. Gyro settles when it is still, accel needs ~6 orientations, mag the figure 8.")
        poll = (lambda: ctx.sim.act("imu.calibrate_motion")) if ctx.dry_run else None
        calib.calibrate_imu(dev, path, timeout=args.timeout, on_poll=poll, poll=0.05 if ctx.dry_run else 0.5)
        print(f"  saved BNO055 offsets to {path}")
    status = 0
    for fn in (check_imu_still, check_imu_upright) + ((check_imu_axes,) if ctx.interactive else ()):
        r = fn(ctx)
        r.name = fn.__name__.replace("check_", "").replace("_", ".")
        print_result(r)
        status |= r.status == FAIL
    if not ctx.interactive:
        print("  run with --interactive to also check the tilt and turn signs")
    return 2 if status else 0


def cmd_servo_id(ctx, args):
    bus = ctx.bus()
    if args.id not in JOINT_IDS.values() and not args.any_id:
        name_ids = ", ".join(f"{j}={i}" for j, i in JOINT_IDS.items())
        raise SystemExit(f"--id {args.id} is not a joint ID ({name_ids}); add --any-id to force")
    if ctx.dry_run and not ctx.sim.faults.get("stray"):
        # one fresh servo alone on the bus, as on the bench
        from odm_maix.diag.simhw import VirtualServo

        ctx.sim.serial.servos[:] = [VirtualServo(1, 1000)]
    current = args.current
    if current is None:
        print("  scanning IDs 0..253 (only the new servo should be connected) ...")
        found = calib.find_single_servo(bus)
        if len(found) != 1:
            raise SystemExit(f"found servos {found}: connect exactly one servo, or pass --current")
        current = found[0]
    joint = next((j for j, i in JOINT_IDS.items() if i == args.id), "?")
    if not ask_yes(ctx, args, f"Set servo {current} -> ID {args.id} ({joint}), position mode, no return delay?"):
        return 1
    got = calib.configure_servo(bus, current, args.id)
    print(f"  read back: {got}")
    if (got["min_angle"], got["max_angle"]) not in ((0, 0), (0, 4095)):
        print("  note: this servo has angle limits set; servo.eeprom in diagnose.py will report them")
    if args.center:
        if ask_yes(ctx, args, "Drive it to its centre (0 rad, raw 2048) so the horn can be fitted? It will move."):
            raw = calib.center_servo(bus, args.id)
            print(f"  at raw {raw}. Fit the horn now in the position the assembly guide shows for 0 rad.")
            if not ctx.dry_run:
                input("  Enter to release torque ")
            bus.disable_torque([args.id])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duck_config_path", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--dry-run", action="store_true", help="simulated duck")
    p.add_argument("--fault", action="append", default=[], help="dry run fault, see diagnose.py --list-faults")
    p.add_argument("-y", "--yes", action="store_true", help="save without asking")
    p.add_argument("-i", "--interactive", action="store_true", help="guided checks after calibrating (imu)")
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offsets", help="joint offsets from the init pose")
    o.add_argument("--joints", nargs="*", choices=JOINT_NAMES)
    o.add_argument("--nudge", action="store_true", help="fine-tune each joint with low torque after capturing")
    o.add_argument("--nudge-only", action="store_true", help="skip the capture, fine-tune the current offsets")
    r = sub.add_parser("ranges", help="measure joint ranges by hand")
    r.add_argument("--joints", nargs="*", choices=JOINT_NAMES)
    r.add_argument("--seconds", type=float, default=4.0)
    r.add_argument("--out", default="~/joint_ranges.json")
    r.add_argument("--save-limits", action="store_true", help="write maixcam2.joint_limits (soft limits) to the config")
    m = sub.add_parser("imu", help="BNO055 calibration and orientation check")
    m.add_argument("--check-only", action="store_true")
    m.add_argument("--timeout", type=float, default=180)
    s = sub.add_parser("servo-id", help="set up a new servo")
    s.add_argument("--id", type=int, required=True, help="joint ID to give it (10-14, 20-24, 30-33)")
    s.add_argument("--current", type=int, help="its ID now (default: scan; factory servos are 1)")
    s.add_argument("--center", action="store_true", help="then drive it to 0 rad to fit the horn")
    s.add_argument("--any-id", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "offsets" and args.nudge_only:
        args.nudge = True

    if args.dry_run and args.duck_config_path == DEFAULT_CONFIG_PATH:
        # never touch the real config in a dry run
        args.duck_config_path = os.path.join("/tmp", "duck_config.dryrun.json")
        print(f"dry run: using {args.duck_config_path}")
    config = load_config(args.duck_config_path, apply_pinmap=not args.dry_run)
    config.path = args.duck_config_path
    if args.dry_run:
        config.maixcam2["imu_offsets_path"] = os.path.join("/tmp", "imu_offsets.dryrun.json")
    ctx = Context(config, dry_run=args.dry_run, interactive=args.interactive, faults=args.fault)
    try:
        return {"offsets": cmd_offsets, "ranges": cmd_ranges, "imu": cmd_imu, "servo-id": cmd_servo_id}[args.cmd](ctx, args)
    except Skip as e:
        print(f"cannot run: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nstopped, torque off")
        return 130
    finally:
        ctx.close()


if __name__ == "__main__":
    sys.exit(main())

"""Pre-flight diagnostics for the duck. No check turns servo torque on.

    python3 scripts/diagnose.py                      # everything that needs no help
    python3 scripts/diagnose.py --interactive        # plus guided steps (tilt the duck, press feet, listen)
    python3 scripts/diagnose.py --only servo,imu     # some groups or checks
    python3 scripts/diagnose.py --dry-run            # on a PC: a simulated duck, wire-level
    python3 scripts/diagnose.py --dry-run --fault missing:23,upside_down    # see what a fault looks like

Groups: config, board, servo, imu, feet, camera, npu, audio, expression, loop.
Writes a JSON report with every number to diag_report.json (--json).
Exit code: 0 all passed, 1 warnings, 2 failures.
"""

import argparse
import sys

from _common import DEFAULT_CONFIG_PATH, load_config

from odm_maix.diag.checks import CHECKS
from odm_maix.diag.core import FAIL, WARN, Context, run_checks, summary, write_report
from odm_maix.diag.simhw import __doc__ as FAULTS_DOC


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duck_config_path", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--dry-run", action="store_true", help="no robot: run against a simulated duck")
    p.add_argument("--fault", action="append", default=[], help="dry run only: inject a fault (see --list-faults)")
    p.add_argument("--list-faults", action="store_true")
    p.add_argument("--list", action="store_true", help="list the checks and exit")
    p.add_argument("-i", "--interactive", action="store_true", help="include guided steps")
    p.add_argument("--only", default="", help="comma-separated groups or check names")
    p.add_argument("--skip", default="", help="comma-separated groups or check names to leave out")
    p.add_argument("--scan", action="store_true", help="scan all 254 servo IDs even if all 14 answer")
    p.add_argument("--loop-seconds", type=float, default=10)
    p.add_argument("--mud", help="NPU policy to test (default: policy_path when policy_backend is npu)")
    p.add_argument("--onnx", help="ONNX policy to compare the NPU against")
    p.add_argument("--obs", help="recorded observations (.npy) for the NPU comparison")
    p.add_argument("--json", default="diag_report.json", help="report path ('' for none)")
    args = p.parse_args(argv)

    if args.list_faults:
        print(FAULTS_DOC)
        return 0
    if args.list:
        for c in CHECKS:
            flags = ("guided " if c.interactive else "") + ("board " if c.board_only else "")
            print(f"{c.name:20s} {c.title}  {flags}")
        return 0
    if args.fault and not args.dry_run:
        p.error("--fault only works with --dry-run")

    config = load_config(args.duck_config_path, apply_pinmap=not args.dry_run)
    opts = {k: v for k, v in vars(args).items() if k in ("scan", "loop_seconds", "mud", "onnx", "obs")}
    ctx = Context(config, dry_run=args.dry_run, interactive=args.interactive, faults=args.fault, opts=opts)
    print(f"Duck diagnostics{' (DRY RUN, simulated duck)' if args.dry_run else ''}; servo torque stays off")
    split = lambda s: {x.strip() for x in s.split(",") if x.strip()}  # noqa: E731
    try:
        results = run_checks(ctx, CHECKS, only=split(args.only), skip=split(args.skip))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    finally:
        ctx.close()
    overall = summary(results)
    if args.json:
        write_report(args.json, results, ctx)
        print(f"report: {args.json}")
    return {FAIL: 2, WARN: 1}.get(overall, 0)


if __name__ == "__main__":
    sys.exit(main())

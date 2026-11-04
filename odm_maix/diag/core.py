"""Check runner for scripts/diagnose.py and scripts/calibrate.py.

A check is a function ``fn(ctx) -> Result | None``. It reports PASS, WARN,
FAIL or SKIP with a one-line summary, optional detail lines, a hint on what to
do next, and numbers for the JSON report. Hardware is opened through the
``Context``, which hands out either the real devices or the simulated duck
(``--dry-run``), so the checks never know which one they talk to.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_ORDER = {PASS: 0, SKIP: 0, WARN: 1, FAIL: 2}


@dataclass
class Result:
    status: str
    summary: str
    details: List[str] = field(default_factory=list)
    hint: str = ""
    data: Dict = field(default_factory=dict)
    name: str = ""
    group: str = ""
    seconds: float = 0.0


def worst(statuses) -> str:
    out = PASS
    for s in statuses:
        if _ORDER[s] > _ORDER[out]:
            out = s
    return out


class Skip(Exception):
    """Raise inside a check to report SKIP with this message."""


@dataclass
class Check:
    name: str
    group: str
    fn: Callable
    title: str
    interactive: bool = False  # only runs with --interactive
    board_only: bool = False  # needs the MaixCAM2 (camera, NPU, audio, /sys)


class Context:
    def __init__(self, config, dry_run=False, interactive=False, faults=None, input_fn=input, out=print, opts=None):
        self.config = config
        self.dry_run = dry_run
        self.interactive = interactive
        self.input_fn = input_fn
        self.out = out
        self.opts = dict(opts or {})
        self.sim = None
        if dry_run:
            from .simhw import SimDuck

            self.sim = SimDuck(faults, config)
        self._bus = None
        self._bus_error = None
        self._imu = None
        self._imu_error = None
        self._feet = None
        self.shared: Dict = {}  # measurements passed between checks

    # environment ------------------------------------------------------------
    @property
    def on_board(self) -> bool:
        if self.dry_run:
            return False
        try:
            import maix  # noqa: F401

            return True
        except Exception:
            return False

    # user interaction ---------------------------------------------------------
    def prompt(self, key: str, text: str) -> None:
        """Ask the person to do something, then wait for Enter. Dry run: the simulator does it."""
        if self.dry_run:
            self.out(f"    > {text}  [simulated]")
            self.sim.act(key)
            return
        self.input_fn(f"    > {text}  [Enter] ")

    def confirm(self, question: str, default: bool = False) -> bool:
        if self.dry_run:
            self.out(f"    ? {question} [simulated: yes]")
            return True
        ans = self.input_fn(f"    ? {question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        return default if not ans else ans.startswith("y")

    # hardware -------------------------------------------------------------------
    def bus(self):
        from ..feetech import FeetechBus

        if self._bus is None and self._bus_error is None:
            m = self.config.maixcam2
            try:
                if self.dry_run:
                    self._bus = FeetechBus(m["servo_port"], m["servo_baudrate"], ser=self.sim.serial)
                else:
                    self._bus = FeetechBus(m["servo_port"], m["servo_baudrate"])
            except Exception as e:
                self._bus_error = e
        if self._bus is None:
            raise Skip(f"servo bus not open ({self._bus_error})")
        return self._bus

    def imu_device(self):
        """An open bno055.BNO055, with the remap from the config applied."""
        from .. import bno055

        if self._imu is None and self._imu_error is None:
            m = self.config.maixcam2
            try:
                if self.dry_run:
                    if self.sim.imu_bus is None:
                        raise IOError("no BNO055 answered on /dev/i2c-0..7")
                    bus, addr = 7, m["imu_address"]
                    dev = bno055.BNO055(bus, addr, smbus=self.sim.imu_bus)
                else:
                    bus, addr = m["imu_i2c_bus"], m["imu_address"]
                    if bus is None:
                        found = bno055.scan_for_bno055()
                        if found is None:
                            raise IOError("no BNO055 answered on /dev/i2c-0..7")
                        bus, addr = found
                    dev = bno055.BNO055(bus, addr)
                dev.address = addr
                dev.set_axis_remap(
                    bno055.REMAP_UPSIDE_DOWN if self.config.imu_upside_down else bno055.REMAP_UPRIGHT
                )
                self._imu = dev
            except Exception as e:
                self._imu_error = e
        if self._imu is None:
            raise Skip(f"IMU not available ({self._imu_error})")
        return self._imu

    def feet(self):
        if self._feet is None:
            if self.dry_run:
                self._feet = self.sim.feet
            else:
                from ..feet_contacts import FeetContacts

                self._feet = FeetContacts(self.config)
        return self._feet

    def close(self) -> None:
        if self._bus is not None:
            try:
                from ..hwi import JOINT_IDS

                self._bus.disable_torque(list(JOINT_IDS.values()))
            except Exception:
                pass
            self._bus.close()


def run_checks(ctx: Context, checks: List[Check], only=None, skip=None) -> List[Result]:
    results = []
    for c in checks:
        if only and c.group not in only and c.name not in only:
            continue
        if skip and (c.group in skip or c.name in skip):
            continue
        t = time.monotonic()
        if c.interactive and not ctx.interactive:
            r = Result(SKIP, "guided step, run with --interactive")
        elif c.board_only and not ctx.on_board:
            r = Result(SKIP, "needs the MaixCAM2" + (" (dry run)" if ctx.dry_run else ""))
        else:
            ctx.out(f"  .. {c.name}: {c.title}")
            try:
                r = c.fn(ctx) or Result(PASS, "ok")
            except Skip as e:
                r = Result(SKIP, str(e))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                r = Result(FAIL, f"{type(e).__name__}: {e}", details=traceback.format_exc().splitlines()[-4:])
        r.name, r.group, r.seconds = c.name, c.group, round(time.monotonic() - t, 2)
        results.append(r)
        print_result(r, ctx.out)
    return results


_COLORS = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}


def _color(status: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return status
    return f"{_COLORS[status]}{status}\033[0m"


def print_result(r: Result, out=print) -> None:
    out(f"  [{_color(r.status)}] {r.name:22s} {r.summary}")
    for line in r.details:
        out(f"           {line}")
    if r.hint and r.status in (WARN, FAIL):
        out(f"           -> {r.hint}")


def summary(results: List[Result], out=print) -> str:
    counts = {s: sum(r.status == s for r in results) for s in (PASS, WARN, FAIL, SKIP)}
    overall = worst(r.status for r in results)
    out("")
    out(f"{_color(overall)}  {counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} failed, {counts[SKIP]} skipped")
    for r in results:
        if r.status in (FAIL, WARN):
            out(f"  {r.status:4s} {r.name}: {r.summary}" + (f"  -> {r.hint}" if r.hint else ""))
    return overall


def write_report(path: str, results: List[Result], ctx: Context) -> None:
    report = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "dry_run": ctx.dry_run,
        "faults": ctx.sim.faults if ctx.sim else {},
        "config": ctx.config.path if not ctx.config.is_default else None,
        "overall": worst(r.status for r in results),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_jsonable)


def _jsonable(x):
    try:
        import numpy as np

        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    return str(x)

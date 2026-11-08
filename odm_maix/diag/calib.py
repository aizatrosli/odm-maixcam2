"""Calibration steps used by scripts/calibrate.py.

* joint offsets: torque off, the duck is held in its init pose, and the
  offset of each joint is what the servo reads minus the init pose. The same
  ``joints_offsets`` the Pi runtime uses, so the policy sees the same angles.
* nudge: one joint at a time with low torque, adjust its offset by eye.
* joint ranges: move each joint through its range by hand, record the
  extremes, flag joints that cross the encoder wrap and optionally save soft
  limits that the HWI clamps targets to.
* IMU: BNO086 motion-engine calibration, saved to the chip's flash (and a
  record of it to imu_offsets_path).
* servo setup: give a new servo its joint ID and the settings the runtime expects.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from .. import bno08x as bno
from .. import feetech as ft
from ..config import JOINT_NAMES
from ..hwi import INIT_POS, JOINT_IDS

RAD_PER_COUNT = 2 * math.pi / 4096


# ------------------------------------------------------------- config file


def load_config_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    example = os.path.join(os.path.dirname(__file__), "..", "..", "example_duck_config.json")
    with open(example) as f:
        return json.load(f)


def update_config_json(path: str, update) -> str:
    """Apply ``update(data)`` to the JSON at ``path``, keeping a .bak copy. Returns the backup path."""
    data = load_config_json(path)
    backup = ""
    if os.path.exists(path):
        backup = path + time.strftime(".%Y%m%d-%H%M%S.bak")
        shutil.copy2(path, backup)
    update(data)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    os.replace(tmp, path)
    return backup


def save_offsets(path: str, offsets: Dict[str, float]) -> str:
    def upd(d):
        d.setdefault("joints_offsets", {})
        for j, v in offsets.items():
            d["joints_offsets"][j] = round(float(v), 4)

    return update_config_json(path, upd)


def save_joint_limits(path: str, limits: Dict[str, List[float]]) -> str:
    def upd(d):
        d.setdefault("maixcam2", {})["joint_limits"] = {j: [round(lo, 3), round(hi, 3)] for j, (lo, hi) in limits.items()}

    return update_config_json(path, upd)


# ------------------------------------------------------------- offsets


def read_raw_median(bus, ids: List[int], samples: int = 15, dt: float = 0.01) -> List[int]:
    reads = []
    for _ in range(samples):
        try:
            reads.append(bus.read_raw_position(ids))
        except ft.FeetechError:
            pass
        time.sleep(dt)
    if not reads:
        raise ft.FeetechError("no position reads succeeded")
    return [int(v) for v in np.median(np.array(reads), axis=0)]


def capture_offsets(bus, joints: Iterable[str] = JOINT_NAMES, samples: int = 15) -> Dict[str, float]:
    """Torque off, duck held in the init pose: offset = measured - init."""
    joints = list(joints)
    raws = read_raw_median(bus, [JOINT_IDS[j] for j in joints], samples)
    return {j: ft.raw_to_rad_position(r) - INIT_POS[j] for j, r in zip(joints, raws)}


def offsets_table(new: Dict[str, float], old: Dict[str, float]) -> List[str]:
    lines = [f"{'joint':16s} {'old':>8s} {'new':>8s} {'change':>8s}"]
    for j in new:
        o = float(old.get(j, 0.0))
        flag = "  <- large, re-seat the horn?" if abs(new[j]) > 0.2 else ""
        lines.append(f"{j:16s} {o:+8.3f} {new[j]:+8.3f} {new[j] - o:+8.3f}{flag}")
    return lines


def nudge_joint(bus, joint: str, offset: float, ask, out=print, kp: int = 8, step: float = 0.01) -> Optional[float]:
    """Hold one joint at init + offset with low torque and let the person adjust the offset.

    ``ask(prompt) -> str``: '+'/'-' move by step ('++'/'--' by 5 steps), a number sets the
    offset, Enter accepts, 's' skips (keeps the old value), 'q' stops. Returns the new
    offset, None for skip, or raises KeyboardInterrupt for quit.
    """
    sid = JOINT_IDS[joint]
    bus.set_kps([sid], [kp])
    bus.write_goal_position([sid], [INIT_POS[joint] + offset])
    bus.enable_torque([sid])
    try:
        while True:
            bus.write_goal_position([sid], [INIT_POS[joint] + offset])
            ans = ask(f"    {joint:16s} offset {offset:+.3f}  [+ - ++ -- number, Enter=ok, s=skip, q=quit] ").strip()
            if ans == "":
                return offset
            if ans == "s":
                return None
            if ans == "q":
                raise KeyboardInterrupt
            if ans in ("+", "-", "++", "--"):
                offset += step * (5 if len(ans) == 2 else 1) * (1 if ans[0] == "+" else -1)
            else:
                try:
                    offset = float(ans)
                except ValueError:
                    out("    ?")
    finally:
        bus.disable_torque([sid])


# ------------------------------------------------------------- joint ranges


def record_range(bus, joint: str, seconds: float, hz: float = 50.0):
    """Sample one joint while it is moved by hand. Returns (min_rad, max_rad, crossed_wrap, n)."""
    sid = JOINT_IDS[joint]
    raws = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            raws.append(bus.read_raw_position([sid])[0])
        except ft.FeetechError:
            pass
        time.sleep(1.0 / hz)
    if not raws:
        raise ft.FeetechError(f"no reads from {joint}")
    a = np.array(raws)
    crossed = bool(np.any(np.abs(np.diff(a)) > 2048))
    rads = [ft.raw_to_rad_position(int(r)) for r in a]
    return min(rads), max(rads), crossed, len(raws)


def analyse_range(joint: str, lo: float, hi: float, crossed: bool, offset: float) -> Dict:
    """Range in the policy frame (offset removed), and how it sits around the init pose."""
    init = INIT_POS[joint]
    plo, phi = lo - offset, hi - offset
    r = {
        "joint": joint,
        "min": round(plo, 3),
        "max": round(phi, 3),
        "span_deg": round(math.degrees(hi - lo), 1),
        "init_inside": plo <= init <= phi,
        "crossed_wrap": crossed,
        "wrap_margin_rad": round(min(ft.rad_to_raw_position(lo), 4095 - ft.rad_to_raw_position(hi)) * RAD_PER_COUNT, 2),
    }
    problems = []
    if crossed:
        problems.append("crosses the encoder wrap (reading jumps by 2 pi): re-seat the horn")
    elif r["wrap_margin_rad"] < 0.2:
        problems.append("ends within 0.2 rad of the encoder wrap")
    if not r["init_inside"]:
        problems.append(f"init pose {init:+.2f} is outside the measured range")
    if hi - lo < math.radians(20):
        problems.append("moved less than 20 degrees: was it moved through its full range?")
    r["problems"] = problems
    return r


def soft_limits(ranges: Dict[str, Dict], margin: float = 0.05) -> Dict[str, List[float]]:
    out = {}
    for j, r in ranges.items():
        if r["crossed_wrap"] or r["span_deg"] < 20:
            continue
        out[j] = [r["min"] + margin, r["max"] - margin]
    return out


# ------------------------------------------------------------- IMU


def calibrate_imu(dev, path: str, timeout: float = 180.0, out=print, poll=0.5, on_poll=None) -> Dict[str, int]:
    """Run the BNO086's own calibration until accel and gyro accuracy reach 3, then save it on the chip.

    The magnetometer accuracy is shown but not waited for: the policy does not
    use it, and the servos and battery disturb it anyway.
    """
    dev.begin_calibration()
    end = time.monotonic() + timeout
    last = None
    while True:
        if on_poll:
            on_poll()
        s = dev.calibration_status()
        if s != last:
            out(f"    accuracy {s}")
            last = s
        if s["accel"] == 3 and s["gyro"] == 3:
            break
        if time.monotonic() > end:
            raise TimeoutError(f"IMU calibration did not finish in {timeout:.0f} s (last accuracy {s})")
        time.sleep(poll)
    dev.save_calibration()
    bno.save_calibration_record(path, s, dev)
    dev.enable_report(bno.REPORT_MAGNETOMETER, 0)  # back to accel + gyro only
    return s


# ------------------------------------------------------------- servo setup


def find_single_servo(bus, ids: Iterable[int] = range(0, 254)) -> List[int]:
    return [i for i in ids if bus.ping(i)]


def configure_servo(bus, current_id: int, new_id: int) -> Dict[str, int]:
    """Give a servo its joint ID, position mode and no return delay. Returns the settings read back."""
    if current_id != new_id:
        bus.write_eeprom(current_id, ft.REG_ID, [new_id])
    bus.write_eeprom(new_id, ft.REG_RETURN_DELAY, [0])
    if bus.read_u8(new_id, ft.REG_MODE) != 0:
        bus.write_eeprom(new_id, ft.REG_MODE, [0])
    return {
        "id": bus.read_u8(new_id, ft.REG_ID),
        "mode": bus.read_u8(new_id, ft.REG_MODE),
        "return_delay": bus.read_u8(new_id, ft.REG_RETURN_DELAY),
        "min_angle": bus.read_u16(new_id, ft.REG_MIN_ANGLE_LIMIT),
        "max_angle": bus.read_u16(new_id, ft.REG_MAX_ANGLE_LIMIT),
    }


def center_servo(bus, sid: int, kp: int = 16) -> int:
    """Drive a loose servo to raw 2048 (0 rad) so its horn can be fitted centred."""
    bus.set_kps([sid], [kp])
    bus.enable_torque([sid])
    bus.write_goal_position([sid], [0.0])
    time.sleep(1.0)
    return bus.read_raw_position([sid])[0]

"""Pre-flight checks for the duck. See scripts/diagnose.py for how to run them.

Every check is safe with the robot on a stand: none of them enables servo
torque. Guided checks (``interactive=True``) ask the person to move the duck
or press a foot and only run with ``--interactive``.
"""

from __future__ import annotations

import glob
import importlib
import math
import os
import time
from typing import List

import numpy as np

from .. import bno08x as bno
from .. import feetech as ft
from ..config import JOINT_NAMES
from ..hwi import INIT_POS, JOINT_IDS
from .core import FAIL, PASS, WARN, Check, Result, Skip, worst

ALL_IDS = [JOINT_IDS[j] for j in JOINT_NAMES]
NAME_OF = {v: k for k, v in JOINT_IDS.items()}
RAD_PER_COUNT = 2 * math.pi / 4096

# thresholds, in one place so they are easy to argue with
OFFSET_WARN_RAD = 0.20  # more than this usually means a horn one spline tooth off
OFFSET_FAIL_RAD = 0.60
VOLT_FAIL, VOLT_WARN, VOLT_HIGH = 6.4, 7.0, 8.5  # 2S Li-ion: 8.4 full, ~6.0 empty
VOLT_SPREAD_WARN = 0.4
TEMP_WARN, TEMP_FAIL = 55, 65
WRAP_FAIL_RAD, WRAP_WARN_RAD = 0.35, 0.8
BUS_P99_WARN_MS, BUS_P99_FAIL_MS = 4.0, 8.0
LOOP_HZ = 50


def _stats_ms(seconds) -> dict:
    a = np.asarray(seconds) * 1000
    return {"mean_ms": round(float(a.mean()), 3), "p99_ms": round(float(np.percentile(a, 99)), 3), "max_ms": round(float(a.max()), 3)}


# ============================================================ config


def check_config_file(ctx):
    c = ctx.config
    if c.is_default:
        return Result(WARN, f"{c.path} not found, running on defaults", hint="cp example_duck_config.json ~/duck_config.json")
    import json

    data = json.load(open(c.path))
    missing = [j for j in JOINT_NAMES if j not in data.get("joints_offsets", {})]
    if missing:
        return Result(WARN, f"joints_offsets has no entry for {', '.join(missing)} (taken as 0)", hint="run scripts/calibrate.py offsets")
    return Result(PASS, f"{c.path} loaded", data={"path": c.path})


def check_config_offsets(ctx):
    off = {j: float(ctx.config.joints_offsets.get(j, 0.0)) for j in JOINT_NAMES}
    biggest = max(off, key=lambda j: abs(off[j]))
    data = {"offsets": off}
    if all(v == 0.0 for v in off.values()):
        return Result(WARN, "all joint offsets are 0, the duck has not been calibrated", hint="python3 scripts/calibrate.py offsets", data=data)
    big = [f"{j} {off[j]:+.3f}" for j in JOINT_NAMES if abs(off[j]) > OFFSET_WARN_RAD]
    if abs(off[biggest]) > OFFSET_FAIL_RAD:
        return Result(FAIL, f"{biggest} offset {off[biggest]:+.3f} rad is too large", details=big,
                      hint="re-seat that servo horn closer to its centre, then recalibrate", data=data)
    if big:
        return Result(WARN, f"{len(big)} offsets above {OFFSET_WARN_RAD} rad", details=big,
                      hint="a horn may be one spline tooth off; re-seating it keeps the joint range symmetric", data=data)
    return Result(PASS, f"largest offset {biggest} {off[biggest]:+.3f} rad", data=data)


def check_config_files(ctx):
    m = ctx.config.maixcam2
    res, details = [], []

    def need(path, what, status=FAIL):
        if path and os.path.exists(os.path.expanduser(path)):
            details.append(f"ok   {what}: {path}")
            res.append(PASS)
        else:
            details.append(f"{'MISSING' if status == FAIL else 'absent '} {what}: {path}")
            res.append(status)

    backend, policy = m["policy_backend"], m["policy_path"]
    need(policy, f"policy ({backend})")
    if backend == "npu" and not str(policy).endswith(".mud"):
        details.append("policy_backend is npu but policy_path is not a .mud file")
        res.append(FAIL)
    if backend == "onnx" and not str(policy).endswith(".onnx"):
        details.append("policy_backend is onnx but policy_path is not a .onnx file")
        res.append(FAIL)
    if m.get("nb_steps_in_period"):
        details.append(f"ok   nb_steps_in_period = {m['nb_steps_in_period']}")
    else:
        need(m["polynomial_coefficients_path"], "gait period (polynomial_coefficients.pkl)")
    need(m["imu_offsets_path"], "IMU calibration", WARN)
    if ctx.config.feature("speaker"):
        d = os.path.expanduser(m.get("expression", {}).get("sounds_dir", "~/sounds"))
        need(d, "sounds dir (scripts/make_sounds.py)", WARN)
    need(m["vision"]["model_path"], "YOLO model (follow mode)", WARN)
    status = worst(res)
    n_bad = sum(r != PASS for r in res)
    return Result(status, "all files present" if status == PASS else f"{n_bad} file(s) missing", details=details,
                  hint="copy the policy and polynomial_coefficients.pkl from the Pi runtime; run calibrate.py imu")


def _pins_in_use(ctx):
    """(pin, user) pairs from the config."""
    m = ctx.config.maixcam2
    from ..expression import DEFAULT_EXPRESSION_CFG

    used = []
    for pin, func in (m.get("pinmap") or {}).items():
        used.append((pin, f"pinmap {func}"))
    for side in ("left", "right"):
        used.append((m[f"{side}_foot_gpio"].replace("GPIO", ""), f"{side} foot"))
    exp = {**DEFAULT_EXPRESSION_CFG, **m.get("expression", {})}
    if ctx.config.feature("antennas"):
        used += [(exp["left_antenna"][0], "left antenna"), (exp["right_antenna"][0], "right antenna")]
    if ctx.config.feature("eyes"):
        used += [(exp["left_eye"][0], "left eye"), (exp["right_eye"][0], "right eye")]
    return used


def check_config_pins(ctx):
    m = ctx.config.maixcam2
    pinmap = m.get("pinmap") or {}
    users = {}
    for pin, who in _pins_in_use(ctx):
        if not who.startswith("pinmap"):
            users.setdefault(pin, []).append(who)
    problems = []
    for pin, whos in users.items():
        if len(whos) > 1:
            problems.append(f"{pin} used by {', '.join(whos)}")
        func = pinmap.get(pin)
        for who in whos:
            want = ("GPIO",) if "foot" in who else ("PWM",) if "antenna" in who else ("PWM", "GPIO")
            if func and not func.startswith(want):
                problems.append(f"{pin} is set to {func} in pinmap but the {who} needs {'/'.join(want)}")
    by_pin = {p: list(users.get(p, [])) + ([f"pinmap {pinmap[p]}"] if p in pinmap else []) for p in set(users) | set(pinmap)}
    pinmap = m.get("pinmap") or {}
    if m.get("imu_i2c_bus") in (None, 7) and not ("A8" in pinmap and "A9" in pinmap):
        problems.append("pinmap has no A8/A9 I2C7 entries, the BNO086 will not be reachable")
    if m.get("imu_i2c_bus") is not None and m.get("imu_address") not in bno.ADDRESSES:
        problems.append(f"imu_address is 0x{int(m.get('imu_address') or 0):02x}; the BNO086 answers at 0x4a or 0x4b (0x28 was the BNO055)")
    if m.get("imu_axis_remap"):
        try:
            bno.remap_from_config(ctx.config)
        except (ValueError, TypeError) as e:
            problems.append(str(e))
    if "ttyS4" in str(m["servo_port"]) and not ("A21" in pinmap and "A22" in pinmap):
        problems.append("servo_port is /dev/ttyS4 but pinmap has no A21/A22 UART4 entries")
    if m["left_foot_gpio"] == m["right_foot_gpio"]:
        problems.append("left and right foot use the same GPIO")
    if problems:
        return Result(FAIL, f"{len(problems)} pin conflict(s)", details=problems, hint="see docs/WIRING.md")
    return Result(PASS, f"{len(by_pin)} pins, no conflicts", details=[f"{p}: {', '.join(w)}" for p, w in sorted(by_pin.items())])


# ============================================================ board


def check_python_deps(ctx):
    need = ["numpy", "serial"]
    if ctx.config.maixcam2["policy_backend"] == "onnx":
        need.append("onnxruntime")
    missing = []
    for mod in need:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    opt = []
    try:
        importlib.import_module("pygame")
    except Exception:
        opt.append("pygame (gamepad) not installed")
    if missing:
        return Result(FAIL, f"missing: {', '.join(missing)}", details=opt, hint="pip3 install -r requirements.txt")
    return Result(PASS, "required modules import", details=opt)


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def check_board_system(ctx):
    details, res, data = [], [], {}
    try:
        import maix

        ver = getattr(maix, "__version__", None)
        if ver is None:
            from maix import app

            ver = app.version()
        details.append(f"MaixPy {ver}")
        data["maixpy"] = str(ver)
    except Exception as e:
        details.append(f"MaixPy version unknown ({e})")
    temps = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        v = _read(z)
        if v and v.lstrip("-").isdigit():
            temps.append(int(v) / 1000)
    if temps:
        t = max(temps)
        data["soc_temp_c"] = t
        details.append(f"SoC {t:.0f} C")
        res.append(FAIL if t > 85 else WARN if t > 70 else PASS)
    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov:
        details.append(f"governor {gov}")
        data["governor"] = gov
        if gov != "performance":
            res.append(WARN)
            details.append("governor is not 'performance': loop jitter goes up (tools/system/setup_board.sh apply)")
    for line in (_read("/proc/meminfo", "") or "").splitlines():
        if line.startswith("MemAvailable:"):
            mb = int(line.split()[1]) / 1024
            data["mem_available_mb"] = round(mb)
            details.append(f"{mb:.0f} MB RAM available")
            res.append(FAIL if mb < 100 else WARN if mb < 200 else PASS)
    cmm = _read("/proc/ax_proc/mem_cmm_info")
    if cmm:
        import re

        mt, mu = re.search(r"total=(\d+)KB", cmm), re.search(r"used=(\d+)KB", cmm)
        if mt and mu:
            total, used = int(mt.group(1)) / 1024, int(mu.group(1)) / 1024
            details.append(f"CMM {used:.0f}/{total:.0f} MB used")
            data["cmm_used_mb"], data["cmm_total_mb"] = used, total
    status = worst(res) if res else PASS
    return Result(status, "board ok" if status == PASS else "see details", details=details, data=data)


# ============================================================ servos


def _ping(bus, sid, tries=3) -> bool:
    return any(bus.ping(sid) for _ in range(tries))


def check_servo_ping(ctx):
    bus = ctx.bus()
    found, missing = [], []
    for j in JOINT_NAMES:
        (found if _ping(bus, JOINT_IDS[j]) else missing).append(j)
    ctx.shared["servos_found"] = [JOINT_IDS[j] for j in found]
    ctx.shared["servos_missing"] = [JOINT_IDS[j] for j in missing]
    if not found:
        return Result(FAIL, f"no servo answers on {ctx.config.maixcam2['servo_port']}",
                      hint="check the 7.4 V supply to the driver board, the USB host mode and servo_baudrate",
                      data={"missing": missing})
    if missing:
        return Result(FAIL, f"{len(found)}/14 answer; missing " + ", ".join(f"{j} (id {JOINT_IDS[j]})" for j in missing),
                      hint="check that servo's cable, then look for it under another ID (servo.scan)", data={"missing": missing})
    return Result(PASS, "all 14 servos answer", data={"found": found})


def check_servo_scan(ctx):
    bus = ctx.bus()
    if not ctx.shared.get("servos_missing") and not ctx.opts.get("scan"):
        raise Skip("all servos found (use --scan to scan anyway)")
    t = time.monotonic()
    extra = [i for i in range(0, 254) if i not in ALL_IDS and _ping(bus, i, tries=1)]
    data = {"extra_ids": extra, "seconds": round(time.monotonic() - t, 1)}
    if extra:
        hint = "ID 1 is the factory default: give it its joint ID with scripts/calibrate.py servo-id" if 1 in extra else \
            "give each servo its joint ID with scripts/calibrate.py servo-id"
        return Result(FAIL, f"servos on unexpected IDs: {extra}", hint=hint, data=data)
    if ctx.shared.get("servos_missing"):
        return Result(WARN, "no servo on any other ID: the missing ones are unpowered or disconnected", data=data,
                      hint="check the daisy-chain cable before the first missing servo")
    return Result(PASS, "no stray IDs on the bus", data=data)


def _found(ctx) -> List[int]:
    ids = ctx.shared.get("servos_found")
    if ids is None:
        bus = ctx.bus()
        ids = [i for i in ALL_IDS if _ping(bus, i)]
        ctx.shared["servos_found"] = ids
    if not ids:
        raise Skip("no servos answer")
    return ids


def check_servo_eeprom(ctx):
    bus = ctx.bus()
    problems, warns, data = [], [], {}
    for sid in _found(ctx):
        j = NAME_OF[sid]
        mode = bus.read_u8(sid, ft.REG_MODE)
        delay = bus.read_u8(sid, ft.REG_RETURN_DELAY)
        lo = bus.read_u16(sid, ft.REG_MIN_ANGLE_LIMIT)
        hi = bus.read_u16(sid, ft.REG_MAX_ANGLE_LIMIT)
        tlim = bus.read_u16(sid, ft.REG_TORQUE_LIMIT)
        data[j] = {"mode": mode, "return_delay": delay, "angle_limits": [lo, hi], "torque_limit": tlim}
        if mode != 0:
            problems.append(f"{j}: mode {mode}, needs 0 (position servo)")
        if delay > 0:
            warns.append(f"{j}: return delay {delay * 2} us (0 is fastest)")
        if (lo, hi) not in ((0, 0), (0, 4095)):
            target = ft.rad_to_raw_position(INIT_POS[j] + ctx.config.joints_offsets.get(j, 0.0))
            if not lo <= target <= hi:
                problems.append(f"{j}: init pose (raw {target}) outside angle limits {lo}..{hi}")
            else:
                warns.append(f"{j}: angle limits {lo}..{hi}")
        if tlim < 500:
            warns.append(f"{j}: torque limit {tlim}/1000")
    if problems:
        return Result(FAIL, f"{len(problems)} servo setting(s) wrong", details=problems + warns,
                      hint="fix them with scripts/calibrate.py servo-id --id N (it writes the stock settings)", data=data)
    if warns:
        return Result(WARN, f"{len(warns)} unusual servo setting(s)", details=warns, data=data)
    return Result(PASS, "position mode, no return delay, no angle limits", data=data)


def check_servo_power(ctx):
    bus = ctx.bus()
    volts, temps, errors = {}, {}, {}
    for sid in _found(ctx):
        j = NAME_OF[sid]
        err, raw = bus.read_with_status(sid, ft.REG_PRESENT_VOLTAGE, 2)
        volts[j], temps[j] = raw[0] / 10.0, raw[1]
        if err:
            errors[j] = ft.describe_status_error(err)
    v = np.array(list(volts.values()))
    vmin, vmax = float(v.min()), float(v.max())
    hottest = max(temps, key=temps.get)
    details = [f"voltage {vmin:.1f}..{vmax:.1f} V, hottest {hottest} {temps[hottest]} C"]
    details += [f"{j}: status error {e}" for j, e in errors.items()]
    data = {"voltage": volts, "temperature": temps, "errors": errors}
    res, hints = [PASS], []
    if vmin < VOLT_FAIL:
        res.append(FAIL)
        hints.append("charge the battery")
    elif vmin < VOLT_WARN:
        res.append(WARN)
        hints.append("battery is low; servos lose torque below ~7 V")
    if vmax > VOLT_HIGH:
        res.append(WARN)
        hints.append("supply above 8.5 V: STS3215 7.4 V servos are rated to 8.4 V")
    if vmax - vmin > VOLT_SPREAD_WARN:
        res.append(WARN)
        details.append(f"{vmax - vmin:.2f} V spread between servos")
        hints.append("big spread = a loose or thin power lead in the chain")
    if temps[hottest] >= TEMP_FAIL:
        res.append(FAIL)
        hints.append(f"{hottest} is hot with torque off: check for a stalled or shorted servo")
    elif temps[hottest] >= TEMP_WARN:
        res.append(WARN)
    if errors:
        res.append(FAIL)
        hints.append("servo error flags set: power-cycle; if it stays, that servo is overloaded or damaged")
    status = worst(res)
    return Result(status, f"{vmin:.1f} V min, {temps[hottest]} C max" + (f", {len(errors)} with errors" if errors else ""),
                  details=details, hint="; ".join(hints), data=data)


def _wrap_margin(raw: int) -> float:
    """Radians between this encoder count and the 4095 -> 0 wrap."""
    return min(raw, 4095 - raw) * RAD_PER_COUNT


def check_servo_pose(ctx):
    bus = ctx.bus()
    ids = _found(ctx)
    if ctx.interactive:
        ctx.prompt("pose.init", "Torque is off. Put the duck in its standing init pose (legs bent, head level)")
    raws = bus.read_raw_position(ids)
    details, res, data = [], [PASS], {}
    worst_wrap = (None, 9.0)
    for sid, raw in zip(ids, raws):
        j = NAME_OF[sid]
        present = ft.raw_to_rad_position(raw)
        dev = present - ctx.config.joints_offsets.get(j, 0.0) - INIT_POS[j]
        target_raw = ft.rad_to_raw_position(INIT_POS[j] + ctx.config.joints_offsets.get(j, 0.0))
        margin = min(_wrap_margin(raw), _wrap_margin(target_raw))
        data[j] = {"raw": raw, "rad": round(present, 3), "from_init": round(dev, 3), "wrap_margin_rad": round(margin, 2)}
        if margin < worst_wrap[1]:
            worst_wrap = (j, margin)
        if margin < WRAP_FAIL_RAD:
            res.append(FAIL)
            details.append(f"{j}: raw {raw}, only {margin:.2f} rad from the encoder wrap")
        elif margin < WRAP_WARN_RAD:
            res.append(WARN)
            details.append(f"{j}: raw {raw}, {margin:.2f} rad from the encoder wrap")
        if ctx.interactive and abs(dev) > 0.15:
            res.append(WARN)
            details.append(f"{j}: {dev:+.2f} rad from the init pose")
    status = worst(res)
    hint = ""
    if status == FAIL:
        hint = "a joint passing raw 0/4095 makes the position jump by 2 pi: re-seat that horn by half a turn, or move it towards 2048"
    elif any("init pose" in d for d in details):
        hint = "offsets look stale: python3 scripts/calibrate.py offsets"
    j, m = worst_wrap
    return Result(status, f"closest to the encoder wrap: {j} {m:.2f} rad", details=details, hint=hint, data=data)


def check_servo_timing(ctx):
    bus = ctx.bus()
    ids = _found(ctx)
    n = int(ctx.opts.get("bus_reads", 300))
    times, errors = [], 0
    for _ in range(n):
        t = time.perf_counter()
        try:
            bus.read_present_position_velocity(ids)
        except ft.FeetechError:
            errors += 1
        times.append(time.perf_counter() - t)
    st = _stats_ms(times)
    rate = errors / n
    data = {**st, "errors": errors, "reads": n, "servos": len(ids)}
    summary = f"{len(ids)}-servo read mean {st['mean_ms']:.2f} ms, p99 {st['p99_ms']:.2f} ms, {errors}/{n} errors"
    if rate > 0.01:
        return Result(FAIL, summary, hint="bus errors: check the baud rate, cable length and the ground between boards", data=data)
    if st["p99_ms"] > BUS_P99_FAIL_MS:
        return Result(FAIL, summary, hint="reads eat most of the 20 ms step: check return delay and the USB adapter latency", data=data)
    if errors or st["p99_ms"] > BUS_P99_WARN_MS:
        return Result(WARN, summary, hint="occasional bus errors or slow reads; run it again under load", data=data)
    return Result(PASS, summary, data=data)


# ============================================================ IMU


def check_imu_chip(ctx):
    dev = ctx.imu_device()
    p = dev.product or dev.request_product_id()
    where = f"i2c-{dev.bus_number} @0x{dev.address:02x}"
    data = {"bus": dev.bus_number, "address": dev.address, **p}
    summary = f"BNO08x on {where}, firmware {p['sw_version']}, last reset: {p['reset_cause_name']}"
    if p["reset_cause"] == 3:
        return Result(FAIL, summary, data=data,
                      hint="the chip's watchdog reset it: check the 3V3 supply and I2C wiring; power-cycle and run again")
    if dev.bus_number != 7 and ctx.config.maixcam2["imu_i2c_bus"] is None:
        return Result(WARN, summary + ", docs/WIRING.md puts it on i2c-7", data=data,
                      hint="fine if intended; set maixcam2.imu_i2c_bus to skip the scan")
    return Result(PASS, summary, data=data)


def check_imu_calibration(ctx):
    dev = ctx.imu_device()
    path = ctx.config.maixcam2["imu_offsets_path"]
    rec = bno.load_calibration_record(path)
    time.sleep(0.05)
    s = dev.calibration_status()
    data = {"record_file": path, "record": rec, "status": s}
    if not rec:
        return Result(WARN, f"no calibration record at {path}; accuracy {s}", hint="python3 scripts/calibrate.py imu", data=data)
    if s["accel"] < 2 or s["gyro"] < 2:
        return Result(WARN, f"calibrated on {rec['bno08x_calibration_saved']}, but accuracy is low now {s}",
                      hint="keep the duck still a few seconds; if it stays low, python3 scripts/calibrate.py imu", data=data)
    return Result(PASS, f"calibrated on {rec['bno08x_calibration_saved']}; accuracy {s}", data=data)


def _imu_samples(dev, seconds, hz=100):
    gyro, acc, lat, errors = [], [], [], 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        t = time.perf_counter()
        try:
            g, a = dev.gyro_and_acceleration()
            gyro.append(g)
            acc.append(a)
        except OSError:
            errors += 1
        lat.append(time.perf_counter() - t)
        time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - t)))
    return np.array(gyro), np.array(acc), lat, errors


def check_imu_still(ctx):
    dev = ctx.imu_device()
    if ctx.interactive:
        ctx.prompt("imu.upright", "Stand the duck upright on a flat surface and don't touch it")
    gyro, acc, lat, errors = _imu_samples(dev, float(ctx.opts.get("imu_seconds", 2.0)))
    if len(acc) == 0:
        return Result(FAIL, f"no IMU reads succeeded ({errors} errors)", hint="I2C errors: lower I2C7 to 100 kHz (docs/HARDWARE_CHECKLIST.md)")
    g = float(np.linalg.norm(acc.mean(0)))
    bias = gyro.mean(0)
    noise = float(gyro.std(0).max())
    st = _stats_ms(lat)
    data = {"gravity": round(g, 3), "gyro_bias": bias.round(4).tolist(), "gyro_noise": round(noise, 4),
            "acc_mean": acc.mean(0).round(3).tolist(), "errors": errors, "samples": len(acc), **st}
    ctx.shared["imu_acc_mean"] = acc.mean(0)
    details = [f"|g| {g:.2f} m/s2, gyro bias {np.around(bias, 3).tolist()} rad/s, read p99 {st['p99_ms']:.1f} ms"]
    res, hints = [PASS], []
    if errors:
        res.append(FAIL if errors > 0.01 * (len(acc) + errors) else WARN)
        hints.append(f"{errors} I2C errors: the BNO086 clock-stretches; drop I2C7 to 100 kHz, keep the wires short")
    if abs(g - 9.81) > 0.6:
        res.append(FAIL)
        hints.append("gravity is off: the duck is moving, or the accelerometer needs calibrating")
    elif abs(g - 9.81) > 0.3:
        res.append(WARN)
    if np.abs(bias).max() > 0.05 or noise > 0.05:
        res.append(WARN)
        hints.append("gyro not settled: keep the duck still, or recalibrate")
    if st["p99_ms"] > 3:
        res.append(WARN)
        hints.append("slow I2C reads, the IMU thread may lag the 50 Hz loop")
    return Result(worst(res), f"|g| = {g:.2f}, gyro bias {np.abs(bias).max():.3f} rad/s, {errors} errors",
                  details=details, hint="; ".join(hints), data=data)


def check_imu_upright(ctx):
    acc = ctx.shared.get("imu_acc_mean")
    if acc is None:
        _, a, _, _ = _imu_samples(ctx.imu_device(), 1.0)
        if len(a) == 0:
            raise Skip("no IMU data")
        acc = a.mean(0)
    x, y, z = acc
    data = {"acc_mean": np.asarray(acc).round(3).tolist(), "imu_upside_down": ctx.config.imu_upside_down}
    summary = f"accelero [{x:+.2f}, {y:+.2f}, {z:+.2f}] (upright should be about [0, 0, +9.8])"
    if z < -8.0:
        return Result(FAIL, summary, data=data,
                      hint=f"z is upside down: set imu_upside_down to {str(not ctx.config.imu_upside_down).lower()} in duck_config.json")
    if z < 8.5:
        return Result(FAIL if z < 6 else WARN, summary, data=data,
                      hint="the duck is not upright and level, or the BNO086 is mounted on its side (set maixcam2.imu_axis_remap)")
    if abs(x) > 1.5 or abs(y) > 1.5:
        return Result(WARN, summary, hint="tilted more than ~9 degrees: check the surface, or the IMU mounting", data=data)
    return Result(PASS, summary, data=data)


def check_imu_axes(ctx):
    """Guided: tilt and turn the duck and check every axis sign the policy relies on."""
    dev = ctx.imu_device()
    details, res, data = [], [], {}

    hold = float(ctx.opts.get("hold_seconds", 0.6))
    turn = float(ctx.opts.get("turn_seconds", 3.0))

    def mean_acc(seconds=hold):
        _, a, _, _ = _imu_samples(dev, seconds)
        return a.mean(0)

    ctx.prompt("imu.nose_down", "Tilt the duck NOSE DOWN about 30 degrees and hold it")
    a = mean_acc()
    ok = a[0] < -2.5 and abs(a[0]) > abs(a[1])
    data["nose_down"] = a.round(2).tolist()
    details.append(f"nose down : accelero x {a[0]:+.2f} (want < -2.5)  {'ok' if ok else 'WRONG'}")
    res.append(PASS if ok else FAIL)

    ctx.prompt("imu.right_down", "Now roll it RIGHT SIDE DOWN about 30 degrees and hold it")
    a = mean_acc()
    ok = a[1] > 2.5 and abs(a[1]) > abs(a[0])
    data["right_down"] = a.round(2).tolist()
    details.append(f"right down: accelero y {a[1]:+.2f} (want > +2.5)  {'ok' if ok else 'WRONG'}")
    res.append(PASS if ok else FAIL)

    ctx.prompt("imu.turn_left", "Put it back upright. After Enter, turn it about 90 degrees to ITS LEFT (counter-clockwise from above) within 3 s")
    gyro, _, lat, _ = _imu_samples(dev, turn)
    dt = turn / max(1, len(gyro))
    yaw = float(gyro[:, 2].sum() * dt) if len(gyro) else 0.0
    other = float(np.abs(gyro[:, :2].sum(0) * dt).max()) if len(gyro) else 0.0
    ok = yaw > 0.7 and yaw > other
    data["turn_left_yaw_rad"] = round(yaw, 2)
    details.append(f"turn left : integrated gyro z {math.degrees(yaw):+.0f} deg (want about +90)  {'ok' if ok else 'WRONG'}")
    res.append(PASS if ok else FAIL)
    ctx.sim and ctx.sim.act("imu.upright")

    status = worst(res)
    return Result(status, "IMU axes match the policy frame (x forward, y left, z up)" if status == PASS else "IMU axes do not match the policy frame",
                  details=details, data=data,
                  hint="check the BNO086 orientation against the Open Duck Mini build guide, imu_upside_down and imu_axis_remap")


# ============================================================ feet


def check_feet_stable(ctx):
    feet = ctx.feet()
    states = []
    for _ in range(50):
        states.append(tuple(feet.get()))
        time.sleep(0.02)
    toggles = sum(a != b for a, b in zip(states, states[1:]))
    last = states[-1]
    desc = f"left {'down' if last[0] else 'up'}, right {'down' if last[1] else 'up'}"
    data = {"toggles": toggles, "last": list(last)}
    if toggles > 2:
        return Result(FAIL, f"inputs flicker ({toggles} changes in 1 s, nobody touching)", data=data,
                      hint="fit the 10 kOhm pull-ups from B20/B21 to 3V3 (maix.gpio has no internal pull-up)")
    return Result(PASS, f"steady: {desc}", data=data)


def check_feet_press(ctx):
    feet = ctx.feet()

    def wait_state(want, timeout=5.0):
        end = time.monotonic() + (0.2 if ctx.dry_run else timeout)
        s = tuple(feet.get())
        while time.monotonic() < end:
            s = tuple(feet.get())
            if s == want:
                break
            time.sleep(0.02)
        return s

    details, res = [], []
    steps = [
        ("feet.both_up", "Lift the duck so neither foot touches anything", (False, False)),
        ("feet.left", "Press and hold the LEFT foot switch (the duck's left)", (True, False)),
        ("feet.right", "Now release it and press the RIGHT foot switch", (False, True)),
    ]
    seen = {}
    for key, text, want in steps:
        ctx.prompt(key, text)
        got = wait_state(want)
        seen[key] = got
        ok = got == want
        details.append(f"{key.split('.')[1]:8s} got left={'down' if got[0] else 'up'} right={'down' if got[1] else 'up'}  {'ok' if ok else 'WRONG'}")
        res.append(PASS if ok else FAIL)
    ctx.sim and ctx.sim.act("feet.both_down")
    hint = ""
    if seen.get("feet.left") == (False, True) and seen.get("feet.right") == (True, False):
        hint = "left and right are swapped: swap left_foot_gpio and right_foot_gpio in duck_config.json"
    elif seen.get("feet.both_up") != (False, False):
        hint = "a switch reads down while lifted: check it is wired to the NO contact, and the pull-up"
    status = worst(res)
    return Result(status, "both switches read the right side" if status == PASS else "foot switch mismatch", details=details, hint=hint)


# ============================================================ camera / NPU / audio (board only)


def check_camera(ctx):
    from maix import camera

    cam = camera.Camera(640, 480)
    try:
        for _ in range(5):
            cam.read()
        t = time.perf_counter()
        n = 30
        for _ in range(n):
            img = cam.read()
        fps = n / (time.perf_counter() - t)
        mean = None
        try:
            from maix import image

            arr = image.image2cv(img, False, False)
            mean = float(np.asarray(arr).mean())
        except Exception:
            pass
    finally:
        try:
            cam.close()
        except Exception:
            pass
    data = {"fps": round(fps, 1), "mean_pixel": mean}
    if mean is not None and mean < 5:
        return Result(FAIL, f"{fps:.0f} fps but the image is black", hint="lens cap, or the sensor ribbon cable", data=data)
    if fps < 20:
        return Result(WARN, f"camera only {fps:.0f} fps at 640x480", data=data)
    return Result(PASS, f"camera {fps:.0f} fps at 640x480" + (f", mean pixel {mean:.0f}" if mean is not None else ""), data=data)


def _rand_obs(n=200, size=101):
    return np.random.default_rng(0).normal(0, 0.5, size=(n, size)).astype(np.float32)


def check_npu_policy(ctx):
    from ..policy import NpuPolicy, OnnxPolicy

    m = ctx.config.maixcam2
    mud = ctx.opts.get("mud") or (m["policy_path"] if m["policy_backend"] == "npu" else None)
    if not mud or not os.path.exists(mud):
        raise Skip("no NPU policy (.mud) configured; pass --mud to test one")
    obs = np.load(ctx.opts["obs"]).astype(np.float32) if ctx.opts.get("obs") else _rand_obs()
    npu = NpuPolicy(mud)
    for o in obs[:10]:
        npu.infer(o)
    t = time.perf_counter()
    out = np.array([npu.infer(o) for o in obs])
    ms = (time.perf_counter() - t) / len(obs) * 1000
    data = {"npu_ms": round(ms, 3)}
    onnx = ctx.opts.get("onnx") or (m["policy_path"] if m["policy_backend"] == "onnx" else None)
    if onnx and os.path.exists(onnx):
        ref = np.array([OnnxPolicy(onnx).infer(o) for o in obs])
        err = float(np.abs(out - ref).max())
        data["max_abs_error"] = round(err, 4)
        if err > 0.04:
            return Result(WARN if ctx.opts.get("obs") is None else FAIL, f"NPU {ms:.2f} ms, max error {err:.3f} vs ONNX", data=data,
                          hint="with random inputs this is only indicative; pass --obs recorded observations")
        return Result(PASS, f"NPU {ms:.2f} ms per step, max error {err:.3f} vs ONNX", data=data)
    return Result(PASS, f"NPU policy loads and runs, {ms:.2f} ms per step", data=data)


def check_npu_yolo(ctx):
    path = ctx.config.maixcam2["vision"]["model_path"]
    if not os.path.exists(path):
        raise Skip(f"{path} not present (only needed for --follow)")
    from maix import image, nn

    det = nn.YOLO11(model=path, dual_buff=False)
    img = image.Image(det.input_width(), det.input_height(), det.input_format())
    det.detect(img)
    t = time.perf_counter()
    for _ in range(20):
        det.detect(img)
    ms = (time.perf_counter() - t) / 20 * 1000
    return Result(PASS if ms < 40 else WARN, f"YOLO11 {ms:.1f} ms per frame on the NPU", data={"ms": round(ms, 2)})


def _write_wav(path, pcm, rate):
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def check_speaker(ctx):
    from maix import audio

    from ..sounds import RATE, synth

    path = "/tmp/odm_diag_quack.wav"
    _write_wav(path, synth("hello"), RATE)
    p = audio.Player(path)
    p.volume(int(ctx.config.maixcam2.get("expression", {}).get("volume", 70)))
    p.play()
    if ctx.confirm("Did you hear three quacks?", default=True):
        return Result(PASS, "speaker plays")
    return Result(FAIL, "no sound heard", hint="check the volume setting and that nothing else holds the audio device")


def check_microphone(ctx):
    from maix import audio

    r = audio.Recorder()
    try:
        r.volume(100)
    except Exception:
        pass
    if ctx.interactive:
        ctx.prompt("mic", "Clap or talk near the duck for one second after Enter")
    data = r.record(1000)
    pcm = np.frombuffer(bytes(data), dtype=np.int16).astype(np.float32)
    if pcm.size == 0:
        return Result(FAIL, "recorder returned no samples")
    rms = float(np.sqrt(np.mean(pcm**2)))
    peak = float(np.abs(pcm).max())
    d = {"rms": round(rms, 1), "peak": peak, "samples": int(pcm.size)}
    if peak < 50:
        return Result(WARN, f"microphone looks silent (peak {peak:.0f})", data=d, hint="make some noise and run again with --interactive")
    return Result(PASS, f"microphone rms {rms:.0f}, peak {peak:.0f}", data=d)


def check_expression_outputs(ctx):
    feats = [f for f in ("antennas", "eyes") if ctx.config.feature(f)]
    if not feats:
        raise Skip("antennas and eyes are off in expression_features")
    from ..behavior import Expression
    from ..expression import Expressions

    ex = Expressions(ctx.config)
    if not ex.antennas and not ex.eyes:
        return Result(FAIL, "could not open the PWM outputs", hint="check the pins in docs/WIRING.md")
    try:
        for i in range(3):
            for v in (-1.0, 1.0):
                ex.apply(Expression(antennas=(v, -v), eye_leds=1.0 if v > 0 else 0.0), now=time.monotonic() + i + v)
                time.sleep(0.5)
        ex.apply(Expression(antennas=(0.0, 0.0), eye_leds=0.5), now=time.monotonic() + 10)
    finally:
        ex.stop()
    if ctx.confirm(f"Did the {' and '.join(feats)} move/blink, both sides?", default=True):
        return Result(PASS, f"{', '.join(feats)} work")
    return Result(FAIL, f"{', '.join(feats)} did not respond", hint="check the PWM pins and the 5 V feed to the antenna servos")


# ============================================================ control loop


class _StandInPolicy:
    """Same shape as the stock policy (101 -> 3x256 -> 14), random weights, numpy."""

    def __init__(self, size=101):
        rng = np.random.default_rng(0)
        dims = [size, 256, 256, 256, 14]
        self.w = [rng.normal(0, 1 / math.sqrt(a), (a, b)).astype(np.float32) for a, b in zip(dims, dims[1:])]

    def infer(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for w in self.w[:-1]:
            x = np.tanh(x @ w)
        return x @ self.w[-1]


def check_loop_timing(ctx):
    """The walk loop's read + infer path at 50 Hz, with torque off (nothing moves)."""
    from ..commands import ZeroCommands
    from ..hwi import HWI
    from ..imu import Imu
    from ..policy import load_policy
    from ..rl_utils import nb_steps_from_config
    from ..walk import RLWalk

    bus = ctx.bus()
    if ctx.shared.get("servos_missing"):
        raise Skip("not all servos answer")
    imu_dev = ctx.imu_device()
    m = ctx.config.maixcam2
    policy, label = None, "stand-in MLP"
    if os.path.exists(m["policy_path"]):
        try:
            policy = load_policy(m["policy_backend"], m["policy_path"])
            label = f"{m['policy_backend']} {os.path.basename(m['policy_path'])}"
        except Exception as e:
            label = f"stand-in MLP ({m['policy_backend']} policy failed to load: {e})"
    if policy is None:
        policy = _StandInPolicy()
    try:
        nb = nb_steps_from_config(ctx.config)
    except Exception:
        nb = 36
    hwi = HWI(ctx.config, bus=bus)
    imu = Imu(ctx.config, sampling_freq=LOOP_HZ, dev=imu_dev)
    walker = RLWalk(ctx.config, hwi, imu, ctx.feet(), policy, ZeroCommands(), nb_steps_in_period=nb)
    seconds = float(ctx.opts.get("loop_seconds", 10))
    period = 1.0 / LOOP_HZ
    torque_off = {i: [0] for i in hwi.ids}
    loop, infer, overruns, failed = [], [], 0, 0
    next_t = time.monotonic()
    end = next_t + seconds
    try:
        while time.monotonic() < end:
            t = time.monotonic()
            obs = walker.get_obs()
            if obs is None:
                failed += 1
            else:
                ti = time.perf_counter()
                policy.infer(obs)
                infer.append(time.perf_counter() - ti)
            # same bus traffic as a goal-position write, but harmless: torque stays off
            bus.sync_write(ft.REG_TORQUE_ENABLE, 1, torque_off)
            loop.append(time.monotonic() - t)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                overruns += 1
                next_t = time.monotonic()
    finally:
        imu.stop()
    st = _stats_ms(loop)
    inf = _stats_ms(infer) if infer else {"mean_ms": float("nan")}
    data = {"policy": label, "steps": len(loop), "overruns": overruns, "failed_reads": failed, "loop": st, "infer": inf}
    summary = f"{len(loop)} steps: loop p99 {st['p99_ms']:.1f} ms of 20, policy {inf['mean_ms']:.2f} ms, {overruns} overruns, {failed} failed reads"
    details = [f"policy: {label}"]
    if failed > 0.01 * len(loop) or st["p99_ms"] > 18:
        return Result(FAIL, summary, details=details, data=data, hint="the loop cannot hold 50 Hz; look at servo.timing and imu.still")
    if failed or overruns or st["p99_ms"] > 12:
        return Result(WARN, summary, details=details, data=data, hint="little headroom; set the CPU governor to performance and retest")
    return Result(PASS, summary, details=details, data=data)


# ============================================================ registry

CHECKS = [
    Check("config.file", "config", check_config_file, "duck_config.json"),
    Check("config.offsets", "config", check_config_offsets, "joint offsets"),
    Check("config.files", "config", check_config_files, "policy, gait and calibration files"),
    Check("config.pins", "config", check_config_pins, "pin assignments"),
    Check("board.deps", "board", check_python_deps, "Python modules"),
    Check("board.system", "board", check_board_system, "MaixPy, temperature, governor, memory", board_only=True),
    Check("servo.ping", "servo", check_servo_ping, "14 servos answer"),
    Check("servo.scan", "servo", check_servo_scan, "stray or unconfigured IDs"),
    Check("servo.eeprom", "servo", check_servo_eeprom, "servo mode, return delay, limits"),
    Check("servo.power", "servo", check_servo_power, "voltage, temperature, error flags"),
    Check("servo.pose", "servo", check_servo_pose, "positions and encoder wrap"),
    Check("servo.timing", "servo", check_servo_timing, "bus read latency and errors"),
    Check("imu.chip", "imu", check_imu_chip, "BNO086 answers (product ID, reset cause)"),
    Check("imu.calibration", "imu", check_imu_calibration, "calibration saved, accuracy"),
    Check("imu.still", "imu", check_imu_still, "gravity, gyro bias, I2C errors"),
    Check("imu.upright", "imu", check_imu_upright, "upright reads +z"),
    Check("imu.axes", "imu", check_imu_axes, "tilt and turn signs", interactive=True),
    Check("feet.stable", "feet", check_feet_stable, "inputs steady"),
    Check("feet.press", "feet", check_feet_press, "left and right switch", interactive=True),
    Check("camera.frames", "camera", check_camera, "camera frames", board_only=True),
    Check("npu.policy", "npu", check_npu_policy, "walking policy on the NPU", board_only=True),
    Check("npu.yolo", "npu", check_npu_yolo, "YOLO11 detector", board_only=True),
    Check("audio.speaker", "audio", check_speaker, "speaker", interactive=True, board_only=True),
    Check("audio.mic", "audio", check_microphone, "microphone level", board_only=True),
    Check("expression.outputs", "expression", check_expression_outputs, "antennas and eye LEDs", interactive=True, board_only=True),
    Check("loop.timing", "loop", check_loop_timing, "50 Hz read + infer loop, torque off"),
]

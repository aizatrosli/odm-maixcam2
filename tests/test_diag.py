"""Diagnostics and calibration against the simulated duck (wire-level)."""

import json

import numpy as np
import pytest

from odm_maix import feetech as ft
from odm_maix.config import JOINT_NAMES, DuckConfig
from odm_maix.diag import calib
from odm_maix.diag.checks import CHECKS
from odm_maix.diag.core import FAIL, PASS, SKIP, Context, run_checks
from odm_maix.diag.simhw import SimDuck, VirtualServo
from odm_maix.hwi import HWI, INIT_POS

HW_GROUPS = {"servo", "imu", "feet", "loop"}
FAST = {"loop_seconds": 0.5, "imu_seconds": 0.3, "bus_reads": 50, "hold_seconds": 0.2, "turn_seconds": 1.2}


def run(faults=(), interactive=False, only=HW_GROUPS, config=None):
    ctx = Context(config or DuckConfig(), dry_run=True, interactive=interactive, faults=list(faults),
                  out=lambda *a: None, opts=FAST)
    try:
        return {r.name: r for r in run_checks(ctx, CHECKS, only=only)}
    finally:
        ctx.close()


def test_healthy_duck_passes_everything():
    res = run(interactive=True)
    bad = {n: (r.status, r.summary) for n, r in res.items() if r.status not in (PASS, SKIP)}
    # no IMU calibration file in a fresh config is the only expected warning
    assert set(bad) == {"imu.calibration"}, bad
    assert res["imu.axes"].status == PASS
    assert res["feet.press"].status == PASS
    assert res["loop.timing"].data["steps"] > 10


def test_guided_checks_skip_without_interactive():
    res = run()
    assert res["imu.axes"].status == SKIP
    assert res["feet.press"].status == SKIP


@pytest.mark.parametrize(
    "fault, check, hint",
    [
        ("missing:23", "servo.ping", "left_knee"),
        ("stray:1", "servo.scan", "ID 1"),
        ("lowvolt:6.2", "servo.power", "charge"),
        ("hot:24", "servo.power", "hot"),
        ("overload:13", "servo.power", "error flags"),
        ("wrap:13", "servo.pose", "wrap"),
        ("eeprom:12", "servo.eeprom", "mode"),
        ("upside_down", "imu.upright", "imu_upside_down to true"),
        ("upside_down", "imu.axes", ""),
        ("imu_selftest", "imu.chip", "self test"),
        ("swapped_feet", "feet.press", "swapped"),
        ("noisy_feet", "feet.stable", "pull-up"),
        ("flaky_bus", "servo.timing", "bus errors"),
    ],
)
def test_faults_are_caught(fault, check, hint):
    if fault == "stray:1":
        fault = "stray:1,missing:22"  # scan only runs when something is missing (or --scan)
    res = run([fault], interactive=True, only={check.split(".")[0]})
    r = res[check]
    assert r.status == FAIL, (r.summary, r.details)
    text = " ".join([r.summary, r.hint, *r.details])
    assert hint in text


def test_no_imu_skips_imu_checks():
    res = run(["no_imu"])
    assert res["imu.chip"].status == SKIP
    assert "no BNO055" in res["imu.chip"].summary


def test_config_pin_conflict():
    cfg = DuckConfig()
    cfg.maixcam2 = dict(cfg.maixcam2, right_foot_gpio="GPIOA8")
    res = run(config=cfg, only={"config.pins"})
    assert res["config.pins"].status == FAIL
    assert any("A8" in d for d in res["config.pins"].details)


def test_capture_offsets_recovers_horn_offsets():
    sim = SimDuck(["offset:22:0.3"])
    bus = ft.FeetechBus("sim", ser=sim.serial)
    got = calib.capture_offsets(bus, samples=5)
    for j in JOINT_NAMES:
        assert got[j] == pytest.approx(sim.true_offsets[j], abs=3 * calib.RAD_PER_COUNT)
    assert got["left_hip_pitch"] == pytest.approx(0.3, abs=0.01)


def test_save_offsets_keeps_other_keys(tmp_path):
    path = tmp_path / "duck_config.json"
    path.write_text(json.dumps({"imu_upside_down": True, "joints_offsets": {"left_knee": 0.1}, "maixcam2": {"servo_port": "/dev/ttyS4"}}))
    bak = calib.save_offsets(str(path), {"right_knee": -0.05})
    d = json.loads(path.read_text())
    assert d["imu_upside_down"] is True and d["maixcam2"]["servo_port"] == "/dev/ttyS4"
    assert d["joints_offsets"] == {"left_knee": 0.1, "right_knee": -0.05}
    assert json.loads(open(bak).read())["joints_offsets"] == {"left_knee": 0.1}
    calib.save_joint_limits(str(path), {"left_knee": [0.5, 2.0]})
    assert DuckConfig.load(str(path)).maixcam2["joint_limits"] == {"left_knee": [0.5, 2.0]}


def test_analyse_range():
    ok = calib.analyse_range("left_knee", 0.6, 2.0, False, 0.0)
    assert ok["problems"] == [] and ok["init_inside"]
    wrapped = calib.analyse_range("left_knee", -3.1, 3.1, True, 0.0)
    assert any("wrap" in p for p in wrapped["problems"])
    assert calib.soft_limits({"left_knee": ok, "x": wrapped}) == {"left_knee": [0.65, 1.95]}


def test_nudge_joint_moves_only_that_joint():
    sim = SimDuck()
    bus = ft.FeetechBus("sim", ser=sim.serial)
    answers = iter(["+", "++", "-", ""])
    v = calib.nudge_joint(bus, "head_yaw", 0.0, lambda p: next(answers))
    assert v == pytest.approx(0.05)
    assert not any(s.mem[ft.REG_TORQUE_ENABLE] for s in sim.servos)


def test_configure_servo_sets_id_and_settings():
    sim = SimDuck()
    fresh = VirtualServo(1, 1000)
    fresh.mem[ft.REG_RETURN_DELAY] = 250
    fresh.mem[ft.REG_MODE] = 3
    sim.serial.servos[:] = [fresh]
    bus = ft.FeetechBus("sim", ser=sim.serial)
    assert calib.find_single_servo(bus) == [1]
    got = calib.configure_servo(bus, 1, 22)
    assert got["id"] == 22 and got["mode"] == 0 and got["return_delay"] == 0
    assert fresh.mem[ft.REG_LOCK] == 1
    assert calib.center_servo(bus, 22) == 2048


def test_eeprom_write_ignored_while_locked():
    s = VirtualServo(5, 2048)
    s.write(ft.REG_ID, bytes([9]))
    assert s.sid == 5


def test_hwi_clamps_to_joint_limits():
    sim = SimDuck()
    cfg = DuckConfig()
    cfg.maixcam2 = dict(cfg.maixcam2, joint_limits={"left_knee": [1.0, 1.5]})
    cfg.joints_offsets = dict(cfg.joints_offsets, left_knee=0.1)
    hwi = HWI(cfg, bus=ft.FeetechBus("sim", ser=sim.serial))
    targets = np.array([INIT_POS[j] for j in JOINT_NAMES])
    targets[JOINT_NAMES.index("left_knee")] = 2.5
    hwi.set_position_array(targets)
    knee = sim.servo("left_knee")
    goal = ft.raw_to_rad_position(int.from_bytes(knee.mem[ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2], "little"))
    assert goal == pytest.approx(1.5 + 0.1, abs=2 * calib.RAD_PER_COUNT)
    hwi.set_position_all({"left_knee": 0.0})
    goal = ft.raw_to_rad_position(int.from_bytes(knee.mem[ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2], "little"))
    assert goal == pytest.approx(1.0 + 0.1, abs=2 * calib.RAD_PER_COUNT)


def test_imu_calibration_saves_offsets(tmp_path):
    sim = SimDuck()
    from odm_maix import bno055

    dev = bno055.BNO055(7, smbus=sim.imu_bus)
    path = tmp_path / "imu.json"
    s = calib.calibrate_imu(dev, str(path), timeout=2, poll=0.01, on_poll=lambda: sim.act("imu.calibrate_motion"), out=lambda *a: None)
    assert all(v == 3 for v in s.values())
    assert len(json.loads(path.read_text())["bno055_offsets"]) == 11


def test_diagnose_cli_exit_code(tmp_path):
    import importlib.util
    import os
    import sys

    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location("diagnose", os.path.join(here, "diagnose.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = tmp_path / "r.json"
    code = mod.main(["--dry-run", "--only", "servo.ping,servo.power", "--json", str(report), "--fault", "lowvolt:6.0"])
    assert code == 2
    assert json.loads(report.read_text())["overall"] == FAIL
    assert mod.main(["--dry-run", "--only", "servo.ping", "--json", ""]) == 0

"""The on-screen dashboard (apps/duck_dash): its C core, its servo driver
against the wire-level simulated duck, and walk.py's telemetry feed."""

import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from odm_maix.config import JOINT_NAMES, DuckConfig
from odm_maix.hwi import INIT_POS, JOINT_IDS
from odm_maix.telemetry import TelemetryWriter
from odm_maix.walk import RLWalk

from .test_walk import FakeCommands, FakeFeet, FakeHWI, FakeImu, RecordingPolicy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "apps", "duck_dash")
sys.path.insert(0, os.path.join(APP, "tools"))

needs_cc = pytest.mark.skipif(
    not (shutil.which("make") and (shutil.which("cc") or shutil.which("gcc"))), reason="needs make and a C compiler"
)


@pytest.fixture(scope="module")
def dash(tmp_path_factory):
    build = str(tmp_path_factory.mktemp("dash_build"))
    r = subprocess.run(
        ["make", "-C", APP, f"BUILD={build}", f"{build}/duck_dash", f"{build}/test_core"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    return build


def run_dash(build, *args, timeout=30):
    r = subprocess.run([os.path.join(build, "duck_dash"), *args], capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr
    return r.stdout


def parse_dump(out):
    servos = {}
    lines = {}
    for line in out.splitlines():
        if line.startswith("servo "):
            parts = line.split()
            servos[int(parts[1])] = dict(kv.split("=") for kv in parts[2:])
        else:
            key = line.split("=", 1)[0]
            lines[key] = dict(kv.split("=", 1) for kv in line.split(" ") if "=" in kv)
    return lines, servos


@needs_cc
def test_c_unit_tests(dash):
    r = subprocess.run([os.path.join(dash, "test_core")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@needs_cc
def test_reads_the_simulated_servo_bus(dash, tmp_path):
    from pty_duck import PtyServoBus

    bus = PtyServoBus(["missing:23,hot:24,overload:13,lowvolt:7.1"]).start()
    try:
        out = run_dash(
            dash, "--servo-port", bus.path, "--no-imu", "--no-feet", "--camera", "off", "--config", str(tmp_path / "none.json"),
            "--feed", str(tmp_path / "nofeed"), "--frames", "6", "--fps", "10", "--dump",
        )  # fmt: skip
    finally:
        bus.stop()
    lines, servos = parse_dump(out)
    assert lines["bus"]["bus"] == "fail"
    assert lines["bus"]["servos_found"] == "13"
    assert "missing ID 23" in out
    assert set(servos) == set(JOINT_IDS.values()) - {23}
    by_id = {v: k for k, v in JOINT_IDS.items()}
    for sid, s in servos.items():
        # SimDuck puts each horn within a few hundredths of a radian of the init pose
        assert abs(float(s["pos"]) - INIT_POS[by_id[sid]]) < 0.15, (sid, s)
    assert servos[24]["temp"] == "72"
    assert servos[13]["err"] == "32"  # overload bit
    assert lines["battery"]["battery"] == "ok"
    assert abs(float(lines["battery"]["volt"]) - 7.1) < 0.15


@needs_cc
def test_port_held_by_another_process_is_left_alone(dash, tmp_path):
    from pty_duck import PtyServoBus

    bus = PtyServoBus().start()
    holder = os.open(bus.path, os.O_RDWR | os.O_NOCTTY)  # stands in for a running walk.py
    try:
        out = run_dash(
            dash, "--servo-port", bus.path, "--no-imu", "--no-feet", "--camera", "off",
            "--feed", str(tmp_path / "nofeed"), "--frames", "3", "--dump",
        )  # fmt: skip
    finally:
        os.close(holder)
        bus.stop()
    assert "bus=warn" in out and "busy:" in out
    assert bus.packets == 0


@needs_cc
def test_missing_adapter(dash, tmp_path):
    out = run_dash(
        dash, "--servo-port", "/dev/ttyDOESNOTEXIST", "--no-imu", "--no-feet", "--camera", "off",
        "--feed", str(tmp_path / "nofeed"), "--frames", "2", "--dump",
    )  # fmt: skip
    assert "bus=fail servos_found=0" in out and "not found" in out


def make_walker():
    w = RLWalk(DuckConfig(), FakeHWI(), FakeImu(), FakeFeet(), RecordingPolicy(), FakeCommands((0.1, 0, 0.2, 0, 0, 0, 0)), 36)
    return w


class FakeIO:
    def __init__(self):
        self.reads = []

    def read(self, sid, addr, n):
        self.reads.append((sid, addr, n))
        return bytes([76, 40])


def test_telemetry_writer_rate_and_content(tmp_path):
    path = str(tmp_path / "feed")
    now = [100.0]
    tw = TelemetryWriter(path, hz=10, mode="follow", clock=lambda: now[0])
    w = make_walker()
    w.hwi.ids = [JOINT_IDS[j] for j in JOINT_NAMES]
    w.hwi.io = FakeIO()
    for _ in range(5):
        w.step(0)
        w.loop_times.append(0.006)
    assert tw.update(w)
    assert not tw.update(w, now=100.05)  # 10 Hz
    text = open(path).read()
    fields = dict(line.split(" ", 1) for line in text.strip().splitlines())
    assert fields["duck_telemetry"] == "1" and fields["mode"] == "follow"
    assert len(fields["pos"].split()) == 14 and len(fields["cmd"].split()) == 7
    assert fields["volt"] == "7.60" and fields["temp_max"] == "40"
    assert fields["acc"].split()[2] == "9.81"
    # the battery read is one 2-byte register read per second, round robin
    assert w.hwi.io.reads == [(20, 62, 2)]
    tw.update(w, now=100.5)
    assert len(w.hwi.io.reads) == 1
    tw.update(w, now=101.2)
    assert w.hwi.io.reads[-1] == (21, 62, 2)
    tw.close()
    assert not os.path.exists(path)


def test_telemetry_errors_never_stop_the_walk(tmp_path):
    w = make_walker()

    class Broken:
        def update(self, walker):
            raise RuntimeError("disk full")

        def close(self):
            pass

    w.telemetry = Broken()
    w._publish()
    assert w.telemetry is None


@needs_cc
def test_dashboard_shows_the_walk_loop_feed(dash, tmp_path):
    path = str(tmp_path / "feed")
    tw = TelemetryWriter(path, mode="interact")
    w = make_walker()
    w.backend = "npu"
    for _ in range(3):
        w.step(0)
        w.loop_times.append(0.007)
    tw.update(w)
    out = run_dash(dash, "--feed", path, "--camera", "off", "--frames", "2", "--dump", "--servo-port", "/dev/null")
    lines, servos = parse_dump(out)
    assert lines["source"]["source"] == "feed"
    assert "walk_live=1 mode=interact" in out
    assert len(servos) == 14
    assert abs(float(servos[23]["pos"]) - INIT_POS["left_knee"]) < 1e-3
    assert "backend=npu" in out


@needs_cc
def test_sim_screenshot(dash, tmp_path):
    shot = str(tmp_path / "s.ppm")
    run_dash(dash, "--sim=missing:23,walk", "--frames", "2", "--snapshot", shot)
    with open(shot, "rb") as f:
        assert f.read(15).startswith(b"P6\n640 480\n255\n")
    assert os.path.getsize(shot) == 15 + 640 * 480 * 3


# ---------------------------------------------------------------- stress tools


def write_feed(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def stress_duck(**faults):
    import random

    from duck_stress import Duck

    d = Duck(random.Random(0))
    d.step(time.time())
    for k, v in faults.items():
        setattr(d, k, v)
    return d


@needs_cc
def test_feed_with_servo_detail(dash, tmp_path):
    from duck_stress import feed_lines

    d = stress_duck()
    d.present[3] = 0  # servo 23 (left_knee)
    d.temp_override = {4: 72}  # servo 24
    d.err[12] = 0x20  # servo 13
    path = str(tmp_path / "feed")
    write_feed(path, feed_lines(d, time.time()))
    out = run_dash(dash, "--feed", path, "--camera", "off", "--frames", "2", "--dump", "--servo-port", "/dev/null")
    lines, servos = parse_dump(out)
    assert lines["source"]["source"] == "feed"
    assert lines["bus"]["bus"] == "fail" and lines["bus"]["servos_found"] == "13"
    assert "missing ID 23" in out
    assert 23 not in servos and servos[24]["temp"] == "72" and servos[13]["err"] == "32"
    assert float(servos[20]["volt"]) > 7 and "steps=" in out


@needs_cc
def test_feed_never_shows_non_finite_or_stale_values(dash, tmp_path):
    from duck_stress import feed_lines

    d = stress_duck(acc=None, feet=None)
    lines = [ln if not ln.startswith("pos ") else "pos " + " ".join(["nan"] * 14) for ln in feed_lines(d, time.time())]
    lines += ["volt inf", "temp_max 1e40", "overruns -5", "det_fps nan"]
    path = str(tmp_path / "feed")
    write_feed(path, lines)
    out = run_dash(dash, "--feed", path, "--camera", "off", "--frames", "2", "--dump", "--servo-port", "/dev/null")
    lines, servos = parse_dump(out)
    assert lines["bus"]["servos_found"] == "0" and not servos
    assert lines["battery"]["volt"] == "0.00"
    assert lines["imu"]["imu"] == "warn" and lines["imu"]["acc"] == "0.00,0.00,0.00"
    assert lines["feet"]["feet"] == "warn" and lines["feet"]["left"] == "-1"
    assert "nan" not in out and "inf" not in out


def _stress_run(dash, tmp_path, scenario, secs=2.5, *dash_args):
    path = str(tmp_path / "feed")
    gen = subprocess.Popen(
        [sys.executable, os.path.join(APP, "tools", "duck_stress.py"), "-s", scenario, "--feed", path,
         "--duration", str(secs + 1), "--seed", "1", "-q"]
    )  # fmt: skip
    try:
        time.sleep(0.4)
        out = run_dash(
            dash, "--feed", path, "--camera", "off", "--servo-port", "/dev/null", "--no-imu", "--no-feet",
            "--config", str(tmp_path / "none.json"), "--fps", "30", "--frames", str(int(secs * 30)),
            "--stats", "--dump", *dash_args,
        )  # fmt: skip
    finally:
        gen.wait(timeout=10)
    return out


@needs_cc
@pytest.mark.parametrize("scenario", ["nominal", "faults", "disconnect", "extreme", "rate", "chaos"])
def test_dashboard_survives_stress_scenarios(dash, tmp_path, scenario):
    out = _stress_run(dash, tmp_path, scenario)
    total = dict(kv.split("=") for kv in out.split("stats_total ", 1)[1].split()) if "stats_total" in out else {}
    assert float(total["sampler_hz"]) > 15  # the sampler never stalls on a bad feed
    assert int(total["frames"]) == 75
    assert "nan" not in out.lower().replace("nan ", "")


@needs_cc
def test_stress_servo_bus_faults(dash, tmp_path):
    gen = subprocess.Popen(
        [sys.executable, os.path.join(APP, "tools", "duck_stress.py"), "--sink", "bus", "-s", "faults", "--duration", "6", "--seed", "1"],
        stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        pty = gen.stdout.readline().strip()
        out = run_dash(
            dash, "--servo-port", pty, "--no-imu", "--no-feet", "--camera", "off", "--config", str(tmp_path / "none.json"),
            "--feed", str(tmp_path / "nofeed"), "--frames", "20", "--fps", "10", "--dump",
        )  # fmt: skip
    finally:
        gen.terminate()
        gen.wait(timeout=5)
    lines, servos = parse_dump(out)
    # the first fault of the cycle is "servo 23 unplugged"; bad checksums and drops never crash the reader
    assert lines["bus"]["bus"] == "fail" and "missing ID 23" in out
    assert 23 not in servos and len(servos) >= 11

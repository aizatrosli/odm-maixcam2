"""Telemetry feed for the on-screen dashboard (apps/duck_dash).

While walk.py runs it owns the servo bus, the IMU and the foot switches, so
the dashboard cannot read them itself. Instead the walk loop rewrites a small
text file in /dev/shm ten times a second, and the dashboard shows that::

    duck_telemetry 1
    time 1730771234.123        # unix time, the dashboard ignores feeds older than 1 s
    mode interact
    state tracking/follow/happy
    paused 0
    backend npu
    loop_ms 6.81 9.60          # mean, p99 over the last 50 steps
    policy_ms 0.31
    overruns 0
    cmd 0.12 0 0.3 0 0 0 0     # the 7 policy commands
    pos <14 floats>            # joint positions, rad, policy frame
    vel <14 floats>
    gyro <3> / acc <3>         # what the policy sees
    feet 1 0
    volt 7.86                  # pack voltage, one servo read per second
    temp_max 41
    bus_errors 0
    det_fps 24.5
    screen 0                   # 1 = the vision process draws on the screen

Writing it costs about 0.1 ms every 5th step, plus one 2-byte servo read per
second for the battery voltage (about 0.3 ms).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from .config import JOINT_NAMES

DEFAULT_PATH = "/dev/shm/duck_telemetry"
DASH_PID_FILE = "/dev/shm/duck_dash.pid"
REG_PRESENT_VOLTAGE = 62  # voltage (1 byte, 0.1 V) then temperature (1 byte, C)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.4g}" for v in values)


def dashboard_running(pid_file: str = DASH_PID_FILE) -> bool:
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


class TelemetryWriter:
    def __init__(self, path: str = DEFAULT_PATH, hz: float = 10.0, mode: str = "walk", screen: bool = False, clock=time.time):
        self.path = path
        self.period = 1.0 / hz
        self.mode = mode
        self.screen = screen
        self.clock = clock
        self._next = 0.0
        self._next_volt = 0.0
        self._volt_idx = 0
        self._volts = {}
        self._temps = {}
        self._bus_errors = 0
        self.enabled = os.path.isdir(os.path.dirname(path) or ".")

    # ------------------------------------------------------------------ io
    def _write(self, lines) -> None:
        if not self.enabled:
            return
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, self.path)  # atomic: the dashboard never reads half a file
        except OSError:
            self.enabled = False

    def announce(self, wait_s: float = 0.3) -> None:
        """Tell a running dashboard to let go of the camera before the detector opens it."""
        self._write(["duck_telemetry 1", f"time {self.clock():.3f}", f"mode {self.mode}", "state starting", "paused 1"])
        if wait_s and dashboard_running():
            time.sleep(wait_s)

    def close(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass

    # ------------------------------------------------------------ sampling
    def _read_battery(self, walker, now: float) -> None:
        """One servo per second, round robin: pack voltage and temperature."""
        if now < self._next_volt:
            return
        self._next_volt = now + 1.0
        hwi = getattr(walker, "hwi", None)
        io = getattr(hwi, "io", None)
        if io is None or not hasattr(io, "read"):
            return
        ids = list(hwi.ids)
        sid = ids[self._volt_idx % len(ids)]
        self._volt_idx += 1
        try:
            data = io.read(sid, REG_PRESENT_VOLTAGE, 2)
            self._volts[sid] = data[0] / 10.0
            self._temps[sid] = data[1]
        except Exception:
            self._bus_errors += 1

    def update(self, walker, now: Optional[float] = None) -> bool:
        """Call once per control step; writes at most `hz` times a second."""
        now = self.clock() if now is None else now
        if now < self._next:
            return False
        self._next = now + self.period
        self._read_battery(walker, now)

        lt = np.asarray(walker.loop_times[-50:]) * 1000 if walker.loop_times else np.zeros(1)
        it = np.asarray(walker.infer_times[-50:]) * 1000 if walker.infer_times else np.zeros(1)
        lines = [
            "duck_telemetry 1",
            f"time {now:.3f}",
            f"mode {self.mode}",
            f"paused {int(bool(walker.paused))}",
            f"backend {getattr(walker, 'backend', '') or type(walker.policy).__name__.replace('Policy', '').lower()}",
            f"loop_ms {lt.mean():.2f} {np.percentile(lt, 99):.2f}",
            f"policy_ms {it.mean():.3f}",
            f"overruns {walker.overruns}",
            f"cmd {_fmt(walker.last_commands)}",
        ]
        state = getattr(walker.commands, "state", None)
        if state:
            lines.append(f"state {state}")
        pos, vel = getattr(walker, "last_dof_pos", None), getattr(walker, "last_dof_vel", None)
        if pos is not None and len(pos) == len(JOINT_NAMES):
            lines.append(f"pos {_fmt(pos)}")
            if vel is not None:
                lines.append(f"vel {_fmt(vel)}")
        if walker.imu is not None:
            try:
                d = walker.imu.get_data()
                lines += [f"gyro {_fmt(d['gyro'])}", f"acc {_fmt(d['accelero'])}"]
            except Exception:
                pass
        if walker.feet_contacts is not None:
            try:
                lines.append(f"feet {_fmt([float(x) for x in walker.feet_contacts.get()])}")
            except Exception:
                pass
        if self._volts:
            lines.append(f"volt {np.mean(list(self._volts.values())):.2f}")
            lines.append(f"temp_max {max(self._temps.values())}")
        lines.append(f"bus_errors {self._bus_errors + getattr(walker, 'read_failures', 0)}")
        det = getattr(walker.commands, "detector", None)
        if det is not None and hasattr(det, "fps"):
            try:
                lines.append(f"det_fps {det.fps():.1f}")
            except Exception:
                pass
        lines.append(f"screen {int(self.screen)}")
        self._write(lines)
        return True

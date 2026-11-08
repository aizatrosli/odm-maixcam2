#!/usr/bin/env python3
"""Simulated duck data for stress testing the dashboard, without hardware.

Plain Python 3 standard library only, so the same file runs on a PC and on the
MaixCAM2 itself (copy it over and run it next to the installed dashboard).

It drives the dashboard the way the real duck does, through one of two sinks:

  feed (default)  rewrites walk.py's telemetry feed (/dev/shm/duck_telemetry).
                  The installed dashboard picks it up on its own, so nothing on
                  the board needs to change: the header turns to WALK LOOP and
                  every panel shows the simulated duck.
  bus             serves a Feetech servo bus on a pseudo-terminal and prints its
                  path. Start the dashboard with --servo-port <that path> to
                  push its real C servo driver through dropouts, garbage bytes,
                  bad checksums and late replies.

Scenarios (--scenario, default nominal):

  nominal     a duck walking: 14 servos following a gait, BNO086 sway and gyro,
              feet alternating, the 2S pack slowly discharging, policy steps
  faults      cycles through every fault the dashboard knows, a few seconds each
  disconnect  the feed goes stale, disappears, comes back, flaps around the
              1 s staleness limit, walk.py "starting" handovers
  extreme     NaN, inf, huge and negative values, too many / too few numbers,
              binary garbage, over-long lines and strings, clock skew
  rate        rewrites the feed as fast as it can (--hz, default 1000)
  soak        nominal with rare random faults, meant to run for hours
  chaos       a different random scenario every couple of seconds

Examples::

    python3 tools/duck_stress.py --scenario faults               # PC or board
    python3 tools/duck_stress.py --scenario chaos --duration 600
    python3 tools/duck_stress.py --sink bus --scenario faults     # prints /dev/pts/N
    ./build/duck_dash --servo-port /dev/pts/N --no-imu --no-feet --stats
"""

from __future__ import annotations

import argparse
import math
import os
import random
import select
import struct
import termios
import threading
import time
import tty

# odm_maix/hwi.py: JOINT_IDS in policy order, INIT_POS
JOINT_IDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14]
INIT_POS = [0.002, 0.053, -0.63, 1.368, -0.784, 0, 0, 0, 0, -0.003, -0.065, 0.635, 1.379, -0.796]
NJ = len(JOINT_IDS)
LEGS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
STATES = ["idle", "searching", "tracking/follow", "tracking/follow/happy", "tracking/look", "lost/scan", "greeting"]
MODES = ["walk", "follow", "interact"]
SCENARIOS = ["nominal", "faults", "disconnect", "extreme", "rate", "soak", "chaos"]

# servo status error bits (Feetech): volt, sensor, heat, amps, load
ERR_BITS = [0x01, 0x02, 0x04, 0x08, 0x20]


def fmt(values) -> str:
    return " ".join(f"{float(v):.4g}" for v in values)


# ----------------------------------------------------------------------------- the duck


class Duck:
    """A plausible walking duck. step(t) advances it; the fields are what the
    feed (and the servo bus) report."""

    def __init__(self, rng: random.Random, discharge_min: float = 30.0):
        self.rng = rng
        self.discharge_s = max(discharge_min, 0.1) * 60
        self.t0 = None
        self.pos = list(INIT_POS)
        self.vel = [0.0] * NJ
        self.load = [0.0] * NJ
        self.temp = [31.0 + (i * 7) % 6 for i in range(NJ)]
        self.svolt = [8.3] * NJ
        self.err = [0] * NJ
        self.present = [1] * NJ
        self.temp_override = {}
        self.gyro = [0.0, 0.0, 0.0]
        self.acc = [0.0, 0.0, 9.81]
        self.feet = [1, 1]
        self.volt = 8.3
        self.cmd = [0.0] * 7
        self.loop_ms = (6.8, 9.6)
        self.policy_ms = 0.31
        self.overruns = 0
        self.bus_errors = 0
        self.det_fps = 24.5
        self.mode = "walk"
        self.state = "idle"
        self.paused = 0
        self.screen = 0
        self.steps = 0

    def temp_c(self, i: int) -> float:
        return self.temp_override.get(i, self.temp[i])

    def step(self, t: float) -> None:
        if self.t0 is None:
            self.t0 = t
        el = t - self.t0
        r = self.rng
        gait = 2 * math.pi * 1.6 * el  # 1.6 Hz steps
        self.steps = int(el * 50)  # the 50 Hz policy loop
        self.cmd = [0.15 * (1 + math.sin(el * 0.1)) / 2, 0.0, 0.4 * math.sin(el * 0.07), 0.2 * math.sin(el * 0.3), 0, 0, 0]
        for i in range(NJ):
            side = -1 if i >= 9 else 1
            if i in LEGS:
                k = LEGS.index(i) % 5
                amp = [0.03, 0.06, 0.35, 0.5, 0.3][k]
                ph = gait + (math.pi if side < 0 else 0) + k * 0.4
                p = INIT_POS[i] + amp * math.sin(ph)
                v = amp * 2 * math.pi * 1.6 * math.cos(ph)
                ld = 25 * math.sin(ph + 0.5) + side * 5
            else:
                p = 0.3 * math.sin(el * 0.5 + i)
                v = 0.15 * math.cos(el * 0.5 + i)
                ld = 4 * math.sin(el + i)
            self.pos[i] = p + r.gauss(0, 0.004)
            self.vel[i] = v + r.gauss(0, 0.02)
            self.load[i] = ld + r.gauss(0, 1.5)
            # temperatures creep up towards ~45 C with load
            target = 38 + abs(ld) * 0.3
            self.temp[i] += (target - self.temp[i]) * 0.002
        # 2S pack: 8.3 V full to 6.9 V over discharge_s, sagging with load
        frac = min(el / self.discharge_s, 1.0)
        sag = 0.08 * abs(math.sin(gait))
        self.volt = 8.3 - 1.4 * frac - sag + r.gauss(0, 0.01)
        for i in range(NJ):
            self.svolt[i] = self.volt + (i % 3) * 0.05
        roll = 4 * math.sin(gait / 2) + r.gauss(0, 0.3)
        pitch = 2.5 * math.sin(gait) + 3 + r.gauss(0, 0.3)
        rr, pp = math.radians(roll), math.radians(pitch)
        self.acc = [-9.81 * math.sin(pp), 9.81 * math.sin(rr) * math.cos(pp), 9.81 * math.cos(rr) * math.cos(pp)]
        self.acc = [a + r.gauss(0, 0.15) for a in self.acc]
        self.gyro = [
            math.radians(4 * math.pi * 1.6 * math.cos(gait / 2)) + r.gauss(0, 0.01),
            math.radians(2.5 * 2 * math.pi * 1.6 * math.cos(gait)) + r.gauss(0, 0.01),
            self.cmd[2] + r.gauss(0, 0.01),
        ]
        ph = (el * 1.6) % 1.0
        self.feet = [int(ph < 0.6), int(ph > 0.4)]
        jit = r.random()
        self.loop_ms = (6.8 + 0.4 * jit, 9.6 + 2 * jit)
        self.policy_ms = 0.31 + 0.03 * r.random()
        self.det_fps = 24.5 + r.gauss(0, 0.6)
        self.state = STATES[int(el / 7) % len(STATES)]
        self.mode = MODES[int(el / 40) % len(MODES)]


# ----------------------------------------------------------------------------- feed text


def feed_lines(d: Duck, now: float, servo_detail: bool = True) -> list:
    lines = [
        "duck_telemetry 1",
        f"time {now:.3f}",
        f"mode {d.mode}",
        f"state {d.state}",
        f"paused {d.paused}",
        "backend npu",
        f"loop_ms {d.loop_ms[0]:.2f} {d.loop_ms[1]:.2f}",
        f"policy_ms {d.policy_ms:.3f}",
        f"overruns {d.overruns}",
        f"steps {d.steps}",
        f"cmd {fmt(d.cmd)}",
        f"pos {fmt(d.pos)}",
        f"vel {fmt(d.vel)}",
    ]
    if servo_detail:  # optional lines the dashboard reads when present
        lines += [
            f"present {' '.join(str(int(p)) for p in d.present)}",
            f"load {fmt(d.load)}",
            f"temp {' '.join(str(int(round(d.temp_c(i)))) for i in range(NJ))}",
            f"svolt {fmt(d.svolt)}",
            f"err {' '.join(str(int(e)) for e in d.err)}",
        ]
    if d.acc is not None:
        lines += [f"gyro {fmt(d.gyro)}", f"acc {fmt(d.acc)}"]
    if d.feet is not None:
        lines.append(f"feet {d.feet[0]} {d.feet[1]}")
    if d.volt is not None:
        lines.append(f"volt {d.volt:.2f}")
        live = [d.temp_c(i) for i in range(NJ) if d.present[i]]
        lines.append(f"temp_max {int(max(live)) if live else 0}")
    lines += [f"bus_errors {d.bus_errors}", f"det_fps {d.det_fps:.1f}", f"screen {d.screen}"]
    return lines


class FeedSink:
    def __init__(self, path: str):
        self.path = path
        self.writes = 0
        self.tmp = f"{path}.{os.getpid()}.tmp"

    def write(self, lines, atomic: bool = True, raw: bytes = None) -> None:
        data = raw if raw is not None else ("\n".join(lines) + "\n").encode("utf-8", "surrogateescape")
        if atomic:
            with open(self.tmp, "wb") as f:
                f.write(data)
            os.replace(self.tmp, self.path)
        else:  # torn writes: the reader can see half a file
            with open(self.path, "wb") as f:
                half = len(data) // 2
                f.write(data[:half])
                f.flush()
                f.write(data[half:])
        self.writes += 1

    def remove(self) -> None:
        for p in (self.path, self.tmp):
            try:
                os.remove(p)
            except OSError:
                pass


# ----------------------------------------------------------------------------- scenarios
#
# A scenario is a generator: each tick it gets (duck, now) after duck.step()
# and yields an action:  ("write", lines) / ("raw", bytes) / ("torn", lines) /
# ("skip", None) / ("remove", None).  It may change the duck in place.


def reset_faults(d: Duck) -> None:
    d.present = [1] * NJ
    d.err = [0] * NJ
    d.paused = 0
    d.screen = 0
    d.temp_override = {}


def sc_nominal(log):
    d, now = yield
    while True:
        reset_faults(d)
        d, now = yield ("write", feed_lines(d, now))


def _fault_list(rng):
    """(name, apply(duck)) pairs; each runs for a few seconds."""

    def drop(ids):
        def f(d):
            for sid in ids:
                d.present[JOINT_IDS.index(sid)] = 0

        return f

    def setv(attr, val):
        return lambda d: setattr(d, attr, val)

    def hot(sid, c):
        return lambda d: d.temp_override.__setitem__(JOINT_IDS.index(sid), c)

    def errbit(sid, bit):
        return lambda d: d.err.__setitem__(JOINT_IDS.index(sid), bit)

    def busy_bus(d):
        d.bus_errors += rng.randint(1, 30)

    return [
        ("servo 23 unplugged", drop([23])),
        ("left leg unplugged (20-24)", drop([20, 21, 22, 23, 24])),
        ("whole bus silent", drop(JOINT_IDS)),
        ("servo 24 at 58 C", hot(24, 58)),
        ("servo 12 at 72 C", hot(12, 72)),
        ("servo 13 overload bit", errbit(13, 0x20)),
        ("servo 31 every error bit", errbit(31, 0x2F)),
        ("battery 6.8 V (low)", setv("volt", 6.8)),
        ("battery 6.2 V (critical)", setv("volt", 6.2)),
        ("battery not read yet", setv("volt", None)),
        ("IMU missing", setv("acc", None)),
        ("IMU reads 0 g", setv("acc", [0.0, 0.0, 0.0])),
        ("IMU upside down", setv("acc", [0.0, 0.0, -9.81])),
        ("IMU lying on its side", setv("acc", [0.0, 9.81, 0.0])),
        ("feet not reported", setv("feet", None)),
        ("both feet in the air", setv("feet", [0, 0])),
        ("bus read errors climbing", busy_bus),
        ("loop overruns, p99 25 ms", lambda d: (setattr(d, "loop_ms", (14.0, 25.0)), setattr(d, "overruns", d.overruns + 3))),
        ("policy 12 ms (CPU fallback speed)", setv("policy_ms", 12.0)),
        ("paused", setv("paused", 1)),
        ("detector stopped", setv("det_fps", 0.0)),
    ]


def sc_faults(log, rng, hold=3.0):
    faults = _fault_list(rng)
    i, until = -1, 0.0
    d, now = yield
    while True:
        if now >= until:
            i = (i + 1) % len(faults)
            until = now + hold
            log(f"fault: {faults[i][0]}")
        reset_faults(d)
        faults[i][1](d)
        d, now = yield ("write", feed_lines(d, now))


def sc_disconnect(log, rng):
    d, now = yield
    while True:
        phase = rng.choice(["live", "stale", "gone", "flap", "starting", "old_stamp"])
        dur = rng.uniform(1.5, 4.0)
        end = now + dur
        log(f"disconnect: {phase} for {dur:.1f}s")
        last_write = now
        while now < end:
            reset_faults(d)
            if phase == "live":
                act = ("write", feed_lines(d, now))
            elif phase == "stale":  # walk.py frozen: file stays, stamp ages
                act = ("skip", None)
            elif phase == "gone":  # walk.py exited and removed it
                act = ("remove", None)
            elif phase == "flap":  # writes every 0.8-1.3 s, around the 1 s limit
                if now - last_write > rng.uniform(0.8, 1.3):
                    last_write = now
                    act = ("write", feed_lines(d, now))
                else:
                    act = ("skip", None)
            elif phase == "starting":  # walk.py announced itself, servos not on yet
                act = ("write", ["duck_telemetry 1", f"time {now - rng.uniform(0, 14):.3f}", "mode walk", "state starting", "paused 1"])
            else:  # a feed whose clock is wrong: from the past or the future
                act = ("write", feed_lines(d, now + rng.choice([-30, -1.5, 3, 30, 1e9])))
            d, now = yield act


EXTREMES = ["nan", "inf", "-inf", "1e30", "-1e30", "1e-40", "0", "-0", "3.4e38", "-999999", "65535"]


def sc_extreme(log, rng):
    d, now = yield
    kinds = [
        "nan_everywhere", "huge", "negative", "too_many", "too_few", "garbage", "long_lines",
        "long_strings", "empty", "no_header", "dup_keys", "bad_ints", "ffff_bytes",
    ]  # fmt: skip
    k, until = 0, 0.0
    while True:
        if now >= until:
            kind = kinds[k % len(kinds)]
            k += 1
            until = now + 1.5
            log(f"extreme: {kind}")
        lines = feed_lines(d, now)
        if kind == "nan_everywhere":
            lines = [ln.split(" ", 1)[0] + " " + " ".join(rng.choice(EXTREMES[:3]) for _ in range(14)) if ln.split(" ", 1)[0] not in ("duck_telemetry", "time", "mode", "state", "backend") else ln for ln in lines]
        elif kind == "huge":
            lines = [ln.split(" ", 1)[0] + " " + " ".join(rng.choice(EXTREMES) for _ in range(14)) if ln.split(" ", 1)[0] not in ("duck_telemetry", "time", "mode", "state", "backend") else ln for ln in lines]
        elif kind == "negative":
            lines += ["volt -7.4", "temp_max -40", "overruns -5", "bus_errors -1", "det_fps -3", "temp " + " ".join(["-300"] * 14), "svolt " + " ".join(["-8"] * 14), "loop_ms -1 -1", "policy_ms -0.3"]
        elif kind == "too_many":
            lines = [ln + " 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" if ln.startswith(("pos", "vel", "acc", "gyro", "cmd", "feet", "load", "temp", "err", "present")) else ln for ln in lines]
        elif kind == "too_few":
            lines = [" ".join(ln.split(" ")[:3]) if ln.startswith(("pos", "vel", "acc", "gyro", "cmd", "feet", "load", "temp ", "err", "present", "svolt", "loop_ms")) else ln for ln in lines]
        elif kind == "garbage":
            junk = bytes(rng.randrange(256) for _ in range(rng.randint(10, 3000)))
            d, now = yield ("raw", ("\n".join(lines[:2]) + "\n").encode() + junk + ("\n" + "\n".join(lines[2:])).encode())
            continue
        elif kind == "long_lines":
            lines = [ln + " " + " ".join(["123.456"] * 200) if ln.startswith(("pos", "state")) else ln for ln in lines]
            lines.insert(3, "x" * 5000)
        elif kind == "long_strings":
            lines = [ln for ln in lines if not ln.startswith(("mode", "state", "backend"))]
            lines += ["mode " + "M" * 300, "state " + "tracking/" * 40, "backend " + "B" * 100]
        elif kind == "empty":
            d, now = yield ("raw", b"")
            continue
        elif kind == "no_header":
            lines = lines[1:]
        elif kind == "dup_keys":
            lines = lines + lines[::-1]
        elif kind == "bad_ints":
            lines += ["paused 99999999999999999999", "overruns 1e40", "screen -7", "bus_errors abc", "err " + " ".join(["4294967295"] * 14)]
        elif kind == "ffff_bytes":
            d, now = yield ("raw", b"duck_telemetry 1\n" + f"time {now:.3f}\n".encode() + b"\xff" * 2000 + b"\x00\n" + b"mode \xc3\x28\xa0\xa1\n")
            continue
        d, now = yield ("write", lines)


def sc_soak(log, rng):
    faults = _fault_list(rng)
    active, until = None, 0.0
    d, now = yield
    while True:
        if active is None and rng.random() < 0.002:
            active = rng.choice(faults)
            until = now + rng.uniform(1, 6)
            log(f"soak fault: {active[0]}")
        if active and now >= until:
            active = None
        reset_faults(d)
        if active:
            active[1](d)
        d, now = yield ("write", feed_lines(d, now))


def sc_chaos(log, rng):
    makers = {
        "nominal": lambda: sc_nominal(log),
        "faults": lambda: sc_faults(log, rng, hold=0.7),
        "disconnect": lambda: sc_disconnect(log, rng),
        "extreme": lambda: sc_extreme(log, rng),
    }
    d, now = yield
    while True:
        name = rng.choice(list(makers))
        log(f"chaos: {name}")
        g = makers[name]()
        next(g)
        end = now + rng.uniform(1, 3)
        while now < end:
            d, now = yield g.send((d, now))


def make_scenario(name, log, rng):
    if name in ("nominal", "rate"):
        g = sc_nominal(log)
    elif name == "faults":
        g = sc_faults(log, rng)
    elif name == "disconnect":
        g = sc_disconnect(log, rng)
    elif name == "extreme":
        g = sc_extreme(log, rng)
    elif name == "soak":
        g = sc_soak(log, rng)
    elif name == "chaos":
        g = sc_chaos(log, rng)
    else:
        raise SystemExit(f"unknown scenario {name}")
    next(g)

    return lambda d, now: g.send((d, now))


# ----------------------------------------------------------------------------- servo bus on a pty


def _checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def status_packet(sid: int, err: int, params: bytes = b"") -> bytes:
    body = bytes([sid, len(params) + 2, err]) + params
    return b"\xff\xff" + body + bytes([_checksum(body)])


def _sign_mag(v: int, bit: int) -> int:
    return (abs(v) & ((1 << bit) - 1)) | ((1 << bit) if v < 0 else 0)


def servo_regs(d: Duck, i: int) -> bytes:
    """Registers 56..63: position, velocity, load, voltage, temperature."""
    pos = int(round((d.pos[i] + math.pi) * 4096 / (2 * math.pi))) & 0x7FFF
    vel = _sign_mag(int(round(d.vel[i] * 4095 / (2 * math.pi))), 15)
    load = _sign_mag(int(round(max(-100, min(100, d.load[i])) * 10)), 10)
    volt = max(0, min(255, int(round((d.svolt[i] if d.volt is not None else 0) * 10))))
    temp = max(0, min(255, int(round(d.temp_c(i)))))
    return struct.pack("<HHHBB", pos, vel, load, volt, temp)


class PtyBus:
    """Answers Feetech PING and READ packets for the 14 servos of `duck`.
    `chaos` maps fault name -> probability per reply:
      drop (no reply), late (reply after 40 ms), garbage (random bytes),
      badsum (checksum off by one), partial (half a packet), wrongid (another
      servo's reply), noise (junk before the reply)."""

    def __init__(self, duck: Duck, rng: random.Random, lock: threading.Lock):
        self.duck, self.rng, self.lock = duck, rng, lock
        self.chaos = {}
        self.master, slave = os.openpty()
        tty.setraw(slave)
        attrs = termios.tcgetattr(slave)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave, termios.TCSANOW, attrs)
        self.path = os.ttyname(slave)
        os.close(slave)  # the dashboard refuses a port someone else holds open
        self.packets = 0
        self.replies = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
        os.close(self.master)

    def _reply(self, pkt: bytes) -> bytes:
        sid, inst, params = pkt[2], pkt[4], pkt[5:-1]
        d = self.duck
        if sid not in JOINT_IDS:
            return b""
        i = JOINT_IDS.index(sid)
        with self.lock:
            if not d.present[i]:
                return b""
            err = d.err[i]
            if inst == 0x01:
                return status_packet(sid, err)
            if inst == 0x02 and len(params) == 2:
                addr, n = params
                mem = bytearray(128)
                mem[56:64] = servo_regs(d, i)
                return status_packet(sid, err, bytes(mem[addr : addr + n]))
        return b""

    def _mangle(self, sid: int, reply: bytes) -> bytes:
        r, c = self.rng, self.chaos
        if not reply:
            return reply
        if r.random() < c.get("drop", 0):
            return b""
        if r.random() < c.get("late", 0):
            self._stop.wait(0.04)
        if r.random() < c.get("garbage", 0):
            return bytes(r.randrange(256) for _ in range(r.randint(1, 40)))
        if r.random() < c.get("badsum", 0):
            return reply[:-1] + bytes([(reply[-1] + 1) & 0xFF])
        if r.random() < c.get("partial", 0):
            return reply[: len(reply) // 2]
        if r.random() < c.get("wrongid", 0):
            other = r.choice([x for x in JOINT_IDS if x != sid])
            body = bytes([other]) + reply[3:-1]
            return b"\xff\xff" + body + bytes([_checksum(body)])
        if r.random() < c.get("noise", 0):
            return bytes(r.randrange(256) for _ in range(r.randint(1, 8))) + reply
        return reply

    def _serve(self):
        buf = b""
        while not self._stop.is_set():
            rd, _, _ = select.select([self.master], [], [], 0.05)
            if not rd:
                continue
            try:
                buf += os.read(self.master, 4096)
            except OSError:  # EIO while nobody has the slave open
                self._stop.wait(0.02)
                continue
            while True:
                i = buf.find(b"\xff\xff")
                if i < 0:
                    buf = b""
                    break
                buf = buf[i:]
                if len(buf) < 4 or len(buf) < 4 + buf[3]:
                    break
                n = 4 + buf[3]
                pkt, buf = buf[:n], buf[n:]
                self.packets += 1
                out = self._mangle(pkt[2], self._reply(pkt))
                if out:
                    self.replies += 1
                    try:
                        os.write(self.master, out)
                    except OSError:
                        pass


BUS_CHAOS = {
    "faults": {"drop": 0.02, "badsum": 0.01, "noise": 0.01},
    "extreme": {"drop": 0.1, "late": 0.05, "garbage": 0.05, "badsum": 0.05, "partial": 0.05, "wrongid": 0.05, "noise": 0.05},
    "chaos": {"drop": 0.05, "late": 0.02, "garbage": 0.03, "badsum": 0.03, "partial": 0.03, "wrongid": 0.03, "noise": 0.03},
    "soak": {"drop": 0.002, "badsum": 0.001},
}


# ----------------------------------------------------------------------------- main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", "-s", default="nominal", choices=SCENARIOS)
    p.add_argument("--sink", default="feed", choices=["feed", "bus"])
    p.add_argument("--feed", default="/dev/shm/duck_telemetry", help="feed path (default /dev/shm/duck_telemetry)")
    p.add_argument("--hz", type=float, default=None, help="updates per second (default 10; 1000 for --scenario rate)")
    p.add_argument("--torn", action="store_true", help="write the feed in place, not atomically (readers can see half a file)")
    p.add_argument("--duration", type=float, default=0, help="seconds to run, 0 = until Ctrl-C")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--discharge-min", type=float, default=30, help="minutes for the pack to go from full to 6.9 V")
    p.add_argument("--no-servo-detail", action="store_true", help="only what walk.py itself publishes (no per-servo load/temp/volt)")
    p.add_argument("--quiet", "-q", action="store_true")
    args = p.parse_args(argv)

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    hz = args.hz or (1000.0 if args.scenario == "rate" else 10.0)
    t_start = time.time()

    def log(msg):
        if not args.quiet:
            print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    duck = Duck(rng, args.discharge_min)
    lock = threading.Lock()
    step = make_scenario(args.scenario, log, rng)

    bus = sink = None
    if args.sink == "bus":
        bus = PtyBus(duck, rng, lock).start()
        bus.chaos = BUS_CHAOS.get(args.scenario, {})
        print(bus.path, flush=True)
        log(f"servo bus on {bus.path}, scenario {args.scenario}, seed {seed}; run duck_dash --servo-port {bus.path}")
    else:
        if not os.path.isdir(os.path.dirname(args.feed) or "."):
            raise SystemExit(f"{os.path.dirname(args.feed)} does not exist")
        sink = FeedSink(args.feed)
        log(f"feed {args.feed} at {hz:g} Hz, scenario {args.scenario}, seed {seed}")

    period = 1.0 / hz
    nxt = time.monotonic()
    ticks = 0
    last_report = time.monotonic()
    try:
        while args.duration <= 0 or time.time() - t_start < args.duration:
            now = time.time()
            with lock:
                duck.step(now)
                act = step(duck, now)
                if args.no_servo_detail and act[0] in ("write", "torn"):
                    keep = ("present ", "load ", "temp ", "svolt ", "err ")
                    act = (act[0], [ln for ln in act[1] if not ln.startswith(keep)])
            ticks += 1
            if sink:
                kind, payload = act
                if kind == "write":
                    sink.write(payload, atomic=not args.torn)
                elif kind == "raw":
                    sink.write(None, atomic=not args.torn, raw=payload)
                elif kind == "remove":
                    sink.remove()
            if time.monotonic() - last_report > 60:
                last_report = time.monotonic()
                done = sink.writes if sink else bus.packets
                log(f"{ticks} ticks, {done} {'feed writes' if sink else 'bus packets'}, battery {duck.volt or 0:.2f} V")
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if sink:
            sink.remove()
            log(f"done: {sink.writes} feed writes in {time.time() - t_start:.1f}s")
        if bus:
            bus.stop()
            log(f"done: {bus.packets} bus packets, {bus.replies} replies")


if __name__ == "__main__":
    main()

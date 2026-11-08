"""A virtual Open Duck Mini for dry runs: servo bus, BNO08x IMU and foot switches.

The simulator works at the wire level (Feetech packets, BNO08x SHTP packets),
so a dry run goes through the real drivers in ``odm_maix.feetech`` and
``odm_maix.bno08x``, not through mocks of them. Faults can be injected to see
what each check reports when something is wrong::

    missing:23        servo 23 does not answer
    stray:1           an unconfigured servo sits on the bus with ID 1
    lowvolt:6.4       battery at 6.4 V
    hot:24            servo 24 reports 72 C
    overload:13       servo 13 sets the overload bit in its status
    wrap:13           servo 13's horn is mounted so the joint crosses the encoder wrap
    offset:22:0.3     servo 22's horn is 0.3 rad off
    eeprom:12         servo 12 is in wheel mode with a 20 us return delay
    slow_bus          every bus transaction takes 6 ms
    flaky_bus         1 in 20 bus reads times out
    upside_down       the BNO08x is mounted upside down but the config says upright
    no_imu            no BNO08x on the I2C bus
    imu_watchdog      the BNO08x says its last reset came from its watchdog
    swapped_feet      left and right foot switches are swapped
    noisy_feet        the foot inputs float (missing pull-ups)
"""

from __future__ import annotations

import math
import random
import struct
import time
from typing import Dict, List, Optional

import numpy as np

from .. import bno08x as bno
from .. import feetech as ft
from ..config import JOINT_NAMES
from ..hwi import INIT_POS, JOINT_IDS


def parse_faults(specs) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for spec in specs or []:
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            name, *args = item.split(":")
            out.setdefault(name, []).append(args)
    return out


class VirtualServo:
    def __init__(self, sid: int, raw_pos: int, voltage: float = 7.6, temp: int = 31):
        self.mem = bytearray(128)
        m = self.mem
        m[ft.REG_ID] = sid
        m[ft.REG_BAUD_RATE] = 0
        m[ft.REG_RETURN_DELAY] = 0
        m[ft.REG_MIN_ANGLE_LIMIT : ft.REG_MIN_ANGLE_LIMIT + 2] = struct.pack("<H", 0)
        m[ft.REG_MAX_ANGLE_LIMIT : ft.REG_MAX_ANGLE_LIMIT + 2] = struct.pack("<H", 4095)
        m[ft.REG_MAX_TEMPERATURE] = 70
        m[ft.REG_MAX_VOLTAGE] = 80
        m[ft.REG_MIN_VOLTAGE] = 40
        m[ft.REG_MAX_TORQUE : ft.REG_MAX_TORQUE + 2] = struct.pack("<H", 1000)
        m[ft.REG_P_COEFF] = 32
        m[ft.REG_MODE] = 0
        m[ft.REG_TORQUE_LIMIT : ft.REG_TORQUE_LIMIT + 2] = struct.pack("<H", 1000)
        m[ft.REG_LOCK] = 1
        m[ft.REG_PRESENT_VOLTAGE] = int(round(voltage * 10))
        m[ft.REG_PRESENT_TEMPERATURE] = temp
        self.error = 0
        self.hand_raw = raw_pos  # where the joint is when nobody drives it
        self._set_present(raw_pos)

    @property
    def sid(self) -> int:
        return self.mem[ft.REG_ID]

    def _set_present(self, raw: int) -> None:
        raw = int(raw) % 4096
        self.mem[ft.REG_PRESENT_POSITION : ft.REG_PRESENT_POSITION + 2] = struct.pack("<H", raw)

    def present_raw(self) -> int:
        return struct.unpack("<H", bytes(self.mem[ft.REG_PRESENT_POSITION : ft.REG_PRESENT_POSITION + 2]))[0]

    def update(self, noise: int = 1) -> None:
        if self.mem[ft.REG_TORQUE_ENABLE]:
            goal = struct.unpack("<H", bytes(self.mem[ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2]))[0]
            self._set_present(goal)
        else:
            self._set_present(self.hand_raw + random.randint(-noise, noise))

    def read(self, addr: int, n: int) -> bytes:
        self.update()
        return bytes(self.mem[addr : addr + n])

    def write(self, addr: int, data: bytes) -> None:
        if addr < ft.REG_TORQUE_ENABLE and addr != ft.REG_LOCK and self.mem[ft.REG_LOCK]:
            return  # EEPROM locked: the real servo ignores the write
        if addr == ft.REG_TORQUE_ENABLE and data and data[0] and not self.mem[ft.REG_TORQUE_ENABLE]:
            # enabling torque holds the current position
            self.mem[ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2] = struct.pack("<H", self.present_raw())
        self.mem[addr : addr + len(data)] = data


def _status(sid: int, error: int, params: bytes = b"") -> bytes:
    body = [sid, len(params) + 2, error, *params]
    return bytes([0xFF, 0xFF, *body, ft.checksum(body)])


class VirtualServoSerial:
    """Serial-port stand-in that answers Feetech packets for a set of servos."""

    def __init__(self, servos: List[VirtualServo], latency_s: float = 0.0, drop_rate: float = 0.0):
        self.servos = servos
        self.latency_s = latency_s
        self.drop_rate = drop_rate
        self._rx = b""
        self.timeout = 0.02

    def by_id(self, sid: int) -> List[VirtualServo]:
        return [s for s in self.servos if s.sid == sid]

    # pyserial surface ----------------------------------------------------
    def reset_input_buffer(self) -> None:
        self._rx = b""

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self, n: int) -> bytes:
        if self.latency_s:
            time.sleep(self.latency_s)
        out, self._rx = self._rx[:n], self._rx[n:]
        return out

    def write(self, pkt: bytes) -> int:
        if len(pkt) < 6 or pkt[:2] != b"\xff\xff":
            return len(pkt)
        sid, length, inst = pkt[2], pkt[3], pkt[4]
        params = pkt[5 : 3 + length]
        if ft.checksum(pkt[2 : 3 + length]) != pkt[3 + length]:
            return len(pkt)
        if self.drop_rate and random.random() < self.drop_rate:
            return len(pkt)
        if inst == ft.INST_PING:
            for s in self.by_id(sid):
                self._rx += _status(s.sid, s.error)
        elif inst == ft.INST_READ:
            addr, n = params[0], params[1]
            for s in self.by_id(sid):
                self._rx += _status(s.sid, s.error, s.read(addr, n))
        elif inst == ft.INST_WRITE:
            addr, data = params[0], bytes(params[1:])
            targets = self.servos if sid == ft.BROADCAST_ID else self.by_id(sid)
            for s in list(targets):
                old = s.sid
                s.write(addr, data)
                if sid != ft.BROADCAST_ID:
                    self._rx += _status(old, s.error)
        elif inst == ft.INST_SYNC_WRITE:
            addr, n = params[0], params[1]
            body = params[2:]
            for i in range(0, len(body), n + 1):
                for s in self.by_id(body[i]):
                    s.write(addr, bytes(body[i + 1 : i + 1 + n]))
        elif inst == ft.INST_SYNC_READ:
            addr, n = params[0], params[1]
            for i in params[2:]:
                for s in self.by_id(i):
                    self._rx += _status(s.sid, s.error, s.read(addr, n))
        return len(pkt)


def _rot_xy(roll: float, pitch: float) -> np.ndarray:
    """Body orientation in the world (roll about x, then pitch about y)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


class VirtualBNO08xBus:
    """I2C stand-in with a BNO08x (SH-2 firmware, SHTP packets) behind it.

    Body frame: x forward, y left, z up. The chip is mounted so that the
    runtime's REMAP_UPRIGHT (or REMAP_UPSIDE_DOWN) turns its sensor axes into
    body axes. ``mounting`` says how it is really mounted. Reads follow the
    SHTP rules: every read starts with a 4-byte header, and a packet longer
    than the read continues in the next one.
    """

    def __init__(self, address: int = bno.ADDRESS_B, mounting: str = "upright", reset_cause: int = 1, seed: int = 0):
        self.address = address
        self.mounting = mounting
        self.reset_cause = reset_cause
        self.roll = 0.0
        self.pitch = 0.0
        self.gyro_body = np.zeros(3)
        self.gyro_bias = np.array([0.002, -0.001, 0.0015])
        self._motion_until = 0.0
        self._rng = np.random.default_rng(seed)
        self.calibrated = False
        self.dcd_saved = False
        self.me_cal = (0, 1, 1)  # accel, gyro, mag dynamic calibration (SH-2 defaults)
        self.enabled = {}  # report id -> interval s
        self._next_due = {}
        self._out = []  # queued (channel, payload)
        self._cur = None  # [channel, payload, offset]
        self._seq = [0] * 6
        self._t0 = time.monotonic()
        self._boot()

    # physics ---------------------------------------------------------------
    def set_pose(self, roll: float = 0.0, pitch: float = 0.0) -> None:
        self.roll, self.pitch = roll, pitch

    def spin(self, rate_z: float, seconds: float) -> None:
        self.gyro_body = np.array([0.0, 0.0, rate_z])
        self._motion_until = time.monotonic() + seconds

    def _body_to_sensor(self, v: np.ndarray) -> np.ndarray:
        b = v
        if self.mounting == "upside_down":
            return np.array([-b[1], -b[0], -b[2]])
        return np.array([b[1], -b[0], b[2]])

    def _vectors(self):
        acc_body = _rot_xy(self.roll, self.pitch).T @ np.array([0.0, 0.0, 9.81])
        acc_body = acc_body + self._rng.normal(0, 0.03, 3)
        if time.monotonic() > self._motion_until:
            self.gyro_body = np.zeros(3)
        gyro_body = self.gyro_body + self.gyro_bias + self._rng.normal(0, 0.002, 3)
        return self._body_to_sensor(acc_body), self._body_to_sensor(gyro_body)

    # SH-2 behaviour ------------------------------------------------------------
    def _boot(self) -> None:
        self.enabled.clear()
        self._out = [(bno.CH_COMMAND, bytes(range(40)) * 7), (bno.CH_EXECUTABLE, bytes([bno.EXE_RESET]))]
        self._cur = None

    def _accuracy(self, rid: int) -> int:
        if self.calibrated:
            return 3
        return {bno.REPORT_ACCELEROMETER: 1, bno.REPORT_GYROSCOPE: 3, bno.REPORT_MAGNETOMETER: 0}.get(rid, 0)

    def _report(self, rid: int, vec, scale: float) -> bytes:
        raw = [int(max(-32768, min(32767, round(v / scale)))) for v in vec]
        return struct.pack("<BBBB3h", rid, 0, self._accuracy(rid), 0, *raw)

    def _due_reports(self) -> None:
        now = time.monotonic()
        due = [rid for rid, t in self._next_due.items() if rid in self.enabled and now >= t]
        if not due or self._out or self._cur:
            return
        acc, gyro = self._vectors()
        body = bytearray(struct.pack("<BI", bno.REPORT_BASE_TIMESTAMP, 0))
        for rid in sorted(due):
            if rid == bno.REPORT_ACCELEROMETER:
                body += self._report(rid, acc, bno.ACC_SCALE)
            elif rid == bno.REPORT_GYROSCOPE:
                body += self._report(rid, gyro, bno.GYRO_SCALE)
            elif rid == bno.REPORT_MAGNETOMETER:
                body += self._report(rid, [20.0, 0.0, -40.0], bno.MAG_SCALE)
            self._next_due[rid] = now + self.enabled[rid]
        self._out.append((bno.CH_REPORTS, bytes(body)))

    def _command_response(self, command: int, status: int, extra=()) -> None:
        r = bytes([bno.REPORT_COMMAND_RESPONSE, 0, command, 0, 0, status] + list(extra))
        self._out.append((bno.CH_CONTROL, r + bytes(16 - len(r))))

    def _control(self, payload: bytes) -> None:
        rid = payload[0]
        if rid == bno.REPORT_PRODUCT_ID_REQUEST:
            entry = struct.pack("<BBBBIIH2x", bno.REPORT_PRODUCT_ID_RESPONSE, self.reset_cause, 3, 12, 10004563, 2, 7)
            self._out.append((bno.CH_CONTROL, entry * 4))
        elif rid == bno.REPORT_SET_FEATURE:
            _, sensor, _, _, interval_us, _, _ = struct.unpack("<BBBHIII", payload[:17])
            if interval_us:
                self.enabled[sensor] = interval_us / 1e6
                self._next_due[sensor] = time.monotonic()
            else:
                self.enabled.pop(sensor, None)
            self._out.append((bno.CH_CONTROL, bytes([bno.REPORT_GET_FEATURE_RESPONSE, sensor]) + payload[2:17]))
        elif rid == bno.REPORT_COMMAND_REQUEST:
            command = payload[2]
            if command == bno.CMD_ME_CALIBRATION and payload[6] == bno.ME_CAL_CONFIGURE:
                self.me_cal = tuple(payload[3:6])
                self._command_response(command, 0, payload[3:6])
            elif command == bno.CMD_SAVE_DCD:
                self.dcd_saved = True
                self._command_response(command, 0)

    # I2C surface (what bno08x.LinuxI2C offers) -----------------------------------
    def _check(self, addr: int) -> None:
        if addr != self.address:
            raise OSError(121, "Remote I/O error")

    def write(self, addr: int, data: bytes) -> None:
        self._check(addr)
        length, channel, _, _ = bno.parse_header(data)
        payload = bytes(data[4:length])
        if channel == bno.CH_EXECUTABLE and payload[:1] == bytes([bno.EXE_RESET]):
            self._boot()
        elif channel == bno.CH_CONTROL and payload:
            self._control(payload)

    def read(self, addr: int, n: int) -> bytes:
        self._check(addr)
        if self._cur is None:
            self._due_reports()
            if not self._out:
                return bytes(n)
            ch, payload = self._out.pop(0)
            self._cur = [ch, payload, 0, False]
        ch, payload, off, cont = self._cur
        left = len(payload) - off
        take = min(n - 4, left)
        out = bno.header(left + 4, ch, self._seq[ch], continuation=cont) + payload[off : off + take]
        self._cur[2] += take
        self._cur[3] = True
        if self._cur[2] >= len(payload) and take > 0 or not payload:
            self._seq[ch] = (self._seq[ch] + 1) & 0xFF
            self._cur = None
        return out + bytes(n - len(out))

    def close(self) -> None:
        pass


class VirtualFeet:
    def __init__(self, swapped: bool = False, noisy: bool = False):
        self.down = [True, True]
        self.swapped = swapped
        self.noisy = noisy

    def get(self):
        v = list(self.down)
        if self.swapped:
            v = v[::-1]
        if self.noisy:
            v = [random.random() < 0.5 for _ in v]
        return v

    def stop(self) -> None:
        pass


class SimDuck:
    """The whole virtual robot, plus the scripted 'person' for guided steps."""

    def __init__(self, faults=None, config=None, seed: int = 0):
        random.seed(seed)
        self.faults = parse_faults(faults) if not isinstance(faults, dict) else faults
        f = self.faults
        rng = np.random.default_rng(seed)
        # true horn misalignment, small like a real build
        self.true_offsets = {j: float(rng.normal(0, 0.03)) for j in JOINT_NAMES}
        for args in f.get("offset", []):
            joint = _joint_for(args[0])
            self.true_offsets[joint] = float(args[1])
        for args in f.get("wrap", []):
            joint = _joint_for(args[0])
            # put the joint ~0.2 rad from the encoder wrap at the init pose
            self.true_offsets[joint] = math.pi - 0.2 - INIT_POS[joint]
        volt = float(f["lowvolt"][0][0]) if "lowvolt" in f else 7.8
        missing = {int(a[0]) for a in f.get("missing", [])}
        self.servos: List[VirtualServo] = []
        for j in JOINT_NAMES:
            sid = JOINT_IDS[j]
            if sid in missing:
                continue
            raw = ft.rad_to_raw_position(INIT_POS[j] + self.true_offsets[j])
            s = VirtualServo(sid, raw, voltage=volt + rng.normal(0, 0.03), temp=int(30 + rng.integers(0, 4)))
            self.servos.append(s)
        for args in f.get("stray", []):
            self.servos.append(VirtualServo(int(args[0]), 2048, voltage=volt))
        for args in f.get("hot", []):
            for s in self.servos:
                if s.sid == int(args[0]):
                    s.mem[ft.REG_PRESENT_TEMPERATURE] = 72
        for args in f.get("overload", []):
            for s in self.servos:
                if s.sid == int(args[0]):
                    s.error = 0x20
        for args in f.get("eeprom", []):
            for s in self.servos:
                if s.sid == int(args[0]):
                    s.mem[ft.REG_MODE] = 1
                    s.mem[ft.REG_RETURN_DELAY] = 10
        self.serial = VirtualServoSerial(
            self.servos,
            latency_s=0.006 if "slow_bus" in f else 0.0,
            drop_rate=0.05 if "flaky_bus" in f else 0.0,
        )
        self.imu_bus: Optional[VirtualBNO08xBus] = None
        if "no_imu" not in f:
            self.imu_bus = VirtualBNO08xBus(
                mounting="upside_down" if "upside_down" in f else "upright",
                reset_cause=3 if "imu_watchdog" in f else 1,
            )
        self.feet = VirtualFeet(swapped="swapped_feet" in f, noisy="noisy_feet" in f)

    def servo(self, joint: str) -> Optional[VirtualServo]:
        sid = JOINT_IDS[joint]
        return next((s for s in self.servos if s.sid == sid), None)

    def pose_hands(self, offsets_from_init: Dict[str, float]) -> None:
        """Move joints by hand (torque off), radians from the init pose."""
        for j, d in offsets_from_init.items():
            s = self.servo(j)
            if s is not None:
                s.hand_raw = ft.rad_to_raw_position(INIT_POS[j] + self.true_offsets[j] + d) % 4096

    # scripted person for guided steps -------------------------------------------
    def act(self, key: str) -> None:
        imu = self.imu_bus
        if key == "pose.init":
            self.pose_hands({j: 0.0 for j in JOINT_NAMES})
        elif key == "imu.upright" and imu:
            imu.set_pose(0.0, 0.0)
            imu.gyro_body = np.zeros(3)
        elif key == "imu.nose_down" and imu:
            imu.set_pose(0.0, math.radians(30))
        elif key == "imu.right_down" and imu:
            imu.set_pose(math.radians(30), 0.0)  # +roll = right side down
        elif key == "imu.turn_left" and imu:
            imu.set_pose(0.0, 0.0)
            imu.spin(0.9, 2.0)
        elif key == "imu.calibrate_motion" and imu:
            imu.calibrated = True
        elif key == "feet.both_up":
            self.feet.down = [False, False]
        elif key == "feet.left":
            self.feet.down = [True, False]
        elif key == "feet.right":
            self.feet.down = [False, True]
        elif key == "feet.both_down":
            self.feet.down = [True, True]
        elif key.startswith("range."):
            joint = key.split(".", 1)[1]
            self._sweep(joint)

    def _sweep(self, joint: str) -> None:
        """Move one joint through +-0.6 rad in the background while the range is sampled."""
        s = self.servo(joint)
        if s is None:
            return
        center = s.hand_raw
        start = time.monotonic()

        def hand():
            t = time.monotonic() - start
            return center + int(390 * math.sin(2 * math.pi * t / 1.5))

        s.update = _sweeping_update(s, hand)


def _sweeping_update(servo, hand):
    def update(noise: int = 1):
        if servo.mem[ft.REG_TORQUE_ENABLE]:
            VirtualServo.update(servo, noise)
        else:
            servo._set_present(hand())

    return update


def _joint_for(token: str) -> str:
    if token in JOINT_IDS:
        return token
    sid = int(token)
    for j, i in JOINT_IDS.items():
        if i == sid:
            return j
    raise ValueError(f"no joint with servo id {sid}")

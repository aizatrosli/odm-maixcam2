"""BNO08x (BNO085 / BNO086) driver over Linux I2C, no extra packages.

The BNO08x does not have a register map like the BNO055. It runs CEVA's SH-2
firmware and talks SHTP: length-prefixed packets on numbered channels. The
host asks for sensor reports (Set Feature) and the chip then streams them.

This driver needs only ``/dev/i2c-N`` (``fcntl`` + ``os.read``/``os.write``,
one I2C transaction per call), so it runs on the MaixCAM2's stock Python
without Adafruit Blinka, CircuitPython or smbus2.

Units and frame are the same as the BNO055 driver it replaces:

* gyro in rad/s from the *calibrated gyroscope* report (0x02, Q9)
* acceleration in m/s^2, gravity included, from the *accelerometer* report
  (0x01, Q8)
* the axis remap that the BNO055 did on the chip is done here in software,
  with the same (axis, axis, axis, sign, sign, sign) tuples.

Calibration (DCD) lives in the chip's own flash. ``begin_calibration`` and
``save_calibration`` drive SH-2's motion-engine calibration commands.

Protocol references: CEVA "SH-2 Reference Manual" (1000-3625), "SH-2 SHTP
Reference Manual" (1000-3535) and the BNO080/BNO085/BNO086 datasheets.
"""

from __future__ import annotations

import json
import os
import struct
import time
from typing import Dict, List, Optional, Sequence, Tuple

ADDRESS_A = 0x4A  # SA0 low  (Adafruit BNO085 default)
ADDRESS_B = 0x4B  # SA0 high (SparkFun BNO086 default)
ADDRESSES = (ADDRESS_A, ADDRESS_B)

# SHTP channels
CH_COMMAND = 0
CH_EXECUTABLE = 1
CH_CONTROL = 2
CH_REPORTS = 3
CH_WAKE_REPORTS = 4
CH_GYRO_RV = 5

# executable channel
EXE_RESET = 0x01  # host -> chip: reset; chip -> host: reset complete

# control channel report IDs (SH-2)
REPORT_COMMAND_RESPONSE = 0xF1
REPORT_COMMAND_REQUEST = 0xF2
REPORT_PRODUCT_ID_RESPONSE = 0xF8
REPORT_PRODUCT_ID_REQUEST = 0xF9
REPORT_TIMESTAMP_REBASE = 0xFA
REPORT_BASE_TIMESTAMP = 0xFB
REPORT_GET_FEATURE_RESPONSE = 0xFC
REPORT_SET_FEATURE = 0xFD

# commands (inside a command request)
CMD_SAVE_DCD = 0x06
CMD_ME_CALIBRATION = 0x07
ME_CAL_CONFIGURE = 0x00

# sensor report IDs
REPORT_ACCELEROMETER = 0x01  # calibrated, with gravity, m/s^2 Q8
REPORT_GYROSCOPE = 0x02  # calibrated, rad/s Q9
REPORT_MAGNETOMETER = 0x03  # calibrated, uT Q4
REPORT_LINEAR_ACCELERATION = 0x04
REPORT_ROTATION_VECTOR = 0x05
REPORT_GRAVITY = 0x06
REPORT_GYROSCOPE_UNCALIBRATED = 0x07
REPORT_GAME_ROTATION_VECTOR = 0x08
REPORT_GEOMAGNETIC_ROTATION_VECTOR = 0x09
REPORT_MAGNETOMETER_UNCALIBRATED = 0x0F

# Length of every report the chip may put in a packet we parse. A packet can
# carry several reports back to back, so an unknown ID ends the parse.
REPORT_LENGTHS = {
    REPORT_COMMAND_RESPONSE: 16,
    REPORT_PRODUCT_ID_RESPONSE: 16,
    REPORT_TIMESTAMP_REBASE: 5,
    REPORT_BASE_TIMESTAMP: 5,
    REPORT_GET_FEATURE_RESPONSE: 17,
    REPORT_ACCELEROMETER: 10,
    REPORT_GYROSCOPE: 10,
    REPORT_MAGNETOMETER: 10,
    REPORT_LINEAR_ACCELERATION: 10,
    REPORT_ROTATION_VECTOR: 14,
    REPORT_GRAVITY: 10,
    REPORT_GYROSCOPE_UNCALIBRATED: 16,
    REPORT_GAME_ROTATION_VECTOR: 12,
    REPORT_GEOMAGNETIC_ROTATION_VECTOR: 14,
    REPORT_MAGNETOMETER_UNCALIBRATED: 16,
}

ACC_SCALE = 2.0**-8  # Q8 -> m/s^2
GYRO_SCALE = 2.0**-9  # Q9 -> rad/s
MAG_SCALE = 2.0**-4  # Q4 -> uT

RESET_CAUSES = {0: "not applicable", 1: "power-on", 2: "internal", 3: "watchdog", 4: "external", 5: "other"}

# Largest single I2C read. Longer packets (the start-up advertisement is about
# 280 bytes) come in several reads, each with its own 4-byte header.
MAX_READ = 256

AXIS_X, AXIS_Y, AXIS_Z = 0, 1, 2
POSITIVE, NEGATIVE = 0, 1

# Same remaps as the BNO055 runtime (raw_imu.py in the Pi runtime): body axis
# i = sign_i * sensor axis axis_i. The BNO08x is built on the BNO055's
# silicon, so the chip axes should match, but a different breakout can sit
# the chip at another angle. ``calibrate.py -i imu`` checks the signs; set
# maixcam2.imu_axis_remap if they are wrong.
REMAP_UPRIGHT = (AXIS_Y, AXIS_X, AXIS_Z, NEGATIVE, POSITIVE, POSITIVE)
REMAP_UPSIDE_DOWN = (AXIS_Y, AXIS_X, AXIS_Z, NEGATIVE, NEGATIVE, NEGATIVE)


def remap_from_config(config) -> Tuple[int, ...]:
    """``maixcam2.imu_axis_remap`` if set, otherwise upright / upside down."""
    custom = config.maixcam2.get("imu_axis_remap")
    if custom:
        if len(custom) != 6 or any(int(v) not in (0, 1, 2) for v in custom[:3]) or any(int(v) not in (0, 1) for v in custom[3:]):
            raise ValueError(f"imu_axis_remap must be [x_axis, y_axis, z_axis, x_neg, y_neg, z_neg], got {custom}")
        return tuple(int(v) for v in custom)
    return REMAP_UPSIDE_DOWN if config.imu_upside_down else REMAP_UPRIGHT


def apply_remap(v: Sequence[float], remap) -> List[float]:
    axes, neg = remap[:3], remap[3:]
    return [(-v[axes[i]] if neg[i] else v[axes[i]]) for i in range(3)]


def header(length: int, channel: int, seq: int, continuation: bool = False) -> bytes:
    n = length | (0x8000 if continuation else 0)
    return struct.pack("<HBB", n, channel, seq & 0xFF)


def parse_header(b: bytes) -> Tuple[int, int, int, bool]:
    """(length, channel, seq, continuation). length includes the 4 header bytes."""
    n, ch, seq = struct.unpack_from("<HBB", b)
    return n & 0x7FFF, ch, seq, bool(n & 0x8000)


def set_feature_report(report_id: int, interval_us: int, batch_us: int = 0, sensitivity: int = 0, specific: int = 0) -> bytes:
    """SH-2 Set Feature Command (17 bytes)."""
    return struct.pack("<BBBHIII", REPORT_SET_FEATURE, report_id, 0, sensitivity, interval_us, batch_us, specific)


def command_request(seq: int, command: int, params: Sequence[int] = ()) -> bytes:
    """SH-2 Command Request (12 bytes): ID, seq, command, P0..P8."""
    p = list(params) + [0] * (9 - len(params))
    return bytes([REPORT_COMMAND_REQUEST, seq & 0xFF, command] + [v & 0xFF for v in p[:9]])


def split_reports(payload: bytes) -> List[bytes]:
    """Split a control / input-report packet payload into its reports."""
    out, i = [], 0
    while i < len(payload):
        n = REPORT_LENGTHS.get(payload[i])
        if n is None or i + n > len(payload):
            break
        out.append(payload[i : i + n])
        i += n
    return out


class LinuxI2C:
    """Raw /dev/i2c-N: each read()/write() is one I2C transaction."""

    I2C_SLAVE = 0x0703

    def __init__(self, bus: int):
        self.bus = bus
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        self._addr = None

    def _select(self, addr: int) -> None:
        if addr != self._addr:
            import fcntl

            fcntl.ioctl(self.fd, self.I2C_SLAVE, addr)
            self._addr = addr

    def read(self, addr: int, n: int) -> bytes:
        self._select(addr)
        b = os.read(self.fd, n)
        if len(b) != n:
            raise OSError(5, f"short I2C read ({len(b)} of {n})")
        return b

    def write(self, addr: int, data: bytes) -> None:
        self._select(addr)
        if os.write(self.fd, bytes(data)) != len(data):
            raise OSError(5, "short I2C write")

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class BNO08x:
    def __init__(self, bus: int, address: int = ADDRESS_B, i2c=None, reset: bool = True):
        """``i2c``: an object with read(addr, n) / write(addr, data) (the dry-run simulator)."""
        if address not in ADDRESSES:
            raise IOError(
                f"imu_address 0x{address:02x} is not a BNO08x address (0x4a or 0x4b); "
                "0x28/0x29 was the BNO055. Fix maixcam2.imu_address or set imu_i2c_bus to null to scan"
            )
        self.i2c = i2c if i2c is not None else LinuxI2C(bus)
        self.bus_number = bus
        self.address = address
        self._seq = [0] * 6
        self._cmd_seq = 0
        self.remap = (AXIS_X, AXIS_Y, AXIS_Z, POSITIVE, POSITIVE, POSITIVE)
        self.product: Optional[Dict] = None
        self.accuracy = {"accel": 0, "gyro": 0, "mag": 0}
        self.enabled: Dict[int, int] = {}  # report id -> interval us
        self.resets = 0
        self._acc: Optional[List[float]] = None
        self._gyro: Optional[List[float]] = None
        self._mag: Optional[List[float]] = None
        self._last_report = 0.0
        self._responses: Dict[int, Tuple[float, bytes]] = {}  # command -> (time, response)
        try:
            if reset:
                self.soft_reset()
            self.request_product_id()
        except OSError as e:
            self.close()
            raise IOError(f"no BNO08x on i2c-{bus} @0x{address:02x} ({e})") from e

    # SHTP transport -----------------------------------------------------------

    def _write_packet(self, channel: int, payload: bytes) -> None:
        self.i2c.write(self.address, header(len(payload) + 4, channel, self._seq[channel]) + bytes(payload))
        self._seq[channel] = (self._seq[channel] + 1) & 0xFF

    def _read_packet(self) -> Optional[Tuple[int, bytes]]:
        """One SHTP packet as (channel, payload), or None when the chip has nothing to send."""
        length, channel, _, _ = parse_header(self.i2c.read(self.address, 4))
        if length == 0 or length == 0x7FFF:  # nothing queued / chip not ready
            return None
        if length < 4 or channel > CH_GYRO_RV:
            raise OSError(5, f"bad SHTP header (length {length}, channel {channel})")
        remaining = length - 4
        payload = bytearray()
        while remaining > 0:
            n = min(remaining + 4, MAX_READ)
            chunk = self.i2c.read(self.address, n)
            payload += chunk[4:]
            remaining -= n - 4
        return channel, bytes(payload)

    # chip control -------------------------------------------------------------

    def soft_reset(self, timeout: float = 1.5) -> None:
        """Reset the chip, then read away the advertisement and the reset message."""
        self._write_packet(CH_EXECUTABLE, bytes([EXE_RESET]))
        self._seq = [0] * 6
        self._seq[CH_EXECUTABLE] = 1
        time.sleep(0.2)
        end = time.monotonic() + timeout
        seen_reset = False
        while time.monotonic() < end:
            try:
                pkt = self._read_packet()
            except OSError:
                pkt = None  # the chip NAKs while it boots
            if pkt is None:
                if seen_reset:
                    break
                time.sleep(0.01)
                continue
            ch, payload = pkt
            if ch == CH_EXECUTABLE and payload[:1] == bytes([EXE_RESET]):
                seen_reset = True
            elif ch in (CH_CONTROL, CH_REPORTS, CH_WAKE_REPORTS):
                self._handle(ch, payload)
        self.enabled.clear()
        self._acc = self._gyro = self._mag = None

    def request_product_id(self, timeout: float = 1.0) -> Dict:
        """Ask for the Product ID report: proves an SH-2 chip answers. Raises OSError if not."""
        self.product = None
        self._write_packet(CH_CONTROL, bytes([REPORT_PRODUCT_ID_REQUEST, 0]))
        end = time.monotonic() + timeout
        while self.product is None:
            if self.poll() == 0:
                if time.monotonic() > end:
                    raise OSError(110, "no Product ID response")
                time.sleep(0.005)
        return self.product

    def enable_report(self, report_id: int, interval_us: int) -> None:
        """Ask for a report every ``interval_us``; 0 turns it off."""
        self._write_packet(CH_CONTROL, set_feature_report(report_id, interval_us))
        if interval_us:
            self.enabled[report_id] = interval_us
        else:
            self.enabled.pop(report_id, None)

    def start(self, rate_hz: float = 100.0) -> None:
        """Stream calibrated accelerometer and gyroscope reports."""
        interval = int(round(1e6 / rate_hz))
        self.enable_report(REPORT_ACCELEROMETER, interval)
        self.enable_report(REPORT_GYROSCOPE, interval)
        self._last_report = time.monotonic()

    def set_axis_remap(self, remap) -> None:
        self.remap = tuple(remap)

    def _command(self, command: int, params: Sequence[int] = (), timeout: float = 1.5) -> bytes:
        t0 = time.monotonic()
        self._write_packet(CH_CONTROL, command_request(self._cmd_seq, command, params))
        self._cmd_seq = (self._cmd_seq + 1) & 0xFF
        while time.monotonic() - t0 < timeout:
            self.poll()
            got = self._responses.get(command)
            if got and got[0] >= t0:
                return got[1]
            time.sleep(0.005)
        raise OSError(110, f"no response to SH-2 command 0x{command:02x}")

    def configure_calibration(self, accel: bool, gyro: bool, mag: bool) -> None:
        """Turn the motion engine's dynamic calibration on or off per sensor."""
        r = self._command(CMD_ME_CALIBRATION, [int(accel), int(gyro), int(mag), ME_CAL_CONFIGURE, 0, 0])
        if r[5] != 0:
            raise OSError(5, f"ME calibration config refused (status {r[5]})")

    def begin_calibration(self, rate_hz: float = 50.0) -> None:
        """Calibrate accel, gyro and mag, and report all three so their accuracy shows."""
        self.configure_calibration(True, True, True)
        interval = int(round(1e6 / rate_hz))
        for rid in (REPORT_ACCELEROMETER, REPORT_GYROSCOPE, REPORT_MAGNETOMETER):
            self.enable_report(rid, interval)
        self._last_report = time.monotonic()

    def save_calibration(self) -> None:
        """Write the dynamic calibration data (DCD) to the chip's flash."""
        r = self._command(CMD_SAVE_DCD)
        if r[5] != 0:
            raise OSError(5, f"saving calibration failed (status {r[5]})")

    # incoming reports ------------------------------------------------------------

    def _handle(self, channel: int, payload: bytes) -> int:
        if channel == CH_EXECUTABLE:
            if payload[:1] == bytes([EXE_RESET]):
                # the chip reset on its own: reports stop until they are enabled again
                self.resets += 1
                for rid, interval in list(self.enabled.items()):
                    self._write_packet(CH_CONTROL, set_feature_report(rid, interval))
            return 0
        if channel not in (CH_CONTROL, CH_REPORTS, CH_WAKE_REPORTS):
            return 0
        n = 0
        for r in split_reports(payload):
            rid = r[0]
            if rid == REPORT_PRODUCT_ID_RESPONSE:
                if self.product is not None:
                    continue  # one response per firmware component; the first is SH-2
                self.product = {
                    "reset_cause": r[1],
                    "reset_cause_name": RESET_CAUSES.get(r[1], f"unknown {r[1]}"),
                    "sw_version": f"{r[2]}.{r[3]}.{struct.unpack_from('<H', r, 12)[0]}",
                    "part_number": struct.unpack_from("<I", r, 4)[0],
                    "build": struct.unpack_from("<I", r, 8)[0],
                }
            elif rid == REPORT_COMMAND_RESPONSE:
                self._responses[r[2]] = (time.monotonic(), r)
            elif rid in (REPORT_ACCELEROMETER, REPORT_GYROSCOPE, REPORT_MAGNETOMETER):
                x, y, z = struct.unpack_from("<3h", r, 4)
                acc = r[2] & 0x03
                if rid == REPORT_ACCELEROMETER:
                    self._acc = [x * ACC_SCALE, y * ACC_SCALE, z * ACC_SCALE]
                    self.accuracy["accel"] = acc
                elif rid == REPORT_GYROSCOPE:
                    self._gyro = [x * GYRO_SCALE, y * GYRO_SCALE, z * GYRO_SCALE]
                    self.accuracy["gyro"] = acc
                else:
                    self._mag = [x * MAG_SCALE, y * MAG_SCALE, z * MAG_SCALE]
                    self.accuracy["mag"] = acc
                self._last_report = time.monotonic()
                n += 1
        return n

    def poll(self, max_packets: int = 32) -> int:
        """Read every queued packet. Returns the number of packets read."""
        count = 0
        while count < max_packets:
            pkt = self._read_packet()
            if pkt is None:
                break
            self._handle(*pkt)
            count += 1
        return count

    # data -------------------------------------------------------------------

    def gyro_and_acceleration(self, stale_s: float = 0.5):
        """Latest (gyro rad/s, acceleration m/s^2), remapped to the body frame."""
        self.poll()
        if self._gyro is None or self._acc is None:
            end = time.monotonic() + 0.2
            while (self._gyro is None or self._acc is None) and time.monotonic() < end:
                time.sleep(0.005)
                self.poll()
            if self._gyro is None or self._acc is None:
                raise OSError(61, "BNO08x has not sent accel/gyro reports (start() not called?)")
        if time.monotonic() - self._last_report > stale_s:
            raise OSError(62, f"BNO08x reports stopped {time.monotonic() - self._last_report:.1f} s ago")
        return apply_remap(self._gyro, self.remap), apply_remap(self._acc, self.remap)

    def gyro(self):
        return self.gyro_and_acceleration()[0]

    def acceleration(self):
        return self.gyro_and_acceleration()[1]

    def calibration_status(self) -> Dict[str, int]:
        """Accuracy of the latest reports, 0 (unreliable) .. 3 (high)."""
        try:
            self.poll()
        except OSError:
            pass
        return dict(self.accuracy)

    def close(self) -> None:
        if self.i2c is not None and hasattr(self.i2c, "close"):
            self.i2c.close()


# calibration record ------------------------------------------------------------
#
# The calibration itself is in the chip. The file at maixcam2.imu_offsets_path
# only records that it was done, so diagnose.py can tell a calibrated duck
# from a fresh one.


def save_calibration_record(path: str, status: Dict[str, int], dev: Optional[BNO08x] = None) -> None:
    rec = {"bno08x_calibration_saved": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": status}
    if dev is not None and dev.product:
        rec["sw_version"] = dev.product["sw_version"]
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)


def load_calibration_record(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and "bno08x_calibration_saved" in rec else None


# discovery ---------------------------------------------------------------------


def probe(i2c, address: int) -> bool:
    """True if something at ``address`` answers a 4-byte read that could be an SHTP header.

    Cheap and not conclusive (an ADS1115 also lives at 0x4a/0x4b): opening the
    BNO08x asks for its Product ID, which is."""
    try:
        length, channel, _, _ = parse_header(i2c.read(address, 4))
    except OSError:
        return False
    return length == 0 or length == 0x7FFF or (length >= 4 and channel <= CH_GYRO_RV)


def scan_for_bno08x(buses=range(0, 8)) -> Optional[tuple]:
    """Return (bus, address) of the first BNO08x found, or None."""
    for b in buses:
        try:
            i2c = LinuxI2C(b)
        except OSError:
            continue
        try:
            for addr in ADDRESSES:
                if probe(i2c, addr):
                    return b, addr
        finally:
            i2c.close()
    return None


def open_from_config(config, scan=scan_for_bno08x, i2c=None) -> BNO08x:
    """Open the IMU named in config.maixcam2 (or scan), remapped and streaming."""
    m = config.maixcam2
    bus, addr = m["imu_i2c_bus"], m["imu_address"]
    if i2c is None and bus is None:
        found = scan()
        if found is None:
            raise IOError("no BNO08x answered at 0x4a/0x4b on /dev/i2c-0..7. Check wiring and the 'pinmap' config")
        bus, addr = found
    dev = BNO08x(7 if bus is None else bus, addr, i2c=i2c)
    dev.set_axis_remap(remap_from_config(config))
    return dev

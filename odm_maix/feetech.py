"""Pure-Python Feetech STS3215 bus driver (pyserial).

Replaces ``rustypot`` from the Raspberry Pi runtime so nothing needs to be
compiled for the MaixCAM2. Unit conversions match rustypot exactly, because
the walking policy was trained against them:

* position: rad = 2*pi*raw/4096 - pi            (raw 2048 -> 0 rad)
* velocity: rad/s = 2*pi*raw/4095, sign-magnitude on bit 15

Packet format (Feetech SCS/STS protocol v1)::

    0xFF 0xFF ID LEN INSTR PARAM... CHECKSUM
    LEN      = len(PARAM) + 2
    CHECKSUM = ~(ID + LEN + INSTR + sum(PARAM)) & 0xFF
"""

from __future__ import annotations

import math
import struct
from typing import Iterable, List, Sequence

# Instructions
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83

BROADCAST_ID = 0xFE

# STS3215 registers (EEPROM below 40, RAM from 40)
REG_ID = 5
REG_BAUD_RATE = 6  # 0 = 1 Mbps
REG_RETURN_DELAY = 7  # x2 us
REG_MIN_ANGLE_LIMIT = 9
REG_MAX_ANGLE_LIMIT = 11
REG_MAX_TEMPERATURE = 13
REG_MAX_VOLTAGE = 14
REG_MIN_VOLTAGE = 15
REG_MAX_TORQUE = 16
REG_P_COEFF = 21
REG_D_COEFF = 22
REG_I_COEFF = 23
REG_HOMING_OFFSET = 31
REG_MODE = 33  # 0 = position servo
REG_TORQUE_ENABLE = 40
REG_GOAL_POSITION = 42
REG_TORQUE_LIMIT = 48
REG_LOCK = 55  # 0 = EEPROM writable, 1 = locked
REG_PRESENT_POSITION = 56
REG_PRESENT_VELOCITY = 58
REG_PRESENT_VOLTAGE = 62
REG_PRESENT_TEMPERATURE = 63

# Error bits in a status packet
STATUS_ERRORS = {
    0x01: "voltage",
    0x02: "angle sensor",
    0x04: "overheat",
    0x08: "overcurrent",
    0x20: "overload",
}


def describe_status_error(error: int) -> str:
    return ", ".join(name for bit, name in STATUS_ERRORS.items() if error & bit) or ("0x%02x" % error if error else "")


class FeetechError(IOError):
    pass


# --- unit conversions (match rustypot sts3215) -----------------------------


def rad_to_raw_position(rad: float) -> int:
    raw = int(4096.0 * (math.pi + rad) / (2.0 * math.pi))
    return max(0, min(4095, raw))


def raw_to_rad_position(raw: int) -> float:
    raw = _from_sign_magnitude(raw, 15)
    return 2.0 * math.pi * raw / 4096.0 - math.pi


def raw_to_rad_velocity(raw: int) -> float:
    return 2.0 * math.pi * _from_sign_magnitude(raw, 15) / 4095.0


def _from_sign_magnitude(raw: int, sign_bit: int) -> int:
    if raw & (1 << sign_bit):
        return -(raw & ((1 << sign_bit) - 1))
    return raw


# --- packet helpers --------------------------------------------------------


def checksum(body: Iterable[int]) -> int:
    return (~sum(body)) & 0xFF


def build_packet(servo_id: int, instruction: int, params: Sequence[int] = ()) -> bytes:
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes([0xFF, 0xFF, *body, checksum(body)])


def build_sync_write(address: int, data_len: int, id_data: dict) -> bytes:
    params = [address, data_len]
    for sid, data in id_data.items():
        if len(data) != data_len:
            raise ValueError("every servo needs data_len bytes")
        params.append(sid)
        params.extend(data)
    return build_packet(BROADCAST_ID, INST_SYNC_WRITE, params)


def build_sync_read(address: int, data_len: int, ids: Sequence[int]) -> bytes:
    return build_packet(BROADCAST_ID, INST_SYNC_READ, [address, data_len, *ids])


def parse_status(buf: bytes, offset: int = 0):
    """Parse one status packet starting at ``offset``.

    Returns ``(servo_id, error, params, next_offset)``.
    """
    i = buf.find(b"\xff\xff", offset)
    if i < 0 or i + 5 > len(buf):
        raise FeetechError("status packet header not found")
    sid, length, error = buf[i + 2], buf[i + 3], buf[i + 4]
    end = i + 3 + length  # index of checksum byte (LEN counts error + params + checksum)
    if end >= len(buf):
        raise FeetechError("truncated status packet")
    params = buf[i + 5 : end]
    if checksum(buf[i + 2 : end]) != buf[end]:
        raise FeetechError(f"bad checksum from servo {sid}")
    return sid, error, bytes(params), end + 1


class FeetechBus:
    """Feetech STS bus on a serial port (USB adapter or a MaixCAM2 UART)."""

    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.02, ser=None):
        """``ser``: an already open serial-like object (used by the dry-run simulator)."""
        if ser is None:
            import serial  # pyserial; imported lazily so tests run without it

            ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.ser = ser

    # low level -------------------------------------------------------------

    def _txrx(self, packet: bytes, expected_len: int) -> bytes:
        self.ser.reset_input_buffer()
        self.ser.write(packet)
        self.ser.flush()
        if expected_len == 0:
            return b""
        data = self.ser.read(expected_len)
        if len(data) < expected_len:
            raise FeetechError(f"timeout: got {len(data)}/{expected_len} bytes")
        return data

    def write(self, servo_id: int, address: int, data: Sequence[int]) -> None:
        self._txrx(build_packet(servo_id, INST_WRITE, [address, *data]), 6)

    def read(self, servo_id: int, address: int, length: int) -> bytes:
        return self.read_with_status(servo_id, address, length)[1]

    def read_with_status(self, servo_id: int, address: int, length: int):
        """(status error byte, data). The error byte flags overload, overheat, etc."""
        resp = self._txrx(build_packet(servo_id, INST_READ, [address, length]), 6 + length)
        _, error, params, _ = parse_status(resp)
        return error, params

    def read_u8(self, servo_id: int, address: int) -> int:
        return self.read(servo_id, address, 1)[0]

    def read_u16(self, servo_id: int, address: int) -> int:
        return struct.unpack("<H", self.read(servo_id, address, 2))[0]

    def write_eeprom(self, servo_id: int, address: int, data: Sequence[int]) -> None:
        """Write a persistent setting: unlock, write, lock again."""
        self.write(servo_id, REG_LOCK, [0])
        try:
            self.write(servo_id, address, data)
        finally:
            # after an ID change the servo answers on the new ID
            new_id = data[0] if address == REG_ID else servo_id
            self.write(new_id, REG_LOCK, [1])

    def scan(self, ids: Iterable[int] = range(0, 254)) -> List[int]:
        return [i for i in ids if self.ping(i)]

    def ping(self, servo_id: int) -> bool:
        try:
            self._txrx(build_packet(servo_id, INST_PING), 6)
            return True
        except FeetechError:
            return False

    def sync_write(self, address: int, data_len: int, id_data: dict) -> None:
        self._txrx(build_sync_write(address, data_len, id_data), 0)

    def sync_read(self, address: int, data_len: int, ids: Sequence[int]) -> List[bytes]:
        resp = self._txrx(build_sync_read(address, data_len, ids), (6 + data_len) * len(ids))
        out = {}
        offset = 0
        for _ in ids:
            sid, _, params, offset = parse_status(resp, offset)
            out[sid] = params
        try:
            return [out[i] for i in ids]
        except KeyError as e:
            raise FeetechError(f"no answer from servo {e}") from None

    # high level (same surface as rustypot.feetech) ----------------------------

    def set_kps(self, ids: Sequence[int], kps: Sequence[float]) -> None:
        self.sync_write(REG_P_COEFF, 1, {i: [int(k) & 0xFF] for i, k in zip(ids, kps)})

    def set_kds(self, ids: Sequence[int], kds: Sequence[float]) -> None:
        self.sync_write(REG_D_COEFF, 1, {i: [int(k) & 0xFF] for i, k in zip(ids, kds)})

    def enable_torque(self, ids: Sequence[int]) -> None:
        self.sync_write(REG_TORQUE_ENABLE, 1, {i: [1] for i in ids})

    def disable_torque(self, ids: Sequence[int]) -> None:
        self.sync_write(REG_TORQUE_ENABLE, 1, {i: [0] for i in ids})

    def write_goal_position(self, ids: Sequence[int], positions_rad: Sequence[float]) -> None:
        data = {i: list(struct.pack("<H", rad_to_raw_position(p))) for i, p in zip(ids, positions_rad)}
        self.sync_write(REG_GOAL_POSITION, 2, data)

    def read_present_position(self, ids: Sequence[int]) -> List[float]:
        raws = self.sync_read(REG_PRESENT_POSITION, 2, ids)
        return [raw_to_rad_position(struct.unpack("<H", r)[0]) for r in raws]

    def read_present_velocity(self, ids: Sequence[int]) -> List[float]:
        raws = self.sync_read(REG_PRESENT_VELOCITY, 2, ids)
        return [raw_to_rad_velocity(struct.unpack("<H", r)[0]) for r in raws]

    def read_present_position_velocity(self, ids: Sequence[int]):
        """Position and velocity in one bus transaction (registers 56..59)."""
        raws = self.sync_read(REG_PRESENT_POSITION, 4, ids)
        pos, vel = [], []
        for r in raws:
            p, v = struct.unpack("<HH", r)
            pos.append(raw_to_rad_position(p))
            vel.append(raw_to_rad_velocity(v))
        return pos, vel

    def read_voltage(self, servo_id: int) -> float:
        return self.read(servo_id, REG_PRESENT_VOLTAGE, 1)[0] / 10.0

    def read_temperature(self, servo_id: int) -> int:
        return self.read(servo_id, REG_PRESENT_TEMPERATURE, 1)[0]

    def read_raw_position(self, ids: Sequence[int]) -> List[int]:
        """Encoder counts 0..4095, no unit conversion."""
        return [struct.unpack("<H", r)[0] & 0x7FFF for r in self.sync_read(REG_PRESENT_POSITION, 2, ids)]

    def close(self) -> None:
        self.ser.close()

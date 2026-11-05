"""Minimal BNO055 driver over Linux I2C (smbus2).

The Pi runtime used ``adafruit_bno055`` through Adafruit Blinka, which does
not know the MaixCAM2 board. This driver talks to ``/dev/i2c-N`` directly and
returns the same units as the Adafruit library:

* gyro in rad/s  (raw LSB = 1/16 deg/s)
* acceleration in m/s^2 (raw LSB = 1/100 m/s^2)
"""

from __future__ import annotations

import json
import math
import struct
import time
from typing import Dict, List, Optional

ADDRESS_A = 0x28
ADDRESS_B = 0x29

REG_CHIP_ID = 0x00
REG_PAGE_ID = 0x07  # same address on both register pages
REG_ACC_DATA = 0x08
REG_GYR_DATA = 0x14
REG_CALIB_STAT = 0x35
REG_ST_RESULT = 0x36  # power-on self test: bit0 acc, bit1 mag, bit2 gyro, bit3 MCU (1 = pass)
REG_SYS_STATUS = 0x39
REG_SYS_ERR = 0x3A
REG_UNIT_SEL = 0x3B
REG_OPR_MODE = 0x3D
REG_PWR_MODE = 0x3E
REG_SYS_TRIGGER = 0x3F
REG_AXIS_MAP_CONFIG = 0x41
REG_AXIS_MAP_SIGN = 0x42
REG_OFFSETS = 0x55  # 22 bytes: acc(6) mag(6) gyro(6) acc_radius(2) mag_radius(2)

CHIP_ID = 0xA0

CONFIG_MODE = 0x00
ACCGYRO_MODE = 0x05
IMUPLUS_MODE = 0x08
NDOF_MODE = 0x0C

AXIS_X, AXIS_Y, AXIS_Z = 0, 1, 2
POSITIVE, NEGATIVE = 0, 1

GYRO_SCALE = math.radians(1.0 / 16.0)
ACC_SCALE = 1.0 / 100.0


def axis_remap_bytes(x: int, y: int, z: int, x_sign: int, y_sign: int, z_sign: int):
    """Same packing as adafruit_bno055's ``axis_remap`` setter."""
    config = (z << 4) | (y << 2) | x
    sign = (x_sign << 2) | (y_sign << 1) | z_sign
    return config, sign


# Remaps used by the original runtime (raw_imu.py)
REMAP_UPRIGHT = (AXIS_Y, AXIS_X, AXIS_Z, NEGATIVE, POSITIVE, POSITIVE)
REMAP_UPSIDE_DOWN = (AXIS_Y, AXIS_X, AXIS_Z, NEGATIVE, NEGATIVE, NEGATIVE)


class BNO055:
    def __init__(self, bus: int, address: int = ADDRESS_A, smbus=None):
        """``smbus``: an open SMBus-like object (used by the dry-run simulator)."""
        if smbus is None:
            from smbus2 import SMBus

            smbus = SMBus(bus)
        self.bus = smbus
        self.bus_number = bus
        self.address = address
        self._select_page0()
        chip = self._read_u8(REG_CHIP_ID)
        if chip != CHIP_ID:
            # BNO055 can need up to ~850 ms after power-on
            time.sleep(1.0)
            self._select_page0()
            chip = self._read_u8(REG_CHIP_ID)
            if chip != CHIP_ID:
                raise IOError(f"BNO055 not found on i2c-{bus} @0x{address:02x} (chip id 0x{chip:02x})")
        self.mode = CONFIG_MODE
        self._write_u8(REG_PWR_MODE, 0x00)  # normal power
        self._write_u8(REG_SYS_TRIGGER, 0x00)  # internal oscillator
        self._write_u8(REG_UNIT_SEL, 0x00)  # m/s^2, deg/s, deg, Celsius, Windows orientation

    # register access --------------------------------------------------------

    def _select_page0(self) -> None:
        # Every register used here is on page 0; a chip left on page 1 would misread all of them.
        try:
            self._write_u8(REG_PAGE_ID, 0x00)
        except OSError:
            pass  # not powered up yet, the CHIP_ID retry handles it

    def _read_u8(self, reg: int) -> int:
        return self.bus.read_byte_data(self.address, reg)

    def _write_u8(self, reg: int, val: int) -> None:
        self.bus.write_byte_data(self.address, reg, val & 0xFF)

    def _read_block(self, reg: int, n: int) -> bytes:
        return bytes(self.bus.read_i2c_block_data(self.address, reg, n))

    # configuration ----------------------------------------------------------

    @property
    def mode(self) -> int:
        return self._read_u8(REG_OPR_MODE) & 0x0F

    @mode.setter
    def mode(self, value: int) -> None:
        self._write_u8(REG_OPR_MODE, CONFIG_MODE)
        time.sleep(0.025)
        if value != CONFIG_MODE:
            self._write_u8(REG_OPR_MODE, value)
            time.sleep(0.01)

    def set_axis_remap(self, remap) -> None:
        config, sign = axis_remap_bytes(*remap)
        current = self.mode
        self.mode = CONFIG_MODE
        self._write_u8(REG_AXIS_MAP_CONFIG, config)
        self._write_u8(REG_AXIS_MAP_SIGN, sign)
        self.mode = current

    def calibration_status(self) -> Dict[str, int]:
        s = self._read_u8(REG_CALIB_STAT)
        return {"sys": (s >> 6) & 3, "gyro": (s >> 4) & 3, "accel": (s >> 2) & 3, "mag": s & 3}

    def self_test(self) -> Dict[str, bool]:
        """Result of the power-on self test (True = passed)."""
        r = self._read_u8(REG_ST_RESULT)
        return {"accel": bool(r & 1), "mag": bool(r & 2), "gyro": bool(r & 4), "mcu": bool(r & 8)}

    def system_status(self):
        """(SYS_STATUS, SYS_ERR). Status 5 = fusion running, 6 = running without fusion, 1 = error."""
        return self._read_u8(REG_SYS_STATUS), self._read_u8(REG_SYS_ERR)

    def read_offsets(self) -> List[int]:
        current = self.mode
        self.mode = CONFIG_MODE
        data = list(struct.unpack("<11h", self._read_block(REG_OFFSETS, 22)))
        self.mode = current
        return data

    def write_offsets(self, offsets: List[int]) -> None:
        current = self.mode
        self.mode = CONFIG_MODE
        raw = struct.pack("<11h", *offsets)
        for i, b in enumerate(raw):
            self._write_u8(REG_OFFSETS + i, b)
        self.mode = current

    def save_offsets(self, path: str) -> None:
        json.dump({"bno055_offsets": self.read_offsets()}, open(path, "w"), indent=2)

    def load_offsets(self, path: str) -> bool:
        try:
            offsets = json.load(open(path))["bno055_offsets"]
        except (OSError, KeyError, ValueError):
            return False
        self.write_offsets(offsets)
        return True

    # data -------------------------------------------------------------------

    def gyro(self):
        x, y, z = struct.unpack("<3h", self._read_block(REG_GYR_DATA, 6))
        return [x * GYRO_SCALE, y * GYRO_SCALE, z * GYRO_SCALE]

    def acceleration(self):
        x, y, z = struct.unpack("<3h", self._read_block(REG_ACC_DATA, 6))
        return [x * ACC_SCALE, y * ACC_SCALE, z * ACC_SCALE]

    def gyro_and_acceleration(self):
        """One 18-byte burst read: acc(0x08) mag(0x0E) gyro(0x14)."""
        v = struct.unpack("<9h", self._read_block(REG_ACC_DATA, 18))
        acc = [v[0] * ACC_SCALE, v[1] * ACC_SCALE, v[2] * ACC_SCALE]
        gyro = [v[6] * GYRO_SCALE, v[7] * GYRO_SCALE, v[8] * GYRO_SCALE]
        return gyro, acc

    def close(self) -> None:
        self.bus.close()


def scan_for_bno055(buses=range(0, 8)) -> Optional[tuple]:
    """Return (bus, address) of the first BNO055 found, or None."""
    from smbus2 import SMBus

    for b in buses:
        try:
            with SMBus(b) as bus:
                for addr in (ADDRESS_A, ADDRESS_B):
                    try:
                        if bus.read_byte_data(addr, REG_CHIP_ID) == CHIP_ID:
                            return b, addr
                    except OSError:
                        pass
        except (OSError, FileNotFoundError):
            pass
    return None

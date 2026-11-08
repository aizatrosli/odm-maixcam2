"""BNO08x driver: SHTP framing, report parsing, remap, against the wire-level simulator."""

import struct

import numpy as np
import pytest

from odm_maix import bno08x
from odm_maix.config import DuckConfig
from odm_maix.diag.simhw import VirtualBNO08xBus


def test_scales_match_sh2_q_points():
    assert bno08x.ACC_SCALE == 1 / 256  # accelerometer Q8
    assert bno08x.GYRO_SCALE == 1 / 512  # gyroscope Q9


def test_header_round_trip():
    h = bno08x.header(21, bno08x.CH_CONTROL, 7, continuation=True)
    assert h == bytes([21, 0x80, 2, 7])
    assert bno08x.parse_header(h) == (21, 2, 7, True)


def test_set_feature_layout():
    r = bno08x.set_feature_report(bno08x.REPORT_GYROSCOPE, 10000)
    assert len(r) == 17
    assert r[0] == 0xFD and r[1] == 0x02
    assert struct.unpack_from("<I", r, 5)[0] == 10000


def test_command_request_layout():
    r = bno08x.command_request(3, bno08x.CMD_ME_CALIBRATION, [1, 1, 1, 0])
    assert r == bytes([0xF2, 3, 0x07, 1, 1, 1, 0, 0, 0, 0, 0, 0])


def test_split_reports_stops_at_unknown_id():
    acc = struct.pack("<BBBB3h", 0x01, 0, 3, 0, 0, 0, 2509)
    gyro = struct.pack("<BBBB3h", 0x02, 0, 2, 0, 1, 2, 3)
    ts = struct.pack("<BI", 0xFB, 0)
    parts = bno08x.split_reports(ts + acc + gyro + b"\x77\x00")
    assert [p[0] for p in parts] == [0xFB, 0x01, 0x02]


def test_remap_matches_bno055_axis_remap():
    v = [1.0, 2.0, 3.0]
    # BNO055 AXIS_MAP_CONFIG 0x21 / SIGN 0x04: x = -y, y = x, z = z
    assert bno08x.apply_remap(v, bno08x.REMAP_UPRIGHT) == [-2.0, 1.0, 3.0]
    assert bno08x.apply_remap(v, bno08x.REMAP_UPSIDE_DOWN) == [-2.0, -1.0, -3.0]


def test_remap_from_config():
    c = DuckConfig()
    assert bno08x.remap_from_config(c) == bno08x.REMAP_UPRIGHT
    c.imu_upside_down = True
    assert bno08x.remap_from_config(c) == bno08x.REMAP_UPSIDE_DOWN
    c.maixcam2["imu_axis_remap"] = [0, 1, 2, 0, 0, 1]
    assert bno08x.remap_from_config(c) == (0, 1, 2, 0, 0, 1)
    c.maixcam2["imu_axis_remap"] = [0, 1, 3, 0, 0, 1]
    with pytest.raises(ValueError):
        bno08x.remap_from_config(c)


def test_rejects_bno055_address():
    with pytest.raises(IOError, match="0x28"):
        bno08x.BNO08x(7, 0x28, i2c=VirtualBNO08xBus())


def test_wrong_address_is_ioerror():
    with pytest.raises(IOError, match="no BNO08x"):
        bno08x.BNO08x(7, bno08x.ADDRESS_A, i2c=VirtualBNO08xBus(address=bno08x.ADDRESS_B))


def test_long_packets_are_read_in_chunks(monkeypatch):
    monkeypatch.setattr(bno08x, "MAX_READ", 32)  # the 280-byte advertisement now takes 10 reads
    dev = bno08x.BNO08x(7, bno08x.ADDRESS_B, i2c=VirtualBNO08xBus())
    assert dev.product["sw_version"] == "3.12.7"
    assert dev.product["reset_cause_name"] == "power-on"


def test_upright_reads_gravity_on_plus_z():
    bus = VirtualBNO08xBus()
    dev = bno08x.BNO08x(7, bus.address, i2c=bus)
    dev.set_axis_remap(bno08x.REMAP_UPRIGHT)
    dev.start(200)
    gyro, acc = dev.gyro_and_acceleration()
    assert acc == pytest.approx([0, 0, 9.81], abs=0.2)
    assert np.abs(gyro).max() < 0.02
    bus.set_pose(0.0, np.radians(30))  # nose down
    for _ in range(3):
        bus._next_due = {k: 0 for k in bus._next_due}
        gyro, acc = dev.gyro_and_acceleration()
    assert acc[0] < -4


def test_reenables_reports_after_chip_reset():
    bus = VirtualBNO08xBus()
    dev = bno08x.BNO08x(7, bus.address, i2c=bus)
    dev.start(100)
    dev.gyro_and_acceleration()
    bus._boot()  # brown-out: the chip forgets its reports
    dev.poll()
    assert dev.resets == 1
    assert set(bus.enabled) == {bno08x.REPORT_ACCELEROMETER, bno08x.REPORT_GYROSCOPE}


def test_open_from_config_uses_scan_result():
    bus = VirtualBNO08xBus()
    dev = bno08x.open_from_config(DuckConfig(), i2c=bus)
    assert dev.address == bno08x.ADDRESS_B and dev.remap == bno08x.REMAP_UPRIGHT

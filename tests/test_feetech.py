import math
import struct

import pytest

from odm_maix import feetech as ft


def test_ping_packet():
    # Known-good Feetech ping to id 1: FF FF 01 02 01 FB
    assert ft.build_packet(1, ft.INST_PING) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])


def test_position_conversion_matches_rustypot():
    assert ft.rad_to_raw_position(0.0) == 2048
    assert ft.raw_to_rad_position(2048) == pytest.approx(0.0)
    for rad in (-1.2, -0.63, 0.0, 0.635, 1.379):
        assert ft.raw_to_rad_position(ft.rad_to_raw_position(rad)) == pytest.approx(rad, abs=2 * math.pi / 4096)


def test_velocity_sign_magnitude():
    assert ft.raw_to_rad_velocity(100) == pytest.approx(2 * math.pi * 100 / 4095)
    assert ft.raw_to_rad_velocity((1 << 15) | 100) == pytest.approx(-2 * math.pi * 100 / 4095)


def test_sync_write_layout():
    pkt = ft.build_sync_write(ft.REG_GOAL_POSITION, 2, {10: [0x00, 0x08], 11: [0x01, 0x08]})
    assert pkt[:2] == b"\xff\xff"
    assert pkt[2] == ft.BROADCAST_ID
    assert pkt[4] == ft.INST_SYNC_WRITE
    assert list(pkt[5:7]) == [ft.REG_GOAL_POSITION, 2]
    assert list(pkt[7:13]) == [10, 0x00, 0x08, 11, 0x01, 0x08]
    assert pkt[3] == len(pkt) - 4
    assert pkt[-1] == ft.checksum(pkt[2:-1])


def status(sid, params, error=0):
    body = [sid, len(params) + 2, error, *params]
    return bytes([0xFF, 0xFF, *body, ft.checksum(body)])


class FakeSerial:
    def __init__(self, reply):
        self.reply = reply
        self.written = b""

    def reset_input_buffer(self):
        pass

    def write(self, b):
        self.written += b

    def flush(self):
        pass

    def read(self, n):
        out, self.reply = self.reply[:n], self.reply[n:]
        return out


def make_bus(reply):
    bus = ft.FeetechBus.__new__(ft.FeetechBus)
    bus.ser = FakeSerial(reply)
    return bus


def test_sync_read_position_velocity():
    ids = [20, 21]
    reply = status(20, struct.pack("<HH", 2048, 100)) + status(21, struct.pack("<HH", 3072, (1 << 15) | 50))
    pos, vel = make_bus(reply).read_present_position_velocity(ids)
    assert pos[0] == pytest.approx(0.0)
    assert pos[1] == pytest.approx(math.pi / 2)
    assert vel[0] > 0 and vel[1] < 0


def test_sync_read_timeout():
    with pytest.raises(ft.FeetechError):
        make_bus(status(20, b"\x00\x08\x00\x00")).read_present_position_velocity([20, 21])


def test_bad_checksum():
    bad = bytearray(status(20, b"\x00\x08\x00\x00"))
    bad[-1] ^= 0xFF
    with pytest.raises(ft.FeetechError):
        make_bus(bytes(bad)).read_present_position_velocity([20])

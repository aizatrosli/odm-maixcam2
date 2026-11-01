import math

import numpy as np
import pytest

from odm_maix.commands import FollowCommands
from odm_maix.config import DuckConfig
from odm_maix.vision.follow import VX_MAX, Detection, FollowController, FollowParams, pick_target, pixel_to_angle


def run(ctrl, det, seconds, t0=0.0, dt=0.02):
    t = t0
    cmd = None
    while t < t0 + seconds:
        cmd = ctrl.update([det] if det else [], t, dt)
        t += dt
    return cmd, t


def test_pixel_to_angle():
    assert pixel_to_angle(0.5, 70) == 0
    assert pixel_to_angle(1.0, 70) == pytest.approx(math.radians(35))
    assert pixel_to_angle(0.0, 70) == pytest.approx(-math.radians(35))


def test_person_ahead_and_far_walks_forward():
    c = FollowController()
    cmd, _ = run(c, Detection(0.5, 0.5, 0.1, 0.25), 2.0)
    assert c.state == "following"
    assert 0 < cmd[0] <= VX_MAX
    assert abs(cmd[2]) < 1e-6


def test_person_on_right_turns_right_without_walking():
    c = FollowController()
    cmd, _ = run(c, Detection(0.95, 0.5, 0.1, 0.3), 2.0)
    assert c.state == "turning"
    assert cmd[2] < 0  # clockwise
    assert cmd[0] == pytest.approx(0.0)
    assert cmd[5] < 0  # head looks right too


def test_too_close_stops():
    c = FollowController()
    run(c, Detection(0.5, 0.5, 0.1, 0.25), 2.0)
    cmd, _ = run(c, Detection(0.5, 0.5, 0.4, 0.9), 2.0, t0=2.0)
    assert c.state == "too_close"
    assert cmd[0] == pytest.approx(0.0)


def test_lost_target_ramps_down():
    c = FollowController()
    cmd, t = run(c, Detection(0.5, 0.5, 0.1, 0.25), 2.0)
    assert cmd[0] > 0
    cmd, _ = run(c, None, 2.0, t0=t)
    assert c.state == "searching"
    np.testing.assert_allclose(cmd, 0.0, atol=1e-9)


def test_commands_are_slew_limited():
    p = FollowParams()
    c = FollowController(p)
    cmd = c.update([Detection(0.0, 0.5, 0.1, 0.3)], 0.0, 0.02)
    assert abs(cmd[2]) <= p.max_wz_rate * 0.02 + 1e-9


def test_camera_on_head_compensates_head_yaw():
    # head already turned right by 0.3 rad (cmd -0.3), person at image centre
    p = FollowParams(camera_on_head=True)
    c = FollowController(p)
    c.cmd[5] = -0.3
    c.update([Detection(0.5, 0.5, 0.1, 0.3)], 0.0, 0.02)
    assert c.cmd[2] < 0  # body turns right toward where the head is looking


def test_pick_target_keeps_same_person():
    a = Detection(0.2, 0.5, 0.1, 0.3)
    b = Detection(0.8, 0.5, 0.15, 0.4)
    assert pick_target([a, b], None) is b  # biggest first
    assert pick_target([a, b], Detection(0.22, 0.5, 0.1, 0.3)) is a  # then continuity


class FakeDetector:
    def __init__(self):
        self.dets = []
        self.seq = 0

    def latest(self):
        self.seq += 1
        return True, 0.0, list(self.dets)

    def stop(self):
        pass


class FakePad:
    def __init__(self, cmd, events=None):
        self.cmd = np.array(cmd, dtype=float)
        self.events = events or {}

    def get(self):
        return self.cmd.copy(), dict(self.events)

    def stop(self):
        pass


def test_follow_commands_uses_detector_and_gamepad_override():
    clock = iter(np.arange(0, 10, 0.02))
    det = FakeDetector()
    det.dets = [Detection(0.5, 0.5, 0.1, 0.25)]
    vision = DuckConfig().maixcam2["vision"]
    fc = FollowCommands(vision, gamepad=FakePad(np.zeros(7)), detector=det, clock=lambda: next(clock))
    for _ in range(100):
        cmd, _ = fc.get()
    assert cmd[0] > 0

    fc.gamepad = FakePad([-0.1, 0, 0.5, 0, 0, 0, 0], {"A": True})
    cmd, events = fc.get()
    assert cmd[0] == pytest.approx(-0.1) and cmd[2] == pytest.approx(0.5)
    assert events["A"]


def test_vision_config_merges_defaults(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"maixcam2": {"vision": {"show": true}}}')
    v = DuckConfig.load(str(f)).maixcam2["vision"]
    assert v["show"] is True and v["class_name"] == "person"
